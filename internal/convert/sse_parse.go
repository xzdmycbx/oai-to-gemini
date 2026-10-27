package convert

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// PayloadSink is the spill-capable destination for large SSE payload fields.
// Size returns the number of bytes written so far, so the parser can record
// ranges without retaining the payload in memory.
type PayloadSink interface {
	io.Writer
	Size() int64
}

type discardPayload struct {
	size int64
}

func (d *discardPayload) Write(p []byte) (int, error) {
	d.size += int64(len(p))
	return len(p), nil
}

func (d *discardPayload) Size() int64 { return d.size }

// ParseGeminiSSEPayload decodes one Gemini SSE event. Large text and inlineData
// string values are decoded incrementally into payload instead of being
// materialized as Go strings. The returned parts point at StoredRange values;
// callers that need the bytes can stream them from the same payload store.
func ParseGeminiSSEPayload(r io.Reader, payload PayloadSink) (*GeminiResponse, error) {
	if payload == nil {
		payload = &discardPayload{}
	}
	p := &geminiSSEParser{
		br:      bufio.NewReaderSize(r, 64*1024),
		payload: payload,
	}
	if err := p.skipSpace(); err != nil {
		if err == io.EOF {
			return nil, nil
		}
		return nil, err
	}
	b, err := p.br.Peek(1)
	if err == io.EOF {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if b[0] == 'n' {
		null, err := p.consumeNull()
		if err != nil {
			return nil, err
		}
		if null {
			if err := p.ensureEOF(); err != nil {
				return nil, err
			}
			return &GeminiResponse{}, nil
		}
	}
	if b[0] != '{' {
		return nil, fmt.Errorf("expected JSON object")
	}
	resp, err := p.parseTop()
	if err != nil {
		return nil, err
	}
	if err := p.ensureEOF(); err != nil {
		return nil, err
	}
	return resp, nil
}

const (
	maxJSONNestingDepth = 1000

	// The SSE event itself may be arbitrarily large because text and inlineData
	// payloads spill to PayloadSink. Everything else is retained as in-memory
	// metadata, so it needs explicit count and byte budgets. Without these a
	// shallow but extremely wide event could allocate millions of small
	// structs while the raw event stays relatively small.
	maxSSECandidates        = 256
	maxSSEParts             = 16384
	maxSSEPartUnknownFields = 1024
	maxSSEUnknownFields     = 65536
	maxSSEMetadataBytes     = 64 << 20
	maxSSELiteralBytes      = 1 << 20
	maxSSEIntegerBytes      = 32
	maxSSEBoolBytes         = 16
)

const (
	candidateMetadataOverhead = 128
	partMetadataOverhead      = 256
)

type geminiSSEParser struct {
	br      *bufio.Reader
	payload PayloadSink
	depth   int

	candidates    int
	parts         int
	unknownFields int
	metadataBytes int64
}

func (p *geminiSSEParser) chargeMetadata(n int64) error {
	if n < 0 {
		return fmt.Errorf("invalid SSE metadata size")
	}
	if n > maxSSEMetadataBytes-p.metadataBytes {
		return fmt.Errorf("SSE event metadata exceeds %d bytes", maxSSEMetadataBytes)
	}
	p.metadataBytes += n
	return nil
}

type metadataWriter struct {
	p *geminiSSEParser
	w io.Writer
}

func (m metadataWriter) Write(b []byte) (int, error) {
	if err := m.p.chargeMetadata(int64(len(b))); err != nil {
		return 0, err
	}
	return m.w.Write(b)
}

func (p *geminiSSEParser) enterNesting() error {
	p.depth++
	if p.depth > maxJSONNestingDepth {
		p.depth--
		return fmt.Errorf("JSON nesting depth exceeds %d", maxJSONNestingDepth)
	}
	return nil
}

func (p *geminiSSEParser) leaveNesting() {
	p.depth--
}

func (p *geminiSSEParser) skipSpace() error {
	for {
		b, err := p.br.ReadByte()
		if err != nil {
			return err
		}
		switch b {
		case ' ', '\t', '\n', '\r':
			continue
		default:
			return p.br.UnreadByte()
		}
	}
}

func (p *geminiSSEParser) expect(c byte) error {
	if err := p.skipSpace(); err != nil {
		return err
	}
	got, err := p.br.ReadByte()
	if err != nil {
		return err
	}
	if got != c {
		return fmt.Errorf("expected %q, got %q", c, got)
	}
	return nil
}

func (p *geminiSSEParser) ensureEOF() error {
	if err := p.skipSpace(); err != nil {
		if err == io.EOF {
			return nil
		}
		return err
	}
	if _, err := p.br.Peek(1); err == io.EOF {
		return nil
	} else if err != nil {
		return err
	}
	return fmt.Errorf("unexpected data after JSON value")
}

func (p *geminiSSEParser) parseTop() (*GeminiResponse, error) {
	resp := &GeminiResponse{}
	if err := p.expect('{'); err != nil {
		return nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return resp, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return nil, err
		}
		if err := p.expect(':'); err != nil {
			return nil, err
		}
		switch key {
		case "candidates":
			null, err := p.consumeNull()
			if err != nil {
				return nil, err
			}
			if null {
				resp.Candidates = nil
				break
			}
			resp.Candidates, err = p.parseCandidates()
			if err != nil {
				return nil, err
			}
		case "usageMetadata":
			null, err := p.consumeNull()
			if err != nil {
				return nil, err
			}
			if null {
				resp.UsageMetadata = nil
				break
			}
			resp.UsageMetadata, err = p.parseUsage()
			if err != nil {
				return nil, err
			}
		case "promptFeedback":
			null, err := p.consumeNull()
			if err != nil {
				return nil, err
			}
			if null {
				resp.PromptFeedback = nil
				break
			}
			resp.PromptFeedback, err = p.parsePromptFeedback()
			if err != nil {
				return nil, err
			}
		case "modelVersion":
			if err := p.parseStringIfPresent(&resp.ModelVersion); err != nil {
				return nil, err
			}
		case "responseId":
			if err := p.parseStringIfPresent(&resp.ResponseID); err != nil {
				return nil, err
			}
		case "error":
			raw, err := p.parseRawValue()
			if err != nil {
				return nil, err
			}
			resp.Error = raw
		default:
			if err := p.skipValue(); err != nil {
				return nil, err
			}
		}
		done, err := p.afterField()
		if err != nil {
			return nil, err
		}
		if done {
			return resp, nil
		}
	}
}

func (p *geminiSSEParser) afterField() (bool, error) {
	if err := p.skipSpace(); err != nil {
		return false, err
	}
	b, err := p.br.ReadByte()
	if err != nil {
		return false, err
	}
	switch b {
	case ',':
		return false, nil
	case '}':
		return true, nil
	default:
		return false, fmt.Errorf("expected ',' or '}'")
	}
}

func (p *geminiSSEParser) consumeNull() (bool, error) {
	if err := p.skipSpace(); err != nil {
		return false, err
	}
	b, err := p.br.Peek(1)
	if err != nil {
		return false, err
	}
	if b[0] != 'n' {
		return false, nil
	}
	var lit [4]byte
	if _, err := io.ReadFull(p.br, lit[:]); err != nil {
		return false, err
	}
	if string(lit[:]) != "null" {
		return false, fmt.Errorf("invalid literal %q", lit[:])
	}
	return true, nil
}

func (p *geminiSSEParser) parseNullableString() (string, error) {
	null, err := p.consumeNull()
	if err != nil || null {
		return "", err
	}
	return p.parseString()
}

// parseStringIfPresent decodes a string field but leaves dst untouched when
// the value is null. This matches encoding/json, where null into a non-pointer
// string has no effect; a repeated key ending in null must not clear an earlier
// value.
func (p *geminiSSEParser) parseStringIfPresent(dst *string) error {
	null, err := p.consumeNull()
	if err != nil || null {
		return err
	}
	*dst, err = p.parseString()
	return err
}

// parseIntegerIfPresent mirrors parseStringIfPresent for integer fields.
func (p *geminiSSEParser) parseIntegerIfPresent(dst *int) error {
	null, err := p.consumeNull()
	if err != nil || null {
		return err
	}
	*dst, err = p.parseInteger()
	return err
}

func (p *geminiSSEParser) parseString() (string, error) {
	var b strings.Builder
	if err := p.parseStringTo(metadataWriter{p: p, w: &b}); err != nil {
		return "", err
	}
	return b.String(), nil
}

func (p *geminiSSEParser) parseStringTo(w io.Writer) error {
	if err := p.skipSpace(); err != nil {
		return err
	}
	open, err := p.br.ReadByte()
	if err != nil {
		return err
	}
	if open != '"' {
		return fmt.Errorf("expected string")
	}

	buf := make([]byte, 0, 32*1024)
	flush := func() error {
		if len(buf) == 0 {
			return nil
		}
		if _, err := w.Write(buf); err != nil {
			return err
		}
		buf = buf[:0]
		return nil
	}
	appendBytes := func(data []byte) error {
		for len(data) > 0 {
			if len(buf) == cap(buf) {
				if err := flush(); err != nil {
					return err
				}
			}
			n := cap(buf) - len(buf)
			if n > len(data) {
				n = len(data)
			}
			buf = append(buf, data[:n]...)
			data = data[n:]
		}
		return nil
	}
	appendRune := func(r rune) error {
		var enc [utf8.UTFMax]byte
		n := utf8.EncodeRune(enc[:], r)
		return appendBytes(enc[:n])
	}

	rawPending := make([]byte, 0, utf8.UTFMax)

	for {
		c, err := p.br.ReadByte()
		if err != nil {
			return err
		}
		switch c {
		case '"':
			if len(rawPending) != 0 {
				return fmt.Errorf("invalid UTF-8 in string")
			}
			return flush()
		case '\\':
			if len(rawPending) != 0 {
				return fmt.Errorf("invalid UTF-8 in string")
			}
			esc, err := p.br.ReadByte()
			if err != nil {
				return err
			}
			switch esc {
			case '"', '\\', '/':
				if err := appendBytes([]byte{esc}); err != nil {
					return err
				}
			case 'b':
				if err := appendBytes([]byte{'\b'}); err != nil {
					return err
				}
			case 'f':
				if err := appendBytes([]byte{'\f'}); err != nil {
					return err
				}
			case 'n':
				if err := appendBytes([]byte{'\n'}); err != nil {
					return err
				}
			case 'r':
				if err := appendBytes([]byte{'\r'}); err != nil {
					return err
				}
			case 't':
				if err := appendBytes([]byte{'\t'}); err != nil {
					return err
				}
			case 'u':
				r, err := p.readHex4()
				if err != nil {
					return err
				}
				if utf16.IsSurrogate(r) {
					r = p.decodeSurrogate(r)
				}
				if err := appendRune(r); err != nil {
					return err
				}
			default:
				return fmt.Errorf("invalid escape sequence")
			}
		default:
			if c < 0x20 {
				return fmt.Errorf("invalid control character in string")
			}
			rawPending = append(rawPending, c)
			if !utf8.FullRune(rawPending) {
				if len(rawPending) >= utf8.UTFMax {
					return fmt.Errorf("invalid UTF-8 in string")
				}
				continue
			}
			r, size := utf8.DecodeRune(rawPending)
			if r == utf8.RuneError && size == 1 {
				return fmt.Errorf("invalid UTF-8 in string")
			}
			if err := appendBytes(rawPending[:size]); err != nil {
				return err
			}
			rawPending = rawPending[:0]
		}
	}
}

func (p *geminiSSEParser) readHex4() (rune, error) {
	var hex [4]byte
	if _, err := io.ReadFull(p.br, hex[:]); err != nil {
		return 0, err
	}
	value, err := strconv.ParseUint(string(hex[:]), 16, 16)
	if err != nil {
		return 0, fmt.Errorf("invalid unicode escape")
	}
	return rune(value), nil
}

func (p *geminiSSEParser) decodeSurrogate(high rune) rune {
	if high < 0xD800 || high > 0xDBFF {
		return utf8.RuneError
	}
	next, err := p.br.Peek(6)
	if err != nil || len(next) != 6 || next[0] != '\\' || next[1] != 'u' {
		return utf8.RuneError
	}
	value, err := strconv.ParseUint(string(next[2:6]), 16, 16)
	if err != nil {
		return utf8.RuneError
	}
	low := rune(value)
	if low < 0xDC00 || low > 0xDFFF {
		return utf8.RuneError
	}
	_, _ = p.br.Discard(6)
	return utf16.DecodeRune(high, low)
}

func (p *geminiSSEParser) parseBool() (*bool, error) {
	null, err := p.consumeNull()
	if err != nil || null {
		return nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, err
	}
	token, err := p.readLiteral(maxSSEBoolBytes)
	if err != nil {
		return nil, err
	}
	switch token {
	case "true":
		v := true
		return &v, nil
	case "false":
		v := false
		return &v, nil
	default:
		return nil, fmt.Errorf("expected boolean")
	}
}

func (p *geminiSSEParser) parseInteger() (int, error) {
	null, err := p.consumeNull()
	if err != nil || null {
		return 0, err
	}
	if err := p.skipSpace(); err != nil {
		return 0, err
	}
	token, err := p.readLiteral(maxSSEIntegerBytes)
	if err != nil {
		return 0, err
	}
	if !validJSONInteger(token) {
		return 0, fmt.Errorf("expected integer")
	}
	value, err := strconv.Atoi(token)
	if err != nil {
		return 0, fmt.Errorf("expected integer")
	}
	return value, nil
}

// validJSONInteger checks the JSON number grammar for integers before handing
// the token to strconv.Atoi. Atoi alone also accepts forms such as "+1" and
// "01" that JSON forbids, which would make SSE parsing disagree with the
// regular JSON decoder.
func validJSONInteger(token string) bool {
	if token == "" {
		return false
	}
	i := 0
	if token[i] == '-' {
		i++
		if i == len(token) {
			return false
		}
	}
	if token[i] == '0' {
		i++
	} else if token[i] >= '1' && token[i] <= '9' {
		i++
		for i < len(token) && token[i] >= '0' && token[i] <= '9' {
			i++
		}
	} else {
		return false
	}
	return i == len(token)
}

func (p *geminiSSEParser) readLiteral(max int) (string, error) {
	var b strings.Builder
	for {
		c, err := p.br.ReadByte()
		if err != nil {
			if err == io.EOF {
				return b.String(), nil
			}
			return "", err
		}
		switch c {
		case ' ', '\t', '\n', '\r', ',', '}', ']':
			_ = p.br.UnreadByte()
			return b.String(), nil
		default:
			if b.Len() >= max {
				return "", fmt.Errorf("SSE literal exceeds %d bytes", max)
			}
			b.WriteByte(c)
		}
	}
}

func (p *geminiSSEParser) parseCandidates() ([]GeminiCandidate, error) {
	if err := p.expect('['); err != nil {
		return nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == ']' {
		_, _ = p.br.ReadByte()
		return nil, nil
	}
	var out []GeminiCandidate
	for {
		if p.candidates >= maxSSECandidates {
			return nil, fmt.Errorf("SSE event has too many candidates (max %d)", maxSSECandidates)
		}
		candidate, err := p.parseCandidate()
		if err != nil {
			return nil, err
		}
		if err := p.chargeMetadata(candidateMetadataOverhead); err != nil {
			return nil, err
		}
		p.candidates++
		out = append(out, candidate)
		if err := p.skipSpace(); err != nil {
			return nil, err
		}
		b, err := p.br.ReadByte()
		if err != nil {
			return nil, err
		}
		if b == ']' {
			return out, nil
		}
		if b != ',' {
			return nil, fmt.Errorf("expected ',' or ']'")
		}
	}
}

func (p *geminiSSEParser) parseCandidate() (GeminiCandidate, error) {
	var candidate GeminiCandidate
	null, err := p.consumeNull()
	if err != nil {
		return candidate, err
	}
	if null {
		return candidate, nil
	}
	if err := p.expect('{'); err != nil {
		return candidate, err
	}
	if err := p.skipSpace(); err != nil {
		return candidate, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return candidate, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return candidate, err
		}
		if err := p.expect(':'); err != nil {
			return candidate, err
		}
		switch key {
		case "content":
			null, err := p.consumeNull()
			if err != nil {
				return candidate, err
			}
			if !null {
				candidate.Content, err = p.parseContent()
				if err != nil {
					return candidate, err
				}
			}
		case "finishReason":
			if err := p.parseStringIfPresent(&candidate.FinishReason); err != nil {
				return candidate, err
			}
		case "index":
			if err := p.parseIntegerIfPresent(&candidate.Index); err != nil {
				return candidate, err
			}
		default:
			if err := p.skipValue(); err != nil {
				return candidate, err
			}
		}
		done, err := p.afterField()
		if err != nil {
			return candidate, err
		}
		if done {
			return candidate, nil
		}
	}
}

func (p *geminiSSEParser) parseContent() (GeminiContent, error) {
	var content GeminiContent
	if err := p.expect('{'); err != nil {
		return content, err
	}
	if err := p.skipSpace(); err != nil {
		return content, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return content, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return content, err
		}
		if err := p.expect(':'); err != nil {
			return content, err
		}
		switch key {
		case "role":
			if err := p.parseStringIfPresent(&content.Role); err != nil {
				return content, err
			}
		case "parts":
			null, err := p.consumeNull()
			if err != nil {
				return content, err
			}
			if null {
				content.Parts = nil
				break
			}
			content.Parts, err = p.parseParts()
			if err != nil {
				return content, err
			}
		default:
			if err := p.skipValue(); err != nil {
				return content, err
			}
		}
		done, err := p.afterField()
		if err != nil {
			return content, err
		}
		if done {
			return content, nil
		}
	}
}

func (p *geminiSSEParser) parseParts() ([]GeminiPart, error) {
	if err := p.expect('['); err != nil {
		return nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == ']' {
		_, _ = p.br.ReadByte()
		return nil, nil
	}
	var parts []GeminiPart
	for {
		if p.parts >= maxSSEParts {
			return nil, fmt.Errorf("SSE event has too many parts (max %d)", maxSSEParts)
		}
		part, err := p.parsePart()
		if err != nil {
			return nil, err
		}
		if err := p.chargeMetadata(partMetadataOverhead); err != nil {
			return nil, err
		}
		p.parts++
		parts = append(parts, part)
		if err := p.skipSpace(); err != nil {
			return nil, err
		}
		b, err := p.br.ReadByte()
		if err != nil {
			return nil, err
		}
		if b == ']' {
			return parts, nil
		}
		if b != ',' {
			return nil, fmt.Errorf("expected ',' or ']'")
		}
	}
}

func (p *geminiSSEParser) parsePart() (GeminiPart, error) {
	var part GeminiPart
	null, err := p.consumeNull()
	if err != nil {
		return part, err
	}
	if null {
		return part, nil
	}
	if err := p.expect('{'); err != nil {
		return part, err
	}
	if err := p.skipSpace(); err != nil {
		return part, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return part, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return part, err
		}
		if err := p.expect(':'); err != nil {
			return part, err
		}
		switch key {
		case "text":
			null, err := p.consumeNull()
			if err != nil {
				return part, err
			}
			if null {
				part.Text = nil
				part.TextRange = nil
				break
			}
			offset := p.payload.Size()
			if err := p.parseStringTo(p.payload); err != nil {
				return part, err
			}
			part.TextRange = &StoredRange{Offset: offset, Length: p.payload.Size() - offset}
			empty := ""
			part.Text = &empty
		case "thought":
			part.Thought, err = p.parseBool()
			if err != nil {
				return part, err
			}
		case "thoughtSignature":
			if err := p.parseStringIfPresent(&part.ThoughtSignature); err != nil {
				return part, err
			}
		case "functionCall":
			null, err := p.consumeNull()
			if err != nil {
				return part, err
			}
			if null {
				part.FunctionCall = nil
				break
			}
			part.FunctionCall, err = p.parseFunctionCall()
			if err != nil {
				return part, err
			}
		case "functionResponse":
			null, err := p.consumeNull()
			if err != nil {
				return part, err
			}
			if null {
				part.FunctionResponse = nil
				break
			}
			part.FunctionResponse, err = p.parseFunctionResponse()
			if err != nil {
				return part, err
			}
		case "inlineData":
			null, err := p.consumeNull()
			if err != nil {
				return part, err
			}
			if null {
				part.InlineData = nil
				part.InlineDataRange = nil
				break
			}
			part.InlineData, part.InlineDataRange, err = p.parseInlineData()
			if err != nil {
				return part, err
			}
		case "fileData":
			null, err := p.consumeNull()
			if err != nil {
				return part, err
			}
			if null {
				part.FileData = nil
				break
			}
			part.FileData, err = p.parseFileData()
			if err != nil {
				return part, err
			}
		default:
			if err := p.skipValue(); err != nil {
				return part, err
			}
			if len(part.unknownFields) >= maxSSEPartUnknownFields {
				return part, fmt.Errorf("SSE part has too many unknown fields (max %d)", maxSSEPartUnknownFields)
			}
			if p.unknownFields >= maxSSEUnknownFields {
				return part, fmt.Errorf("SSE event has too many unknown part fields (max %d)", maxSSEUnknownFields)
			}
			p.unknownFields++
			part.unknownFields = append(part.unknownFields, key)
		}
		done, err := p.afterField()
		if err != nil {
			return part, err
		}
		if done {
			sort.Strings(part.unknownFields)
			return part, nil
		}
	}
}

func (p *geminiSSEParser) parseInlineData() (*GeminiInlineData, *StoredRange, error) {
	var data GeminiInlineData
	var stored *StoredRange
	if err := p.expect('{'); err != nil {
		return nil, nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, nil, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return &data, nil, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return nil, nil, err
		}
		if err := p.expect(':'); err != nil {
			return nil, nil, err
		}
		switch key {
		case "mimeType":
			if err := p.parseStringIfPresent(&data.MIMEType); err != nil {
				return nil, nil, err
			}
		case "data":
			null, err := p.consumeNull()
			if err != nil {
				return nil, nil, err
			}
			if null {
				break
			}
			offset := p.payload.Size()
			if err := p.parseStringTo(p.payload); err != nil {
				return nil, nil, err
			}
			stored = &StoredRange{Offset: offset, Length: p.payload.Size() - offset}
		default:
			if err := p.skipValue(); err != nil {
				return nil, nil, err
			}
		}
		done, err := p.afterField()
		if err != nil {
			return nil, nil, err
		}
		if done {
			return &data, stored, nil
		}
	}
}

func (p *geminiSSEParser) parseFunctionCall() (*GeminiFunctionCall, error) {
	var call GeminiFunctionCall
	if err := p.expect('{'); err != nil {
		return nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return &call, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return nil, err
		}
		if err := p.expect(':'); err != nil {
			return nil, err
		}
		switch key {
		case "name":
			if err := p.parseStringIfPresent(&call.Name); err != nil {
				return nil, err
			}
		case "id":
			if err := p.parseStringIfPresent(&call.ID); err != nil {
				return nil, err
			}
		case "args":
			call.Args, err = p.parseRawValue()
			if err != nil {
				return nil, err
			}
		default:
			if err := p.skipValue(); err != nil {
				return nil, err
			}
		}
		done, err := p.afterField()
		if err != nil {
			return nil, err
		}
		if done {
			return &call, nil
		}
	}
}

func (p *geminiSSEParser) parseFunctionResponse() (*GeminiFunctionResponse, error) {
	var resp GeminiFunctionResponse
	if err := p.expect('{'); err != nil {
		return nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return &resp, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return nil, err
		}
		if err := p.expect(':'); err != nil {
			return nil, err
		}
		switch key {
		case "name":
			if err := p.parseStringIfPresent(&resp.Name); err != nil {
				return nil, err
			}
		case "id":
			if err := p.parseStringIfPresent(&resp.ID); err != nil {
				return nil, err
			}
		case "response":
			resp.Response, err = p.parseRawValue()
			if err != nil {
				return nil, err
			}
		default:
			if err := p.skipValue(); err != nil {
				return nil, err
			}
		}
		done, err := p.afterField()
		if err != nil {
			return nil, err
		}
		if done {
			return &resp, nil
		}
	}
}

func (p *geminiSSEParser) parseFileData() (*GeminiFileData, error) {
	var data GeminiFileData
	if err := p.expect('{'); err != nil {
		return nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return &data, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return nil, err
		}
		if err := p.expect(':'); err != nil {
			return nil, err
		}
		switch key {
		case "mimeType":
			if err := p.parseStringIfPresent(&data.MIMEType); err != nil {
				return nil, err
			}
		case "fileUri":
			if err := p.parseStringIfPresent(&data.FileURI); err != nil {
				return nil, err
			}
		default:
			if err := p.skipValue(); err != nil {
				return nil, err
			}
		}
		done, err := p.afterField()
		if err != nil {
			return nil, err
		}
		if done {
			return &data, nil
		}
	}
}

func (p *geminiSSEParser) parseUsage() (*GeminiUsage, error) {
	var usage GeminiUsage
	if err := p.expect('{'); err != nil {
		return nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return &usage, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return nil, err
		}
		if err := p.expect(':'); err != nil {
			return nil, err
		}
		switch key {
		case "promptTokenCount":
			err = p.parseIntegerIfPresent(&usage.PromptTokenCount)
		case "candidatesTokenCount":
			err = p.parseIntegerIfPresent(&usage.CandidatesTokenCount)
		case "totalTokenCount":
			err = p.parseIntegerIfPresent(&usage.TotalTokenCount)
		case "cachedContentTokenCount":
			err = p.parseIntegerIfPresent(&usage.CachedContentTokenCount)
		case "thoughtsTokenCount":
			err = p.parseIntegerIfPresent(&usage.ThoughtsTokenCount)
		default:
			err = p.skipValue()
		}
		if err != nil {
			return nil, err
		}
		done, err := p.afterField()
		if err != nil {
			return nil, err
		}
		if done {
			return &usage, nil
		}
	}
}

func (p *geminiSSEParser) parsePromptFeedback() (*GeminiPromptFeedback, error) {
	var feedback GeminiPromptFeedback
	if err := p.expect('{'); err != nil {
		return nil, err
	}
	if err := p.skipSpace(); err != nil {
		return nil, err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return &feedback, nil
	}
	for {
		key, err := p.parseString()
		if err != nil {
			return nil, err
		}
		if err := p.expect(':'); err != nil {
			return nil, err
		}
		switch key {
		case "blockReason":
			err = p.parseStringIfPresent(&feedback.BlockReason)
		case "safetyRatings":
			feedback.SafetyRatings, err = p.parseRawValue()
		default:
			err = p.skipValue()
		}
		if err != nil {
			return nil, err
		}
		done, err := p.afterField()
		if err != nil {
			return nil, err
		}
		if done {
			return &feedback, nil
		}
	}
}

func (p *geminiSSEParser) parseRawValue() (json.RawMessage, error) {
	var buf bytes.Buffer
	if err := p.copyValue(metadataWriter{p: p, w: &buf}); err != nil {
		return nil, err
	}
	return json.RawMessage(buf.Bytes()), nil
}

func (p *geminiSSEParser) skipValue() error {
	return p.copyValue(io.Discard)
}

func (p *geminiSSEParser) copyValue(w io.Writer) error {
	if err := p.skipSpace(); err != nil {
		return err
	}
	b, err := p.br.Peek(1)
	if err != nil {
		return err
	}
	switch b[0] {
	case '"':
		return p.copyStringRaw(w)
	case '{':
		return p.copyObject(w)
	case '[':
		return p.copyArray(w)
	default:
		return p.copyLiteral(w)
	}
}

func (p *geminiSSEParser) copyStringRaw(w io.Writer) error {
	if err := p.skipSpace(); err != nil {
		return err
	}
	open, err := p.br.ReadByte()
	if err != nil {
		return err
	}
	if open != '"' {
		return fmt.Errorf("expected string")
	}
	if err := writeByte(w, '"'); err != nil {
		return err
	}
	for {
		c, err := p.br.ReadByte()
		if err != nil {
			return err
		}
		switch c {
		case '"':
			return writeByte(w, '"')
		case '\\':
			esc, err := p.br.ReadByte()
			if err != nil {
				return err
			}
			switch esc {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				if err := writeByte(w, '\\'); err != nil {
					return err
				}
				if err := writeByte(w, esc); err != nil {
					return err
				}
			case 'u':
				var hex [4]byte
				if _, err := io.ReadFull(p.br, hex[:]); err != nil {
					return err
				}
				if _, err := strconv.ParseUint(string(hex[:]), 16, 16); err != nil {
					return fmt.Errorf("invalid unicode escape")
				}
				if _, err := w.Write([]byte{'\\', 'u'}); err != nil {
					return err
				}
				if _, err := w.Write(hex[:]); err != nil {
					return err
				}
			default:
				return fmt.Errorf("invalid escape sequence")
			}
		default:
			if c < 0x20 {
				return fmt.Errorf("invalid control character in string")
			}
			if err := writeByte(w, c); err != nil {
				return err
			}
		}
	}
}

func (p *geminiSSEParser) copyObject(w io.Writer) error {
	if err := p.expect('{'); err != nil {
		return err
	}
	if err := p.enterNesting(); err != nil {
		return err
	}
	defer p.leaveNesting()
	if err := writeByte(w, '{'); err != nil {
		return err
	}
	if err := p.skipSpace(); err != nil {
		return err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == '}' {
		_, _ = p.br.ReadByte()
		return writeByte(w, '}')
	}
	for {
		if err := p.copyStringRaw(w); err != nil {
			return err
		}
		if err := p.expect(':'); err != nil {
			return err
		}
		if err := writeByte(w, ':'); err != nil {
			return err
		}
		if err := p.copyValue(w); err != nil {
			return err
		}
		if err := p.skipSpace(); err != nil {
			return err
		}
		b, err := p.br.ReadByte()
		if err != nil {
			return err
		}
		switch b {
		case ',':
			if err := writeByte(w, ','); err != nil {
				return err
			}
		case '}':
			return writeByte(w, '}')
		default:
			return fmt.Errorf("expected ',' or '}'")
		}
	}
}

func (p *geminiSSEParser) copyArray(w io.Writer) error {
	if err := p.expect('['); err != nil {
		return err
	}
	if err := p.enterNesting(); err != nil {
		return err
	}
	defer p.leaveNesting()
	if err := writeByte(w, '['); err != nil {
		return err
	}
	if err := p.skipSpace(); err != nil {
		return err
	}
	if b, err := p.br.Peek(1); err == nil && b[0] == ']' {
		_, _ = p.br.ReadByte()
		return writeByte(w, ']')
	}
	for {
		if err := p.copyValue(w); err != nil {
			return err
		}
		if err := p.skipSpace(); err != nil {
			return err
		}
		b, err := p.br.ReadByte()
		if err != nil {
			return err
		}
		switch b {
		case ',':
			if err := writeByte(w, ','); err != nil {
				return err
			}
		case ']':
			return writeByte(w, ']')
		default:
			return fmt.Errorf("expected ',' or ']'")
		}
	}
}

func (p *geminiSSEParser) copyLiteral(w io.Writer) error {
	if err := p.skipSpace(); err != nil {
		return err
	}
	skip := w == io.Discard
	limit := maxSSELiteralBytes
	if !skip {
		limit = maxSSEMetadataBytes + 1
	}
	var v literalValidator
	var buf [32 * 1024]byte
	bufN := 0
	flush := func() error {
		if skip || bufN == 0 {
			bufN = 0
			return nil
		}
		_, err := w.Write(buf[:bufN])
		bufN = 0
		return err
	}
	appendByte := func(c byte) error {
		if skip {
			return nil
		}
		buf[bufN] = c
		bufN++
		if bufN == len(buf) {
			return flush()
		}
		return nil
	}
	n := 0
	for {
		c, err := p.br.ReadByte()
		if err != nil {
			if err == io.EOF {
				if n == 0 {
					return fmt.Errorf("invalid JSON value")
				}
				if err := v.finish(); err != nil {
					return err
				}
				return flush()
			}
			return err
		}
		switch c {
		case ' ', '\t', '\n', '\r', ',', '}', ']':
			_ = p.br.UnreadByte()
			if n == 0 {
				return fmt.Errorf("invalid JSON value")
			}
			if err := v.finish(); err != nil {
				return err
			}
			return flush()
		default:
			if n >= limit {
				if skip {
					return fmt.Errorf("SSE literal exceeds %d bytes", maxSSELiteralBytes)
				}
				return fmt.Errorf("SSE event metadata exceeds %d bytes", maxSSEMetadataBytes)
			}
			if err := v.write(c); err != nil {
				return err
			}
			if err := appendByte(c); err != nil {
				return err
			}
			n++
		}
	}
}

type literalValidator struct {
	state      int
	keyword    string
	keywordPos int
}

func (v *literalValidator) write(c byte) error {
	switch v.state {
	case 0:
		switch c {
		case 't':
			v.state = 9
			v.keyword = "true"
			v.keywordPos = 1
		case 'f':
			v.state = 9
			v.keyword = "false"
			v.keywordPos = 1
		case 'n':
			v.state = 9
			v.keyword = "null"
			v.keywordPos = 1
		case '-':
			v.state = 1
		case '0':
			v.state = 2
		default:
			if c < '1' || c > '9' {
				return fmt.Errorf("invalid JSON value")
			}
			v.state = 3
		}
	case 1:
		switch {
		case c == '0':
			v.state = 2
		case c >= '1' && c <= '9':
			v.state = 3
		default:
			return fmt.Errorf("invalid JSON value")
		}
	case 2:
		switch c {
		case '.':
			v.state = 4
		case 'e', 'E':
			v.state = 6
		default:
			return fmt.Errorf("invalid JSON value")
		}
	case 3:
		switch {
		case c >= '0' && c <= '9':
		case c == '.':
			v.state = 4
		case c == 'e' || c == 'E':
			v.state = 6
		default:
			return fmt.Errorf("invalid JSON value")
		}
	case 4:
		if c < '0' || c > '9' {
			return fmt.Errorf("invalid JSON value")
		}
		v.state = 5
	case 5:
		switch {
		case c >= '0' && c <= '9':
		case c == 'e' || c == 'E':
			v.state = 6
		default:
			return fmt.Errorf("invalid JSON value")
		}
	case 6:
		switch {
		case c == '+' || c == '-':
			v.state = 7
		case c >= '0' && c <= '9':
			v.state = 8
		default:
			return fmt.Errorf("invalid JSON value")
		}
	case 7:
		if c < '0' || c > '9' {
			return fmt.Errorf("invalid JSON value")
		}
		v.state = 8
	case 8:
		if c < '0' || c > '9' {
			return fmt.Errorf("invalid JSON value")
		}
	case 9:
		if v.keywordPos >= len(v.keyword) || c != v.keyword[v.keywordPos] {
			return fmt.Errorf("invalid JSON value")
		}
		v.keywordPos++
	default:
		return fmt.Errorf("invalid JSON value")
	}
	return nil
}

func (v *literalValidator) finish() error {
	switch v.state {
	case 2, 3, 5, 8:
		return nil
	case 9:
		if v.keywordPos == len(v.keyword) {
			return nil
		}
	}
	return fmt.Errorf("invalid JSON value")
}

func writeByte(w io.Writer, b byte) error {
	var buf [1]byte
	buf[0] = b
	_, err := w.Write(buf[:])
	return err
}
