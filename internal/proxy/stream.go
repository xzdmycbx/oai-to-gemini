package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/xzdmycbx/oai-to-gemini/internal/aggregate"
	"github.com/xzdmycbx/oai-to-gemini/internal/config"
	"github.com/xzdmycbx/oai-to-gemini/internal/convert"
)

func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request, provider config.Provider, req *convert.ChatRequest, raw map[string]json.RawMessage) {
	geminiReq, err := convert.ToGemini(req, raw)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "invalid_request")
		return
	}
	payload, err := json.Marshal(geminiReq)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, "failed to encode upstream request", "server_error", "encode_error")
		return
	}

	path := "/models/" + url.PathEscape(req.Model) + ":streamGenerateContent"
	query := url.Values{"alt": []string{"sse"}}
	upstream, err := s.newUpstreamRequest(r.Context(), r, provider, http.MethodPost, path, payload, query)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, "failed to build upstream request", "server_error", "upstream_request")
		return
	}
	upstream.Header.Set("Accept", "text/event-stream")

	resp, err := s.doUpstream(r.Context(), upstream)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "upstream request failed: "+err.Error(), "upstream_error", "upstream_unreachable")
		return
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		writeUpstreamError(w, resp)
		return
	}
	defer resp.Body.Close()

	if !isEventStream(resp.Header.Get("Content-Type")) {
		s.writeSingleResponseAsStream(w, resp.Body, req.Model)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	builder := convert.NewChunkBuilder(req.Model)
	if err := writeSSE(w, builder.RoleChunk()); err != nil {
		return
	}

	var finishReason string
	var promptBlockReason string
	var usage *convert.GeminiUsage

	completed, valid, err := readGeminiSSEWithStore(resp.Body, s.store, func(chunk *convert.GeminiResponse, payload io.ReaderAt) error {
		if msg := embeddedGeminiError(chunk.Error); msg != "" {
			return fmt.Errorf("%s", msg)
		}
		if chunk.UsageMetadata != nil {
			usage = chunk.UsageMetadata
		}
		if chunk.PromptFeedback != nil && chunk.PromptFeedback.BlockReason != "" {
			promptBlockReason = chunk.PromptFeedback.BlockReason
		}
		if len(chunk.Candidates) == 0 {
			return nil
		}
		candidate := chunk.Candidates[0]
		if candidate.FinishReason != "" {
			if _, err := convert.FinishReasonFor(candidate.FinishReason, false); err != nil {
				return err
			}
			finishReason = candidate.FinishReason
		}
		return writeCandidateChunks(w, builder, candidate, payload)
	})
	if err != nil {
		_ = writeSSE(w, errorEnvelope{Error: errorBody{
			Message: "upstream stream failed: " + err.Error(),
			Type:    "upstream_error",
			Code:    streamErrorCode(err),
		}})
		writeSSEDone(w)
		return
	}
	if !completed {
		_ = writeSSE(w, errorEnvelope{Error: errorBody{
			Message: "upstream stream ended before a final finishReason or [DONE]",
			Type:    "upstream_error",
			Code:    "upstream_incomplete",
		}})
		writeSSEDone(w)
		return
	}
	if !valid {
		err := invalidStreamResponse()
		_ = writeSSE(w, errorEnvelope{Error: errorBody{
			Message: err.Error(),
			Type:    "upstream_error",
			Code:    conversionErrorCode(err),
		}})
		writeSSEDone(w)
		return
	}

	if promptBlockReason != "" {
		finishReason = convert.MapPromptBlockReason(promptBlockReason)
	}
	finishChunk, err := builder.FinishChunk(finishReason, usage)
	if err != nil {
		_ = writeSSE(w, errorEnvelope{Error: errorBody{
			Message: "upstream stream failed: " + err.Error(),
			Type:    "upstream_error",
			Code:    "abnormal_finish",
		}})
		writeSSEDone(w)
		return
	}
	if err := writeSSE(w, finishChunk); err != nil {
		return
	}
	writeSSEDone(w)
}

func writeCandidateChunks(w http.ResponseWriter, builder *convert.ChunkBuilder, candidate convert.GeminiCandidate, payload io.ReaderAt) error {
	var pendingText strings.Builder
	flushText := func() error {
		if pendingText.Len() == 0 {
			return nil
		}
		err := writeSSE(w, builder.ContentChunk(pendingText.String()))
		pendingText.Reset()
		return err
	}

	for _, part := range candidate.Content.Parts {
		if err := part.UnsupportedContentError(); err != nil {
			return err
		}
		if isThought(part) {
			continue
		}
		switch {
		case part.TextRange != nil:
			if part.TextRange.Length == 0 {
				continue
			}
			if err := flushText(); err != nil {
				return err
			}
			if payload == nil {
				return fmt.Errorf("missing stored text payload")
			}
			reader := io.NewSectionReader(payload, part.TextRange.Offset, part.TextRange.Length)
			if err := writeSSEChunkJSON(w, func(out io.Writer, chunk *convert.OpenAIChunk) error {
				return convert.WriteOpenAIChunkWithContent(out, chunk, reader)
			}, builder.ContentChunk("")); err != nil {
				return err
			}
		case part.Text != nil:
			pendingText.WriteString(*part.Text)
		case part.FunctionCall != nil:
			if err := flushText(); err != nil {
				return err
			}
			if err := writeSSE(w, builder.ToolCallChunk(part.FunctionCall, part.ThoughtSignature)); err != nil {
				return err
			}
		case part.InlineData != nil:
			if err := flushText(); err != nil {
				return err
			}
			if part.InlineDataRange != nil {
				if payload == nil {
					return fmt.Errorf("missing stored image payload")
				}
				mimeType, err := convert.ValidateInlineImageMetadata(part.InlineData)
				if err != nil {
					return err
				}
				if part.InlineDataRange.Length == 0 {
					return &convert.UnsupportedContentError{Message: "Gemini image inlineData is empty"}
				}
				reader := io.NewSectionReader(payload, part.InlineDataRange.Offset, part.InlineDataRange.Length)
				if err := writeSSEChunkJSON(w, func(out io.Writer, chunk *convert.OpenAIChunk) error {
					return convert.WriteOpenAIChunkWithImage(out, chunk, mimeType, reader)
				}, builder.ContentPartsChunk(nil)); err != nil {
					return err
				}
				break
			}
			contentPart, err := convert.InlineDataToContentPart(part.InlineData)
			if err != nil {
				return err
			}
			if err := writeSSE(w, builder.ContentPartsChunk([]convert.OpenAIContentPart{*contentPart})); err != nil {
				return err
			}
		case part.FileData != nil:
			return &convert.UnsupportedContentError{Message: "unsupported Gemini response content type fileData"}
		case part.FunctionResponse != nil:
			return &convert.UnsupportedContentError{Message: "unsupported Gemini response content type functionResponse"}
		default:
			return &convert.UnsupportedContentError{Message: "unsupported Gemini response content part with no recognized content"}
		}
	}
	return flushText()
}

func writeSSEChunkJSON(w http.ResponseWriter, write func(io.Writer, *convert.OpenAIChunk) error, chunk *convert.OpenAIChunk) error {
	if _, err := io.WriteString(w, "data: "); err != nil {
		return err
	}
	if err := write(w, chunk); err != nil {
		return err
	}
	if _, err := io.WriteString(w, "\n\n"); err != nil {
		return err
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

func (s *Server) writeSingleResponseAsStream(w http.ResponseWriter, body io.Reader, model string) {
	data, tooLarge, err := readAllLimited(body, maxUpstreamJSONBytes)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "failed to read upstream response", "upstream_error", "upstream_read")
		return
	}
	if tooLarge {
		writeOpenAIError(w, http.StatusBadGateway, "upstream response too large", "upstream_error", "upstream_response_too_large")
		return
	}
	geminiResp, err := convert.ParseGeminiResponseJSON(data)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "invalid upstream response", "upstream_error", "invalid_upstream_json")
		return
	}
	if msg := embeddedGeminiError(geminiResp.Error); msg != "" {
		writeOpenAIError(w, http.StatusBadGateway, msg, "upstream_error", "upstream_error")
		return
	}

	if _, err := convert.GeminiResponseToOpenAI(geminiResp, model); err != nil {
		writeConversionError(w, err)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	builder := convert.NewChunkBuilder(model)
	_ = writeSSE(w, builder.RoleChunk())

	var finishReason string
	if len(geminiResp.Candidates) > 0 {
		candidate := geminiResp.Candidates[0]
		finishReason = candidate.FinishReason
		if err := writeCandidateChunks(w, builder, candidate, nil); err != nil {
			_ = writeSSE(w, errorEnvelope{Error: errorBody{
				Message: "upstream response failed: " + err.Error(),
				Type:    "upstream_error",
				Code:    conversionErrorCode(err),
			}})
			writeSSEDone(w)
			return
		}
	}
	if geminiResp.PromptFeedback != nil && geminiResp.PromptFeedback.BlockReason != "" {
		finishReason = convert.MapPromptBlockReason(geminiResp.PromptFeedback.BlockReason)
	}
	finishChunk, err := builder.FinishChunk(finishReason, geminiResp.UsageMetadata)
	if err != nil {
		_ = writeSSE(w, errorEnvelope{Error: errorBody{
			Message: "upstream response failed: " + err.Error(),
			Type:    "upstream_error",
			Code:    "abnormal_finish",
		}})
		writeSSEDone(w)
		return
	}
	_ = writeSSE(w, finishChunk)
	writeSSEDone(w)
}

// readGeminiSSE is the compatibility wrapper for callers that need a fully
// materialized response. Proxy handlers use readGeminiSSEWithStore so large
// text and inlineData payloads can stay in spill storage.
func readGeminiSSE(r io.Reader, fn func(*convert.GeminiResponse) error) (completed bool, valid bool, err error) {
	store, err := aggregate.NewStore("", config.DefaultSSESpillThresholdBytes, config.DefaultSSECacheMaxBytes, nil)
	if err != nil {
		return false, false, err
	}
	defer func() { _ = store.Close() }()
	return readGeminiSSEWithStore(r, store, func(chunk *convert.GeminiResponse, payload io.ReaderAt) error {
		if payload != nil {
			if err := materializeStoredParts(chunk, payload); err != nil {
				return err
			}
		}
		return fn(chunk)
	})
}

// readGeminiSSEWithStore reads Gemini's SSE stream. Data lines are combined per
// SSE event (blank-line boundary) in a spill-backed buffer, so a single large
// event is not kept in memory as a Scanner token plus a strings.Builder plus a
// decoded []byte. Large JSON text and inlineData strings are decoded
// incrementally into a second spill-backed payload buffer.
//
// It reports terminal and structural validity separately: completed means a
// [DONE] event or a finishReason/promptFeedback block reason was observed;
// valid means at least one structurally usable candidate or an explicit prompt
// block reason was observed. Gemini does not always send [DONE], so a final
// finishReason is also accepted as completion. A bare [DONE], an empty object,
// or an empty candidate list is not a valid response.
func readGeminiSSEWithStore(r io.Reader, store *aggregate.Store, fn func(*convert.GeminiResponse, io.ReaderAt) error) (completed bool, valid bool, err error) {
	br := bufio.NewReaderSize(r, 64*1024)
	eventBuf := store.NewBuffer()
	defer func() {
		if eventBuf != nil {
			_ = eventBuf.Close()
		}
	}()

	var (
		lineBytes        int
		linePrefix       []byte
		lineDataKnown    bool
		lineIsData       bool
		lineNewlineAdded bool
		lineValueStarted bool
		skipLF           bool
	)

	resetLine := func() {
		lineBytes = 0
		linePrefix = linePrefix[:0]
		lineDataKnown = false
		lineIsData = false
		lineNewlineAdded = false
		lineValueStarted = false
	}

	startDataLine := func() error {
		if lineNewlineAdded {
			return nil
		}
		if eventBuf.Size() > 0 {
			if _, err := eventBuf.WriteString("\n"); err != nil {
				return err
			}
		}
		lineNewlineAdded = true
		return nil
	}

	processFragment := func(fragment []byte, lineEnd bool) error {
		lineBytes += len(fragment)
		if !lineDataKnown {
			need := 5 - len(linePrefix)
			if need > len(fragment) {
				need = len(fragment)
			}
			linePrefix = append(linePrefix, fragment[:need]...)
			fragment = fragment[need:]
			if len(linePrefix) == 5 {
				lineIsData = string(linePrefix) == "data:"
				lineDataKnown = true
			} else if !lineEnd {
				return nil
			} else {
				lineIsData = false
				lineDataKnown = true
			}
		}
		if !lineIsData {
			return nil
		}
		if !lineValueStarted {
			if len(fragment) == 0 {
				if lineEnd {
					return startDataLine()
				}
				return nil
			}
			if fragment[0] == ' ' {
				fragment = fragment[1:]
			}
			lineValueStarted = true
		}
		if err := startDataLine(); err != nil {
			return err
		}
		if len(fragment) == 0 {
			return nil
		}
		if _, err := eventBuf.Write(fragment); err != nil {
			return err
		}
		return nil
	}

	handleEvent := func(chunk *convert.GeminiResponse, payloadBuf *aggregate.Buffer) (bool, error) {
		if chunk.PromptFeedback != nil && chunk.PromptFeedback.BlockReason != "" {
			completed = true
			valid = true
		}
		for _, candidate := range chunk.Candidates {
			if candidate.FinishReason != "" {
				completed = true
			}
			// A terminal candidate is allowed to have empty content, and a
			// content-only candidate is allowed to omit finishReason. Either
			// shape makes the response structurally usable.
			if len(candidate.Content.Parts) > 0 || candidate.FinishReason != "" {
				valid = true
			}
		}

		var payload io.ReaderAt
		var payloadCloser io.Closer
		if payloadBuf.Size() > 0 {
			readerAt, closer, err := payloadBuf.ReaderAt()
			if err != nil {
				return false, err
			}
			payload = readerAt
			payloadCloser = closer
			defer func() { _ = payloadCloser.Close() }()
		}
		if err := fn(chunk, payload); err != nil {
			return false, err
		}
		return false, nil
	}

	processEvent := func() (bool, error) {
		size := eventBuf.Size()
		if size == 0 {
			return false, nil
		}

		var (
			chunk *convert.GeminiResponse
			err   error
		)
		if size <= 64 {
			reader, readErr := eventBuf.Reader()
			if readErr != nil {
				return false, readErr
			}
			data, readErr := io.ReadAll(reader)
			_ = reader.Close()
			if readErr != nil {
				return false, readErr
			}
			data = bytes.TrimSpace(data)
			if len(data) == 0 {
				return false, nil
			}
			if string(data) == "[DONE]" {
				return true, nil
			}
			payloadBuf := store.NewBuffer()
			defer func() { _ = payloadBuf.Close() }()
			chunk, err = convert.ParseGeminiSSEPayload(bytes.NewReader(data), payloadBuf)
			if err != nil {
				return false, fmt.Errorf("invalid SSE JSON: %w", err)
			}
			if chunk == nil {
				return false, nil
			}
			return handleEvent(chunk, payloadBuf)
		}

		readerAt, closer, readErr := eventBuf.ReaderAt()
		if readErr != nil {
			return false, readErr
		}
		payloadBuf := store.NewBuffer()
		defer func() { _ = payloadBuf.Close() }()
		chunk, err = convert.ParseGeminiSSEPayload(io.NewSectionReader(readerAt, 0, size), payloadBuf)
		_ = closer.Close()
		if err != nil {
			return false, fmt.Errorf("invalid SSE JSON: %w", err)
		}
		if chunk == nil {
			return false, nil
		}
		return handleEvent(chunk, payloadBuf)
	}

	finishLine := func() (bool, error) {
		if lineBytes != 0 {
			resetLine()
			return false, nil
		}
		resetLine()
		if eventBuf.Size() == 0 {
			return false, nil
		}
		done, err := processEvent()
		if err != nil {
			return false, err
		}
		_ = eventBuf.Close()
		eventBuf = store.NewBuffer()
		return done, nil
	}

	buf := make([]byte, 64*1024)
	for {
		n, readErr := br.Read(buf)
		i := 0
		for i < n {
			if skipLF {
				skipLF = false
				if buf[i] == '\n' {
					i++
					continue
				}
			}
			j := bytes.IndexAny(buf[i:n], "\r\n")
			if j < 0 {
				if err := processFragment(buf[i:n], false); err != nil {
					return completed, valid, err
				}
				i = n
				break
			}
			end := i + j
			if err := processFragment(buf[i:end], true); err != nil {
				return completed, valid, err
			}
			if buf[end] == '\r' {
				if end+1 < n {
					if buf[end+1] == '\n' {
						i = end + 2
					} else {
						i = end + 1
					}
				} else {
					skipLF = true
					i = end + 1
				}
			} else {
				i = end + 1
			}
			done, err := finishLine()
			if err != nil {
				return completed, valid, err
			}
			if done {
				return true, valid, nil
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			return completed, valid, readErr
		}
	}

	if lineBytes != 0 {
		resetLine()
	}
	if eventBuf.Size() > 0 {
		done, err := processEvent()
		if err != nil {
			return completed, valid, err
		}
		if done {
			return true, valid, nil
		}
	}
	return completed, valid, nil
}

func materializeStoredParts(chunk *convert.GeminiResponse, payload io.ReaderAt) error {
	for ci := range chunk.Candidates {
		parts := chunk.Candidates[ci].Content.Parts
		for pi := range parts {
			part := &parts[pi]
			if part.TextRange != nil {
				text, err := readStoredRange(payload, *part.TextRange)
				if err != nil {
					return err
				}
				part.Text = &text
				part.TextRange = nil
			}
			if part.InlineDataRange != nil && part.InlineData != nil {
				data, err := readStoredRange(payload, *part.InlineDataRange)
				if err != nil {
					return err
				}
				part.InlineData.Data = data
				part.InlineDataRange = nil
			}
		}
	}
	return nil
}

func readStoredRange(payload io.ReaderAt, r convert.StoredRange) (string, error) {
	if r.Length < 0 || r.Offset < 0 {
		return "", fmt.Errorf("invalid stored payload range")
	}
	data, err := io.ReadAll(io.NewSectionReader(payload, r.Offset, r.Length))
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func copyStoredRange(dst io.Writer, payload io.ReaderAt, r convert.StoredRange) error {
	if r.Length < 0 || r.Offset < 0 {
		return fmt.Errorf("invalid stored payload range")
	}
	buf := make([]byte, 32*1024)
	_, err := io.CopyBuffer(dst, io.NewSectionReader(payload, r.Offset, r.Length), buf)
	return err
}

// splitSSELines splits SSE lines on CR, LF, or CRLF. A trailing CR is held
// back until the next byte arrives so CRLF is not mistaken for two line ends.
func splitSSELines(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if len(data) == 0 {
		return 0, nil, nil
	}
	if i := bytes.IndexAny(data, "\r\n"); i >= 0 {
		if data[i] == '\r' && i+1 == len(data) && !atEOF {
			return 0, nil, nil
		}
		advance = i + 1
		if data[i] == '\r' && i+1 < len(data) && data[i+1] == '\n' {
			advance = i + 2
		}
		return advance, data[:i], nil
	}
	if atEOF {
		return len(data), data, nil
	}
	return 0, nil, nil
}

func invalidStreamResponse() error {
	return &convert.InvalidUpstreamResponseError{
		Reason: "stream contained no valid candidate and no prompt block reason",
	}
}

func writeSSE(w http.ResponseWriter, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

func writeSSEDone(w http.ResponseWriter) {
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}
