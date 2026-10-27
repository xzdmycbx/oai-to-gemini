package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
	"unicode/utf8"
)

// WriteOpenAIResponseJSON writes a complete OpenAI chat completion response
// while streaming message content from content. It avoids materializing the
// content in memory, which keeps the final stage of SSE aggregation bounded.
//
// A nil content reader means the response has no text. In that case an empty
// string is written when there are no tool calls (matching
// GeminiResponseToOpenAI), and null is written when tool calls are present.
func WriteOpenAIResponseJSON(w io.Writer, resp *GeminiResponse, model string, content io.Reader) error {
	return WriteOpenAIResponseJSONWithParts(w, resp, model, content, nil, nil, 0)
}

// WriteOpenAIResponseJSONWithParts is WriteOpenAIResponseJSON plus support for
// non-text content parts. parts are emitted in order, and offsets[i] is the
// number of text bytes that precede parts[i]. textSize is the total number of
// bytes available from content.
func WriteOpenAIResponseJSONWithParts(w io.Writer, resp *GeminiResponse, model string, content io.Reader, parts []OpenAIContentPart, offsets []int64, textSize int64) error {
	return writeOpenAIResponseJSON(w, resp, model, func(out io.Writer, toolCalls []OpenAIToolCallOut) error {
		switch {
		case len(parts) > 0:
			return writeContentPartsArray(out, content, parts, offsets, textSize)
		case content != nil:
			return writeJSONStringFromReader(out, content)
		case len(toolCalls) == 0:
			_, err := io.WriteString(out, `""`)
			return err
		default:
			_, err := io.WriteString(out, `null`)
			return err
		}
	})
}

// StoredImagePart identifies an image payload in a spill-backed store. Length
// is the number of raw inlineData bytes, not the size of the resulting data URL.
type StoredImagePart struct {
	MIMEType string
	Offset   int64
	Length   int64
}

// WriteOpenAIResponseJSONWithStoredParts is WriteOpenAIResponseJSON plus
// support for image payloads kept in a spill-backed store. offsets[i] is the
// number of text bytes that precede images[i]. Image bytes are copied from
// imageData in bounded chunks while the response is written, so the data URL
// is never materialized as one string.
func WriteOpenAIResponseJSONWithStoredParts(w io.Writer, resp *GeminiResponse, model string, content io.Reader, offsets []int64, textSize int64, images []StoredImagePart, imageData io.ReaderAt) error {
	return writeOpenAIResponseJSON(w, resp, model, func(out io.Writer, toolCalls []OpenAIToolCallOut) error {
		switch {
		case len(images) > 0:
			return writeStoredImagesArray(out, content, offsets, textSize, images, imageData)
		case content != nil:
			return writeJSONStringFromReader(out, content)
		case len(toolCalls) == 0:
			_, err := io.WriteString(out, `""`)
			return err
		default:
			_, err := io.WriteString(out, `null`)
			return err
		}
	})
}

func writeOpenAIResponseJSON(w io.Writer, resp *GeminiResponse, model string, writeContent func(io.Writer, []OpenAIToolCallOut) error) error {
	var candidate GeminiCandidate
	if len(resp.Candidates) > 0 {
		candidate = resp.Candidates[0]
	}

	toolCalls := ToolCallsFromParts(candidate.Content.Parts)
	finishReason := candidate.FinishReason
	if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
		finishReason = MapPromptBlockReason(resp.PromptFeedback.BlockReason)
	}
	mappedFinishReason, err := FinishReasonFor(finishReason, len(toolCalls) > 0)
	if err != nil {
		return err
	}
	finishReason = mappedFinishReason

	id := newID("chatcmpl-")
	created := time.Now().Unix()

	if _, err := io.WriteString(w, `{"id":`); err != nil {
		return err
	}
	if err := writeJSONValue(w, id); err != nil {
		return err
	}
	if _, err := io.WriteString(w, `,"object":"chat.completion","created":`); err != nil {
		return err
	}
	if err := writeJSONValue(w, created); err != nil {
		return err
	}
	if _, err := io.WriteString(w, `,"model":`); err != nil {
		return err
	}
	if err := writeJSONValue(w, model); err != nil {
		return err
	}
	if _, err := io.WriteString(w, `,"choices":[{"index":0,"message":{"role":"assistant","content":`); err != nil {
		return err
	}
	if err := writeContent(w, toolCalls); err != nil {
		return err
	}
	if len(toolCalls) > 0 {
		if _, err := io.WriteString(w, `,"tool_calls":`); err != nil {
			return err
		}
		if err := writeJSONValue(w, toolCalls); err != nil {
			return err
		}
	}
	if _, err := io.WriteString(w, `},"finish_reason":`); err != nil {
		return err
	}
	if err := writeJSONValue(w, finishReason); err != nil {
		return err
	}
	if _, err := io.WriteString(w, `}]`); err != nil {
		return err
	}
	if usage := UsageFromGemini(resp.UsageMetadata); usage != nil {
		if _, err := io.WriteString(w, `,"usage":`); err != nil {
			return err
		}
		if err := writeJSONValue(w, usage); err != nil {
			return err
		}
	}
	_, err = io.WriteString(w, "}\n")
	return err
}

func writeContentPartsArray(w io.Writer, content io.Reader, parts []OpenAIContentPart, offsets []int64, textSize int64) error {
	if len(parts) != len(offsets) {
		return fmt.Errorf("content parts and offsets length mismatch")
	}
	if textSize < 0 {
		textSize = 0
	}
	if _, err := io.WriteString(w, `[`); err != nil {
		return err
	}

	wroteAny := false
	written := int64(0)
	writeComma := func() error {
		if wroteAny {
			_, err := io.WriteString(w, `,`)
			return err
		}
		wroteAny = true
		return nil
	}
	writeText := func(n int64) error {
		if n <= 0 {
			return nil
		}
		if err := writeComma(); err != nil {
			return err
		}
		if _, err := io.WriteString(w, `{"type":"text","text":`); err != nil {
			return err
		}
		if content != nil {
			if err := writeJSONStringFromReader(w, io.LimitReader(content, n)); err != nil {
				return err
			}
		} else if _, err := io.WriteString(w, `""`); err != nil {
			return err
		}
		if _, err := io.WriteString(w, `}`); err != nil {
			return err
		}
		written += n
		return nil
	}

	for i, part := range parts {
		offset := offsets[i]
		if offset < written {
			offset = written
		}
		if offset > textSize {
			offset = textSize
		}
		if err := writeText(offset - written); err != nil {
			return err
		}
		if err := writeComma(); err != nil {
			return err
		}
		if err := writeJSONValue(w, part); err != nil {
			return err
		}
	}
	if err := writeText(textSize - written); err != nil {
		return err
	}
	_, err := io.WriteString(w, `]`)
	return err
}

func writeStoredImagesArray(w io.Writer, content io.Reader, offsets []int64, textSize int64, images []StoredImagePart, imageData io.ReaderAt) error {
	if len(images) != len(offsets) {
		return fmt.Errorf("stored images and offsets length mismatch")
	}
	if imageData == nil {
		return fmt.Errorf("stored image data reader is nil")
	}
	if textSize < 0 {
		textSize = 0
	}
	if _, err := io.WriteString(w, `[`); err != nil {
		return err
	}

	wroteAny := false
	written := int64(0)
	writeComma := func() error {
		if wroteAny {
			_, err := io.WriteString(w, `,`)
			return err
		}
		wroteAny = true
		return nil
	}
	writeText := func(n int64) error {
		if n <= 0 {
			return nil
		}
		if err := writeComma(); err != nil {
			return err
		}
		if _, err := io.WriteString(w, `{"type":"text","text":`); err != nil {
			return err
		}
		if content != nil {
			if err := writeJSONStringFromReader(w, io.LimitReader(content, n)); err != nil {
				return err
			}
		} else if _, err := io.WriteString(w, `""`); err != nil {
			return err
		}
		if _, err := io.WriteString(w, `}`); err != nil {
			return err
		}
		written += n
		return nil
	}

	for i, image := range images {
		offset := offsets[i]
		if offset < written {
			offset = written
		}
		if offset > textSize {
			offset = textSize
		}
		if err := writeText(offset - written); err != nil {
			return err
		}
		if err := writeComma(); err != nil {
			return err
		}
		if err := writeStoredImage(w, image, imageData); err != nil {
			return err
		}
	}
	if err := writeText(textSize - written); err != nil {
		return err
	}
	_, err := io.WriteString(w, `]`)
	return err
}

func writeStoredImage(w io.Writer, image StoredImagePart, imageData io.ReaderAt) error {
	if image.Length < 0 || image.Offset < 0 {
		return fmt.Errorf("invalid stored image range")
	}
	prefix := "data:" + image.MIMEType + ";base64,"
	if _, err := io.WriteString(w, `{"type":"image_url","image_url":{"url":"`); err != nil {
		return err
	}
	if err := writeEscapedJSONStringBody(w, []byte(prefix)); err != nil {
		return err
	}
	if err := writeEscapedJSONStringBodyFromReader(w, io.NewSectionReader(imageData, image.Offset, image.Length)); err != nil {
		return err
	}
	_, err := io.WriteString(w, `"}}`)
	return err
}

// WriteOpenAIChunkWithContent writes a streaming chunk whose delta.content is
// escaped incrementally from content. It avoids materializing a large stored
// text payload as a Go string before sending it downstream.
func WriteOpenAIChunkWithContent(w io.Writer, chunk *OpenAIChunk, content io.Reader) error {
	return writeOpenAIChunkJSON(w, chunk, func(out io.Writer) error {
		return writeJSONStringFromReader(out, content)
	})
}

// WriteOpenAIChunkWithImage writes a streaming chunk with one image_url content
// part whose data URL payload is escaped incrementally from data.
func WriteOpenAIChunkWithImage(w io.Writer, chunk *OpenAIChunk, mimeType string, data io.Reader) error {
	prefix := "data:" + mimeType + ";base64,"
	return writeOpenAIChunkJSON(w, chunk, func(out io.Writer) error {
		if _, err := io.WriteString(out, `[{"type":"image_url","image_url":{"url":`); err != nil {
			return err
		}
		if err := writeJSONStringFromReader(out, io.MultiReader(strings.NewReader(prefix), data)); err != nil {
			return err
		}
		_, err := io.WriteString(out, `}}]`)
		return err
	})
}

func writeOpenAIChunkJSON(w io.Writer, chunk *OpenAIChunk, writeContent func(io.Writer) error) error {
	if _, err := io.WriteString(w, `{"id":`); err != nil {
		return err
	}
	if err := writeJSONValue(w, chunk.ID); err != nil {
		return err
	}
	if _, err := io.WriteString(w, `,"object":`); err != nil {
		return err
	}
	if err := writeJSONValue(w, chunk.Object); err != nil {
		return err
	}
	if _, err := io.WriteString(w, `,"created":`); err != nil {
		return err
	}
	if err := writeJSONValue(w, chunk.Created); err != nil {
		return err
	}
	if _, err := io.WriteString(w, `,"model":`); err != nil {
		return err
	}
	if err := writeJSONValue(w, chunk.Model); err != nil {
		return err
	}
	if _, err := io.WriteString(w, `,"choices":[`); err != nil {
		return err
	}
	for i, choice := range chunk.Choices {
		if i > 0 {
			if _, err := io.WriteString(w, `,`); err != nil {
				return err
			}
		}
		if _, err := io.WriteString(w, `{"index":`); err != nil {
			return err
		}
		if err := writeJSONValue(w, choice.Index); err != nil {
			return err
		}
		if _, err := io.WriteString(w, `,"delta":{`); err != nil {
			return err
		}
		wroteField := false
		if choice.Delta.Role != "" {
			if _, err := io.WriteString(w, `"role":`); err != nil {
				return err
			}
			if err := writeJSONValue(w, choice.Delta.Role); err != nil {
				return err
			}
			wroteField = true
		}
		if writeContent != nil {
			if wroteField {
				if _, err := io.WriteString(w, `,`); err != nil {
					return err
				}
			}
			if _, err := io.WriteString(w, `"content":`); err != nil {
				return err
			}
			if err := writeContent(w); err != nil {
				return err
			}
			wroteField = true
		}
		if len(choice.Delta.ToolCalls) > 0 {
			if wroteField {
				if _, err := io.WriteString(w, `,`); err != nil {
					return err
				}
			}
			if _, err := io.WriteString(w, `"tool_calls":`); err != nil {
				return err
			}
			if err := writeJSONValue(w, choice.Delta.ToolCalls); err != nil {
				return err
			}
		}
		if _, err := io.WriteString(w, `},"finish_reason":`); err != nil {
			return err
		}
		if choice.FinishReason != nil {
			if err := writeJSONValue(w, *choice.FinishReason); err != nil {
				return err
			}
		} else if _, err := io.WriteString(w, `null`); err != nil {
			return err
		}
		if _, err := io.WriteString(w, `}`); err != nil {
			return err
		}
	}
	if _, err := io.WriteString(w, `]`); err != nil {
		return err
	}
	if chunk.Usage != nil {
		if _, err := io.WriteString(w, `,"usage":`); err != nil {
			return err
		}
		if err := writeJSONValue(w, chunk.Usage); err != nil {
			return err
		}
	}
	_, err := io.WriteString(w, `}`)
	return err
}

func writeJSONValue(w io.Writer, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// writeJSONStringFromReader writes a JSON string by escaping the reader in
// chunks. Chunks are split on UTF-8 rune boundaries so multi-byte characters
// are never replaced with the Unicode replacement character.
func writeJSONStringFromReader(w io.Writer, r io.Reader) error {
	if _, err := io.WriteString(w, `"`); err != nil {
		return err
	}
	if err := writeEscapedJSONStringBodyFromReader(w, r); err != nil {
		return err
	}
	_, err := io.WriteString(w, `"`)
	return err
}

// writeEscapedJSONStringBodyFromReader writes the body of a JSON string
// (without the surrounding quotes) while escaping incrementally.
func writeEscapedJSONStringBodyFromReader(w io.Writer, r io.Reader) error {
	buf := make([]byte, 32*1024)
	pending := make([]byte, 0, utf8.UTFMax-1)
	for {
		n, readErr := r.Read(buf)
		if n > 0 {
			var data []byte
			if len(pending) == 0 {
				data = buf[:n]
			} else {
				data = make([]byte, 0, len(pending)+n)
				data = append(data, pending...)
				data = append(data, buf[:n]...)
			}

			// Only an unfinished character at the end of the read can be
			// split across buffers. Keep at most utf8.UTFMax-1 bytes for it;
			// every other byte is either written now or replaced, so a
			// permanently invalid byte cannot make pending grow without bound.
			keep := incompleteUTF8Suffix(data)
			if keep > utf8.UTFMax-1 {
				keep = utf8.UTFMax - 1
			}
			prefix := data[:len(data)-keep]
			if len(prefix) > 0 {
				if !utf8.Valid(prefix) {
					prefix = bytes.ToValidUTF8(prefix, []byte("\uFFFD"))
				}
				if err := writeEscapedJSONStringBody(w, prefix); err != nil {
					return err
				}
			}
			pending = append(pending[:0], data[len(data)-keep:]...)
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			return readErr
		}
	}
	if len(pending) > 0 {
		if err := writeEscapedJSONStringBody(w, pending); err != nil {
			return err
		}
	}
	return nil
}

// incompleteUTF8Suffix returns the length of an unfinished UTF-8 sequence at
// the end of data. Invalid bytes are not treated as unfinished: they are
// reported as complete (width-1) runes by utf8.FullRune and get replaced by
// the caller.
func incompleteUTF8Suffix(data []byte) int {
	if len(data) == 0 {
		return 0
	}
	start := len(data) - utf8.UTFMax
	if start < 0 {
		start = 0
	}
	for i := len(data) - 1; i >= start; i-- {
		if data[i]&0xC0 == 0x80 {
			continue
		}
		if utf8.FullRune(data[i:]) {
			return 0
		}
		return len(data) - i
	}
	return 0
}

func writeEscapedJSONStringBody(w io.Writer, data []byte) error {
	escaped, err := json.Marshal(string(data))
	if err != nil {
		return err
	}
	if len(escaped) < 2 {
		return fmt.Errorf("invalid escaped JSON string")
	}
	_, err = w.Write(escaped[1 : len(escaped)-1])
	return err
}
