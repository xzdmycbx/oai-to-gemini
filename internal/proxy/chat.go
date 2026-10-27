package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/xzdmycbx/oai-to-gemini/internal/config"
	"github.com/xzdmycbx/oai-to-gemini/internal/convert"
)

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request, provider config.Provider) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBodyBytes+1))
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "failed to read request body", "invalid_request_error", "invalid_body")
		return
	}
	if len(body) > maxRequestBodyBytes {
		writeOpenAIError(w, http.StatusRequestEntityTooLarge, "request body too large", "invalid_request_error", "body_too_large")
		return
	}

	req, raw, err := convert.ParseChatRequest(body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error(), "invalid_request_error", "invalid_json")
		return
	}
	if req.Model == "" {
		writeOpenAIError(w, http.StatusBadRequest, "model is required", "invalid_request_error", "missing_model")
		return
	}

	// stream=true always wins over aggregate_sse, including validation of the
	// aggregate switch. It is intentionally parsed only for non-stream requests.
	if req.Stream {
		s.handleChatStream(w, r, provider, req, raw)
		return
	}

	aggregateSSE, err := parseAggregateSSE(raw, r.URL.Query(), r.Header)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "invalid_aggregate_sse")
		return
	}
	if aggregateSSE {
		s.handleChatAggregate(w, r, provider, req, raw)
		return
	}
	s.handleChatNonStream(w, r, provider, req, raw)
}

func (s *Server) handleChatNonStream(w http.ResponseWriter, r *http.Request, provider config.Provider, req *convert.ChatRequest, raw map[string]json.RawMessage) {
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

	path := "/models/" + url.PathEscape(req.Model) + ":generateContent"
	upstream, err := s.newUpstreamRequest(r.Context(), r, provider, http.MethodPost, path, payload, nil)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, "failed to build upstream request", "server_error", "upstream_request")
		return
	}

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

	data, tooLarge, err := readAllLimited(resp.Body, maxUpstreamJSONBytes)
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
		writeOpenAIError(w, http.StatusBadGateway, "invalid upstream JSON: "+err.Error(), "upstream_error", "invalid_upstream_json")
		return
	}
	if msg := embeddedGeminiError(geminiResp.Error); msg != "" {
		writeOpenAIError(w, http.StatusBadGateway, msg, "upstream_error", "upstream_error")
		return
	}

	out, err := convert.GeminiResponseToOpenAI(geminiResp, req.Model)
	if err != nil {
		writeConversionError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, out)
}

const (
	maxAggregatedToolCalls          = 4096
	maxAggregatedToolCallBytes      = 16 << 20
	maxAggregatedImages             = 4096
	maxAggregatedImageMetadataBytes = 1 << 20
)

func (s *Server) handleChatAggregate(w http.ResponseWriter, r *http.Request, provider config.Provider, req *convert.ChatRequest, raw map[string]json.RawMessage) {
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

	// Some Gemini-compatible upstreams ignore alt=sse and return a normal
	// JSON response. Accept that instead of failing.
	if !isEventStream(resp.Header.Get("Content-Type")) {
		data, tooLarge, err := readAllLimited(resp.Body, maxUpstreamJSONBytes)
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
		out, err := convert.GeminiResponseToOpenAI(geminiResp, req.Model)
		if err != nil {
			writeConversionError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, out)
		return
	}

	buf := s.store.NewBuffer()
	defer func() { _ = buf.Close() }()
	imageBuf := s.store.NewBuffer()
	defer func() { _ = imageBuf.Close() }()

	var functionParts []convert.GeminiPart
	var toolCallBytes int
	var storedImages []convert.StoredImagePart
	var contentOffsets []int64
	var imageMetadataBytes int
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
				if payload == nil {
					return fmt.Errorf("missing stored text payload")
				}
				if err := copyStoredRange(buf, payload, *part.TextRange); err != nil {
					return err
				}
			case part.Text != nil:
				if _, err := buf.WriteString(*part.Text); err != nil {
					return err
				}
			case part.FunctionCall != nil:
				// Count every field retained by the function part, including
				// the ID, so the in-memory metadata budget covers the whole
				// object rather than just its name and arguments.
				toolCallBytes += len(part.FunctionCall.Name) + len(part.FunctionCall.ID) +
					len(part.FunctionCall.Args) + len(part.ThoughtSignature)
				if len(functionParts) >= maxAggregatedToolCalls || toolCallBytes > maxAggregatedToolCallBytes {
					return fmt.Errorf("aggregated tool call data exceeds limit")
				}
				functionParts = append(functionParts, convert.GeminiPart{
					FunctionCall:     part.FunctionCall,
					ThoughtSignature: part.ThoughtSignature,
				})
			case part.InlineData != nil:
				var (
					mimeType string
					length   int64
					write    func() error
				)
				if part.InlineDataRange != nil {
					if payload == nil {
						return fmt.Errorf("missing stored image payload")
					}
					mimeType, err = convert.ValidateInlineImageMetadata(part.InlineData)
					if err != nil {
						return err
					}
					if part.InlineDataRange.Length == 0 {
						return &convert.UnsupportedContentError{Message: "Gemini image inlineData is empty"}
					}
					length = part.InlineDataRange.Length
					write = func() error {
						return copyStoredRange(imageBuf, payload, *part.InlineDataRange)
					}
				} else {
					mimeType, err = convert.ValidateInlineImageData(part.InlineData)
					if err != nil {
						return err
					}
					length = int64(len(part.InlineData.Data))
					write = func() error {
						_, err := imageBuf.WriteString(part.InlineData.Data)
						return err
					}
				}
				// Each image retains a MIME type plus an offset, a length and
				// a content offset in memory. Bound both the count and the
				// metadata bytes; the image payload itself already spills to
				// disk.
				metadataBytes := len(mimeType) + 3*8
				if len(storedImages) >= maxAggregatedImages || imageMetadataBytes+metadataBytes > maxAggregatedImageMetadataBytes {
					return fmt.Errorf("aggregated image metadata exceeds limit")
				}
				offset := imageBuf.Size()
				if err := write(); err != nil {
					return err
				}
				imageMetadataBytes += metadataBytes
				storedImages = append(storedImages, convert.StoredImagePart{
					MIMEType: mimeType,
					Offset:   offset,
					Length:   length,
				})
				contentOffsets = append(contentOffsets, buf.Size())
			case part.FileData != nil:
				return &convert.UnsupportedContentError{Message: "unsupported Gemini response content type fileData"}
			case part.FunctionResponse != nil:
				return &convert.UnsupportedContentError{Message: "unsupported Gemini response content type functionResponse"}
			default:
				return &convert.UnsupportedContentError{Message: "unsupported Gemini response content part with no recognized content"}
			}
		}
		return nil
	})
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "upstream stream failed: "+err.Error(), "upstream_error", streamErrorCode(err))
		return
	}
	if !completed {
		writeOpenAIError(w, http.StatusBadGateway, "upstream stream ended before a final finishReason or [DONE]", "upstream_error", "upstream_incomplete")
		return
	}
	if !valid {
		writeConversionError(w, invalidStreamResponse())
		return
	}

	size := buf.Size()
	totalSize := size + imageBuf.Size()
	if totalSize > s.cfg.SSECacheMaxBytes {
		// The limit is a cache soft limit, not a truncation limit. The full
		// response is returned and temp files are cleaned up by defer.
		s.logger.Printf("aggregate response size %d exceeds sse_cache_max_bytes %d; returning full response and cleaning up", totalSize, s.cfg.SSECacheMaxBytes)
	}

	if promptBlockReason != "" {
		finishReason = convert.MapPromptBlockReason(promptBlockReason)
	}
	if _, err := convert.FinishReasonFor(finishReason, len(functionParts) > 0); err != nil {
		writeOpenAIError(w, http.StatusBadGateway, err.Error(), "upstream_error", "abnormal_finish")
		return
	}

	var content io.ReadCloser
	if size > 0 {
		content, err = buf.Reader()
		if err != nil {
			writeOpenAIError(w, http.StatusInternalServerError, "failed to read aggregated response", "server_error", "aggregate_read")
			return
		}
		defer func() { _ = content.Close() }()
	}

	var imageData io.ReaderAt
	if imageBuf.Size() > 0 {
		readerAt, closer, err := imageBuf.ReaderAt()
		if err != nil {
			writeOpenAIError(w, http.StatusInternalServerError, "failed to read aggregated image data", "server_error", "aggregate_read")
			return
		}
		defer func() { _ = closer.Close() }()
		imageData = readerAt
	}

	geminiResp := &convert.GeminiResponse{
		Candidates: []convert.GeminiCandidate{{
			Content:      convert.GeminiContent{Role: "model", Parts: functionParts},
			FinishReason: finishReason,
		}},
		UsageMetadata: usage,
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if err := convert.WriteOpenAIResponseJSONWithStoredParts(w, geminiResp, req.Model, content, contentOffsets, size, storedImages, imageData); err != nil {
		s.logger.Printf("write aggregated response failed: %v", err)
	}
}

func isThought(part convert.GeminiPart) bool {
	return part.Thought != nil && *part.Thought
}

func isEventStream(contentType string) bool {
	return strings.HasPrefix(strings.ToLower(contentType), "text/event-stream")
}
