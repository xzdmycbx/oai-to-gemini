package convert

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// InvalidUpstreamResponseError reports a syntactically valid upstream JSON
// response that lacks the structure required for a successful completion.
type InvalidUpstreamResponseError struct {
	Reason string
}

func (e *InvalidUpstreamResponseError) Error() string {
	return "invalid upstream response: " + e.Reason
}

// ValidateGeminiResponse checks that a complete upstream response can be
// converted. A response without candidates is only valid when Gemini
// explicitly reports a prompt block reason.
func ValidateGeminiResponse(resp *GeminiResponse) error {
	if resp == nil {
		return &InvalidUpstreamResponseError{Reason: "empty response"}
	}
	blocked := resp.PromptFeedback != nil && strings.TrimSpace(resp.PromptFeedback.BlockReason) != ""
	if len(resp.Candidates) == 0 {
		if blocked {
			return nil
		}
		return &InvalidUpstreamResponseError{Reason: "response contains no candidates and no prompt block reason"}
	}
	candidate := resp.Candidates[0]
	if !blocked && len(candidate.Content.Parts) == 0 && candidate.FinishReason == "" {
		return &InvalidUpstreamResponseError{Reason: "candidate contains no content and no finish reason"}
	}
	return nil
}

// UnsupportedContentError reports a Gemini response content type that this
// proxy cannot convert. It is a distinct type so all response paths can map it
// to the same OpenAI error code.
type UnsupportedContentError struct {
	Message string
}

func (e *UnsupportedContentError) Error() string {
	return e.Message
}

// UnsupportedContentError reports part types this proxy does not model. The
// JSON decoder retains the original field names so they can be named in the
// error rather than being silently ignored.
func (p GeminiPart) UnsupportedContentError() error {
	if len(p.unknownFields) == 0 {
		return nil
	}
	return &UnsupportedContentError{
		Message: "unsupported Gemini response content type " + strings.Join(p.unknownFields, ", "),
	}
}

// ConcatTextParts concatenates all non-thought text parts verbatim. No
// newline, space, or any other separator is inserted.
func ConcatTextParts(parts []GeminiPart) string {
	var b strings.Builder
	for _, part := range parts {
		if part.Text == nil {
			continue
		}
		if part.Thought != nil && *part.Thought {
			continue
		}
		b.WriteString(*part.Text)
	}
	return b.String()
}

// MapFinishReason maps Gemini finish reasons to OpenAI finish reasons. An
// empty result means the reason is abnormal and cannot be represented as a
// successful OpenAI finish_reason.
func MapFinishReason(reason string) string {
	switch reason {
	case "STOP", "", "FINISH_REASON_UNSPECIFIED":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
		"IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION":
		return "content_filter"
	default:
		return ""
	}
}

// MapPromptBlockReason maps any non-empty prompt block reason to
// content_filter. A blocked prompt must never be reported as a normal stop,
// regardless of the specific reason value.
func MapPromptBlockReason(reason string) string {
	if reason == "" {
		return ""
	}
	return "content_filter"
}

// AbnormalFinishError reports a Gemini finish reason that cannot be
// represented as a successful OpenAI finish_reason.
type AbnormalFinishError struct {
	Reason string
}

func (e *AbnormalFinishError) Error() string {
	return fmt.Sprintf("Gemini finish reason %q indicates abnormal termination", e.Reason)
}

// FinishReasonFor maps a Gemini finish reason to an OpenAI finish_reason. It
// returns an error for abnormal termination reasons such as
// MALFORMED_FUNCTION_CALL or UNEXPECTED_TOOL_CALL instead of reporting them as
// a normal stop or tool_calls completion.
func FinishReasonFor(reason string, hasToolCalls bool) (string, error) {
	// Prompt blocking is handled separately and passes an already-mapped
	// OpenAI reason here.
	if reason == "length" || reason == "content_filter" {
		return reason, nil
	}
	mapped := MapFinishReason(reason)
	if mapped == "" {
		return "", &AbnormalFinishError{Reason: reason}
	}
	// Abnormal termination takes precedence over the presence of tool calls:
	// a length-limited or filtered tool call is not a normal tool_calls finish.
	if mapped == "length" || mapped == "content_filter" {
		return mapped, nil
	}
	if hasToolCalls {
		return "tool_calls", nil
	}
	return mapped, nil
}

func UsageFromGemini(u *GeminiUsage) *OpenAIUsage {
	if u == nil {
		return nil
	}
	out := &OpenAIUsage{
		PromptTokens:     u.PromptTokenCount,
		CompletionTokens: u.CandidatesTokenCount + u.ThoughtsTokenCount,
		TotalTokens:      u.TotalTokenCount,
	}
	if u.CachedContentTokenCount > 0 {
		out.PromptTokensDetails = &OpenAIPromptTokensDetails{CachedTokens: u.CachedContentTokenCount}
	}
	if u.ThoughtsTokenCount > 0 {
		out.CompletionTokensDetails = &OpenAICompletionTokensDetails{ReasoningTokens: u.ThoughtsTokenCount}
	}
	return out
}

// GeminiResponseToOpenAI converts a complete Gemini response to an
// OpenAI-compatible chat completion. Unsupported non-text output is an error,
// never a silently empty successful response.
func GeminiResponseToOpenAI(resp *GeminiResponse, model string) (*OpenAIResponse, error) {
	if err := ValidateGeminiResponse(resp); err != nil {
		return nil, err
	}
	out := &OpenAIResponse{
		ID:      newID("chatcmpl-"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Usage:   UsageFromGemini(resp.UsageMetadata),
	}

	var candidate GeminiCandidate
	if len(resp.Candidates) > 0 {
		candidate = resp.Candidates[0]
	}

	text, contentParts, err := ContentFromGeminiParts(candidate.Content.Parts)
	if err != nil {
		return nil, err
	}
	toolCalls := ToolCallsFromParts(candidate.Content.Parts)

	var content *string
	if len(contentParts) == 0 && (text != "" || len(toolCalls) == 0) {
		content = &text
	}

	finishReason := candidate.FinishReason
	if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
		finishReason = MapPromptBlockReason(resp.PromptFeedback.BlockReason)
	}
	mappedFinishReason, err := FinishReasonFor(finishReason, len(toolCalls) > 0)
	if err != nil {
		return nil, err
	}

	out.Choices = []OpenAIChoice{{
		Index: 0,
		Message: OpenAIMessageOut{
			Role:         "assistant",
			Content:      content,
			ContentParts: contentParts,
			ToolCalls:    toolCalls,
		},
		FinishReason: mappedFinishReason,
	}}
	return out, nil
}

// ContentFromGeminiParts converts text and supported non-text parts to an
// OpenAI-compatible content representation. Text parts are concatenated
// verbatim without separators. When contentParts is non-nil, text is also
// represented inside contentParts so ordering is preserved.
func ContentFromGeminiParts(parts []GeminiPart) (string, []OpenAIContentPart, error) {
	var allText strings.Builder
	var pendingText strings.Builder
	var contentParts []OpenAIContentPart
	hasNonText := false

	flushText := func() {
		if pendingText.Len() == 0 {
			return
		}
		contentParts = append(contentParts, OpenAIContentPart{Type: "text", Text: pendingText.String()})
		pendingText.Reset()
	}

	for _, part := range parts {
		if err := part.UnsupportedContentError(); err != nil {
			return "", nil, err
		}
		if part.Thought != nil && *part.Thought {
			continue
		}
		switch {
		case part.Text != nil:
			allText.WriteString(*part.Text)
			pendingText.WriteString(*part.Text)
		case part.FunctionCall != nil:
			// Tool calls are converted separately.
		case part.InlineData != nil:
			contentPart, err := InlineDataToContentPart(part.InlineData)
			if err != nil {
				return "", nil, err
			}
			flushText()
			contentParts = append(contentParts, *contentPart)
			hasNonText = true
		case part.FileData != nil:
			return "", nil, &UnsupportedContentError{Message: "unsupported Gemini response content type fileData"}
		case part.FunctionResponse != nil:
			return "", nil, &UnsupportedContentError{Message: "unsupported Gemini response content type functionResponse"}
		default:
			return "", nil, &UnsupportedContentError{Message: "unsupported Gemini response content part with no recognized content"}
		}
	}

	text := allText.String()
	if !hasNonText {
		return text, nil, nil
	}
	flushText()
	return text, contentParts, nil
}

// MaxImageMIMETypeBytes bounds the MIME type metadata retained for an inline
// image. Real image MIME types are far shorter than this.
const MaxImageMIMETypeBytes = 255

// ValidateInlineImageMetadata checks inline image metadata without looking at
// the payload bytes, so streaming callers can keep the data in spill storage.
func ValidateInlineImageMetadata(data *GeminiInlineData) (string, error) {
	if data == nil {
		return "", &UnsupportedContentError{Message: "unsupported Gemini response content type inlineData"}
	}
	mimeType := strings.TrimSpace(data.MIMEType)
	if mimeType == "" {
		return "", &UnsupportedContentError{Message: "unsupported Gemini response content type inlineData without mimeType"}
	}
	if len(mimeType) > MaxImageMIMETypeBytes {
		return "", &UnsupportedContentError{Message: fmt.Sprintf("Gemini image mimeType exceeds %d bytes", MaxImageMIMETypeBytes)}
	}
	if !strings.HasPrefix(strings.ToLower(mimeType), "image/") {
		return "", &UnsupportedContentError{Message: fmt.Sprintf("unsupported Gemini response content type %q", mimeType)}
	}
	return mimeType, nil
}

// ValidateInlineImageData checks inline image metadata without building a data
// URL, so callers can write the payload to spill storage directly.
func ValidateInlineImageData(data *GeminiInlineData) (string, error) {
	mimeType, err := ValidateInlineImageMetadata(data)
	if err != nil {
		return "", err
	}
	if data.Data == "" {
		return "", &UnsupportedContentError{Message: "Gemini image inlineData is empty"}
	}
	return mimeType, nil
}

// InlineDataToContentPart converts Gemini inline image data to an OpenAI
// image_url content part. Non-image inline data is rejected explicitly.
func InlineDataToContentPart(data *GeminiInlineData) (*OpenAIContentPart, error) {
	mimeType, err := ValidateInlineImageData(data)
	if err != nil {
		return nil, err
	}
	return &OpenAIContentPart{
		Type:     "image_url",
		ImageURL: &OpenAIImageURL{URL: "data:" + mimeType + ";base64," + data.Data},
	}, nil
}

func ToolCallsFromParts(parts []GeminiPart) []OpenAIToolCallOut {
	var calls []OpenAIToolCallOut
	for _, part := range parts {
		if part.FunctionCall == nil {
			continue
		}
		id := part.FunctionCall.ID
		if id == "" {
			id = newID("call_")
		}
		args := strings.TrimSpace(string(part.FunctionCall.Args))
		if args == "" || !json.Valid([]byte(args)) {
			args = "{}"
		}
		calls = append(calls, OpenAIToolCallOut{
			ID:               id,
			Type:             "function",
			ThoughtSignature: part.ThoughtSignature,
			Function: OpenAIToolCallFuncOut{
				Name:      part.FunctionCall.Name,
				Arguments: args,
			},
		})
	}
	return calls
}

// ChunkBuilder builds OpenAI streaming chunks for one response.
type ChunkBuilder struct {
	id            string
	created       int64
	model         string
	toolIndex     int
	roleSent      bool
	toolCallsSent bool
}

func NewChunkBuilder(model string) *ChunkBuilder {
	return &ChunkBuilder{
		id:      newID("chatcmpl-"),
		created: time.Now().Unix(),
		model:   model,
	}
}

func (b *ChunkBuilder) baseChunk() *OpenAIChunk {
	return &OpenAIChunk{
		ID:      b.id,
		Object:  "chat.completion.chunk",
		Created: b.created,
		Model:   b.model,
	}
}

func (b *ChunkBuilder) RoleChunk() *OpenAIChunk {
	b.roleSent = true
	chunk := b.baseChunk()
	chunk.Choices = []OpenAIChunkChoice{{
		Index: 0,
		Delta: OpenAIDelta{Role: "assistant"},
	}}
	return chunk
}

func (b *ChunkBuilder) ContentChunk(text string) *OpenAIChunk {
	chunk := b.baseChunk()
	delta := OpenAIDelta{Content: &text}
	if !b.roleSent {
		delta.Role = "assistant"
		b.roleSent = true
	}
	chunk.Choices = []OpenAIChunkChoice{{
		Index: 0,
		Delta: delta,
	}}
	return chunk
}

func (b *ChunkBuilder) ContentPartsChunk(parts []OpenAIContentPart) *OpenAIChunk {
	chunk := b.baseChunk()
	delta := OpenAIDelta{ContentParts: parts}
	if !b.roleSent {
		delta.Role = "assistant"
		b.roleSent = true
	}
	chunk.Choices = []OpenAIChunkChoice{{
		Index: 0,
		Delta: delta,
	}}
	return chunk
}

func (b *ChunkBuilder) ToolCallChunk(fc *GeminiFunctionCall, thoughtSignature ...string) *OpenAIChunk {
	id := fc.ID
	if id == "" {
		id = newID("call_")
	}
	args := strings.TrimSpace(string(fc.Args))
	if args == "" || !json.Valid([]byte(args)) {
		args = "{}"
	}

	signature := ""
	if len(thoughtSignature) > 0 {
		signature = thoughtSignature[0]
	}

	chunk := b.baseChunk()
	delta := OpenAIDelta{
		ToolCalls: []OpenAIToolCallDelta{{
			Index:            b.toolIndex,
			ID:               id,
			Type:             "function",
			ThoughtSignature: signature,
			Function: OpenAIToolCallFuncDelta{
				Name:      fc.Name,
				Arguments: args,
			},
		}},
	}
	if !b.roleSent {
		delta.Role = "assistant"
		b.roleSent = true
	}
	b.toolIndex++
	b.toolCallsSent = true
	chunk.Choices = []OpenAIChunkChoice{{
		Index: 0,
		Delta: delta,
	}}
	return chunk
}

func (b *ChunkBuilder) FinishChunk(finishReason string, usage *GeminiUsage) (*OpenAIChunk, error) {
	reason, err := FinishReasonFor(finishReason, b.toolCallsSent)
	if err != nil {
		return nil, err
	}
	chunk := b.baseChunk()
	chunk.Choices = []OpenAIChunkChoice{{
		Index:        0,
		Delta:        OpenAIDelta{},
		FinishReason: &reason,
	}}
	chunk.Usage = UsageFromGemini(usage)
	return chunk, nil
}
