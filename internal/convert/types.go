package convert

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

// ---------------------------------------------------------------------------
// OpenAI-compatible request types
// ---------------------------------------------------------------------------

type ChatRequest struct {
	Model               string          `json:"model"`
	Messages            []Message       `json:"messages"`
	Stream              bool            `json:"stream"`
	N                   *int            `json:"n"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Stop                json.RawMessage `json:"stop"`
	Tools               []Tool          `json:"tools"`
	ToolChoice          json.RawMessage `json:"tool_choice"`
	ResponseFormat      json.RawMessage `json:"response_format"`
}

type Message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	Name       string          `json:"name"`
	ToolCallID string          `json:"tool_call_id"`
	ToolCalls  []ToolCall      `json:"tool_calls"`
}

type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
	// ThoughtSignature round-trips Gemini's thoughtSignature through an
	// OpenAI-compatible extension field. Both snake_case and camelCase are
	// accepted on input; snake_case is emitted on output.
	ThoughtSignature      string `json:"thought_signature"`
	ThoughtSignatureCamel string `json:"thoughtSignature"`
}

func (tc ToolCall) signature() string {
	if tc.ThoughtSignature != "" {
		return tc.ThoughtSignature
	}
	return tc.ThoughtSignatureCamel
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type contentPart struct {
	Type       string          `json:"type"`
	Text       string          `json:"text"`
	ImageURL   json.RawMessage `json:"image_url"`
	InputAudio *struct {
		Data   string `json:"data"`
		Format string `json:"format"`
	} `json:"input_audio"`
	File *struct {
		FileData string `json:"file_data"`
		Filename string `json:"filename"`
		FileID   string `json:"file_id"`
	} `json:"file"`
}

// ParseChatRequest decodes both the typed request and the raw top-level object
// so that Gemini-native passthrough fields can be handled without losing them.
func ParseChatRequest(body []byte) (*ChatRequest, map[string]json.RawMessage, error) {
	if err := checkChatRequestBudgets(body); err != nil {
		return nil, nil, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, nil, err
	}
	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, nil, err
	}
	return &req, raw, nil
}

// ---------------------------------------------------------------------------
// Gemini-compatible request/response types
// ---------------------------------------------------------------------------

type GeminiRequest struct {
	Contents          []GeminiContent            `json:"contents"`
	SystemInstruction *GeminiContent             `json:"systemInstruction,omitempty"`
	Tools             []GeminiTool               `json:"tools,omitempty"`
	ToolConfig        *GeminiToolConfig          `json:"toolConfig,omitempty"`
	GenerationConfig  map[string]json.RawMessage `json:"generationConfig,omitempty"`
	SafetySettings    json.RawMessage            `json:"safetySettings,omitempty"`
}

type GeminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []GeminiPart `json:"parts"`
}

type GeminiPart struct {
	Text             *string                 `json:"text,omitempty"`
	Thought          *bool                   `json:"thought,omitempty"`
	ThoughtSignature string                  `json:"thoughtSignature,omitempty"`
	FunctionCall     *GeminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *GeminiFunctionResponse `json:"functionResponse,omitempty"`
	InlineData       *GeminiInlineData       `json:"inlineData,omitempty"`
	FileData         *GeminiFileData         `json:"fileData,omitempty"`

	// TextRange and InlineDataRange are populated by the streaming SSE parser
	// instead of materializing large payloads. Offset and Length refer to the
	// event-local payload store. Text/InlineData still describe field presence
	// and metadata so callers can keep the same validation rules.
	TextRange       *StoredRange `json:"-"`
	InlineDataRange *StoredRange `json:"-"`

	unknownFields []string
}

// StoredRange identifies a byte range in an event-local payload store.
type StoredRange struct {
	Offset int64
	Length int64
}

var knownGeminiPartFields = map[string]struct{}{
	"text":             {},
	"thought":          {},
	"thoughtSignature": {},
	"functionCall":     {},
	"functionResponse": {},
	"inlineData":       {},
	"fileData":         {},
}

// UnmarshalJSON records unrecognized fields so response conversion can reject
// them explicitly instead of silently dropping a legal Gemini part type that
// this proxy does not model.
func (p *GeminiPart) UnmarshalJSON(data []byte) error {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		*p = GeminiPart{}
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '{' {
		return fmt.Errorf("Gemini part must be a JSON object")
	}

	*p = GeminiPart{}
	for dec.More() {
		keyToken, err := dec.Token()
		if err != nil {
			return err
		}
		name, ok := keyToken.(string)
		if !ok {
			return fmt.Errorf("Gemini part field name is not a string")
		}
		switch name {
		case "text":
			if err := dec.Decode(&p.Text); err != nil {
				return err
			}
		case "thought":
			if err := dec.Decode(&p.Thought); err != nil {
				return err
			}
		case "thoughtSignature":
			if err := dec.Decode(&p.ThoughtSignature); err != nil {
				return err
			}
		case "functionCall":
			if err := dec.Decode(&p.FunctionCall); err != nil {
				return err
			}
		case "functionResponse":
			if err := dec.Decode(&p.FunctionResponse); err != nil {
				return err
			}
		case "inlineData":
			if err := dec.Decode(&p.InlineData); err != nil {
				return err
			}
		case "fileData":
			if err := dec.Decode(&p.FileData); err != nil {
				return err
			}
		default:
			var ignored json.RawMessage
			if err := dec.Decode(&ignored); err != nil {
				return err
			}
			p.unknownFields = append(p.unknownFields, name)
		}
	}
	if _, err := dec.Token(); err != nil {
		return err
	}
	sort.Strings(p.unknownFields)
	return nil
}

type GeminiInlineData struct {
	MIMEType string `json:"mimeType"`
	Data     string `json:"data"`
}

type GeminiFileData struct {
	MIMEType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type GeminiFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
	ID   string          `json:"id,omitempty"`
}

type GeminiFunctionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
	ID       string          `json:"id,omitempty"`
}

type GeminiTool struct {
	FunctionDeclarations []GeminiFunctionDeclaration `json:"functionDeclarations"`
}

type GeminiFunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type GeminiToolConfig struct {
	FunctionCallingConfig GeminiFunctionCallingConfig `json:"functionCallingConfig"`
}

type GeminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type GeminiResponse struct {
	Candidates     []GeminiCandidate     `json:"candidates"`
	UsageMetadata  *GeminiUsage          `json:"usageMetadata,omitempty"`
	PromptFeedback *GeminiPromptFeedback `json:"promptFeedback,omitempty"`
	ModelVersion   string                `json:"modelVersion,omitempty"`
	ResponseID     string                `json:"responseId,omitempty"`
	Error          json.RawMessage       `json:"error,omitempty"`
}

// ParseGeminiResponseJSON decodes a complete Gemini JSON response after
// bounding the decoded candidate, part and part-metadata counts. The raw
// response byte limit alone would still allow a shallow but extremely wide
// array to allocate far more memory than its serialized size.
func ParseGeminiResponseJSON(data []byte) (*GeminiResponse, error) {
	if err := checkGeminiResponseBudgets(data); err != nil {
		return nil, err
	}
	var resp GeminiResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

type GeminiPromptFeedback struct {
	BlockReason   string          `json:"blockReason,omitempty"`
	SafetyRatings json.RawMessage `json:"safetyRatings,omitempty"`
}

type GeminiCandidate struct {
	Content       GeminiContent   `json:"content"`
	FinishReason  string          `json:"finishReason,omitempty"`
	Index         int             `json:"index,omitempty"`
	SafetyRatings json.RawMessage `json:"safetyRatings,omitempty"`
}

type GeminiUsage struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
}

// ---------------------------------------------------------------------------
// OpenAI-compatible response types
// ---------------------------------------------------------------------------

type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
}

type OpenAIChoice struct {
	Index        int              `json:"index"`
	Message      OpenAIMessageOut `json:"message"`
	FinishReason string           `json:"finish_reason"`
}

// OpenAIContentPart represents a non-text response part. Gemini image output
// is exposed as an OpenAI-compatible image_url content part.
type OpenAIContentPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *OpenAIImageURL `json:"image_url,omitempty"`
}

type OpenAIImageURL struct {
	URL string `json:"url"`
}

type OpenAIMessageOut struct {
	Role         string              `json:"role"`
	Content      *string             `json:"content"`
	ContentParts []OpenAIContentPart `json:"-"`
	ToolCalls    []OpenAIToolCallOut `json:"tool_calls,omitempty"`
}

// MarshalJSON emits content as a string when possible, or as an array of
// content parts for multimodal responses.
func (m OpenAIMessageOut) MarshalJSON() ([]byte, error) {
	type messageJSON struct {
		Role      string              `json:"role"`
		Content   any                 `json:"content"`
		ToolCalls []OpenAIToolCallOut `json:"tool_calls,omitempty"`
	}
	var content any
	switch {
	case len(m.ContentParts) > 0:
		content = m.ContentParts
	case m.Content != nil:
		content = *m.Content
	}
	return json.Marshal(messageJSON{Role: m.Role, Content: content, ToolCalls: m.ToolCalls})
}

type OpenAIToolCallOut struct {
	ID               string                `json:"id"`
	Type             string                `json:"type"`
	Function         OpenAIToolCallFuncOut `json:"function"`
	ThoughtSignature string                `json:"thought_signature,omitempty"`
}

type OpenAIToolCallFuncOut struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAIChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []OpenAIChunkChoice `json:"choices"`
	Usage   *OpenAIUsage        `json:"usage,omitempty"`
}

type OpenAIChunkChoice struct {
	Index        int         `json:"index"`
	Delta        OpenAIDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type OpenAIDelta struct {
	Role         string                `json:"role,omitempty"`
	Content      *string               `json:"content,omitempty"`
	ContentParts []OpenAIContentPart   `json:"-"`
	ToolCalls    []OpenAIToolCallDelta `json:"tool_calls,omitempty"`
}

// MarshalJSON emits content as a string or an array of content parts. An
// empty delta (used by finish chunks) remains {}.
func (d OpenAIDelta) MarshalJSON() ([]byte, error) {
	type deltaJSON struct {
		Role      string                `json:"role,omitempty"`
		Content   any                   `json:"content,omitempty"`
		ToolCalls []OpenAIToolCallDelta `json:"tool_calls,omitempty"`
	}
	out := deltaJSON{Role: d.Role, ToolCalls: d.ToolCalls}
	switch {
	case len(d.ContentParts) > 0:
		out.Content = d.ContentParts
	case d.Content != nil:
		out.Content = *d.Content
	}
	return json.Marshal(out)
}

type OpenAIToolCallDelta struct {
	Index            int                     `json:"index"`
	ID               string                  `json:"id,omitempty"`
	Type             string                  `json:"type,omitempty"`
	Function         OpenAIToolCallFuncDelta `json:"function"`
	ThoughtSignature string                  `json:"thought_signature,omitempty"`
}

type OpenAIToolCallFuncDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens            int                            `json:"prompt_tokens"`
	CompletionTokens        int                            `json:"completion_tokens"`
	TotalTokens             int                            `json:"total_tokens"`
	PromptTokensDetails     *OpenAIPromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *OpenAICompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type OpenAIPromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

type OpenAICompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

func newID(prefix string) string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return prefix + "fallback"
	}
	return prefix + hex.EncodeToString(b[:])
}
