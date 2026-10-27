package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// ToGemini converts an OpenAI-compatible chat request into a Gemini
// generateContent request body.
func ToGemini(req *ChatRequest, raw map[string]json.RawMessage) (*GeminiRequest, error) {
	if err := validateOpenAICandidateCount(req.N); err != nil {
		return nil, err
	}
	out := &GeminiRequest{}

	systemText, err := collectSystemText(req.Messages)
	if err != nil {
		return nil, err
	}
	if systemText != "" {
		text := systemText
		out.SystemInstruction = &GeminiContent{Parts: []GeminiPart{{Text: &text}}}
	}

	callNames := make(map[string]string)
	for _, msg := range req.Messages {
		if msg.Role != "assistant" {
			continue
		}
		for _, tc := range msg.ToolCalls {
			if tc.ID != "" && tc.Function.Name != "" {
				callNames[tc.ID] = tc.Function.Name
			}
		}
	}

	lastWasTool := false
	for _, msg := range req.Messages {
		if msg.Role != "tool" {
			lastWasTool = false
		}
		switch msg.Role {
		case "system", "developer":
			// Already handled above.
		case "user":
			parts, err := textParts(msg.Content)
			if err != nil {
				return nil, fmt.Errorf("user message content: %w", err)
			}
			if len(parts) > 0 {
				out.Contents = append(out.Contents, GeminiContent{Role: "user", Parts: parts})
			}
		case "assistant":
			parts, err := textParts(msg.Content)
			if err != nil {
				return nil, fmt.Errorf("assistant message content: %w", err)
			}
			for _, tc := range msg.ToolCalls {
				args := toolArguments(tc.Function.Arguments)
				fc := &GeminiFunctionCall{Name: tc.Function.Name, Args: args}
				if tc.ID != "" {
					fc.ID = tc.ID
				}
				parts = append(parts, GeminiPart{
					FunctionCall:     fc,
					ThoughtSignature: tc.signature(),
				})
			}
			if len(parts) > 0 {
				out.Contents = append(out.Contents, GeminiContent{Role: "model", Parts: parts})
			}
		case "tool":
			name := callNames[msg.ToolCallID]
			if name == "" {
				name = msg.Name
			}
			if name == "" {
				name = "tool"
			}
			response, err := toolResponse(msg.Content)
			if err != nil {
				return nil, fmt.Errorf("tool message content: %w", err)
			}
			fr := &GeminiFunctionResponse{
				Name:     name,
				Response: response,
			}
			if msg.ToolCallID != "" {
				fr.ID = msg.ToolCallID
			}
			part := GeminiPart{FunctionResponse: fr}
			// Parallel tool results belong to the same Gemini user turn. Merge
			// consecutive role:"tool" messages into one parts array so the
			// number of functionResponse parts matches the preceding
			// functionCall parts.
			if lastWasTool && len(out.Contents) > 0 && out.Contents[len(out.Contents)-1].Role == "user" {
				last := &out.Contents[len(out.Contents)-1]
				last.Parts = append(last.Parts, part)
			} else {
				out.Contents = append(out.Contents, GeminiContent{
					Role:  "user",
					Parts: []GeminiPart{part},
				})
			}
			lastWasTool = true
		default:
			// Unknown roles are treated as user content rather than dropped.
			parts, err := textParts(msg.Content)
			if err != nil {
				return nil, fmt.Errorf("%s message content: %w", msg.Role, err)
			}
			if len(parts) > 0 {
				out.Contents = append(out.Contents, GeminiContent{Role: "user", Parts: parts})
			}
		}
	}

	out.Tools, out.ToolConfig = buildTools(req)
	out.GenerationConfig, err = buildGenerationConfig(req, raw)
	if err != nil {
		return nil, err
	}
	if err := validateCandidateCount(out.GenerationConfig); err != nil {
		return nil, err
	}

	for _, key := range []string{"safetySettings", "safety_settings"} {
		if v, ok := raw[key]; ok && !isJSONNull(v) {
			out.SafetySettings = v
			break
		}
	}

	return out, nil
}

func collectSystemText(messages []Message) (string, error) {
	var b strings.Builder
	for _, msg := range messages {
		if msg.Role != "system" && msg.Role != "developer" {
			continue
		}
		text, err := contentText(msg.Content)
		if err != nil {
			return "", fmt.Errorf("system message content: %w", err)
		}
		// Hard requirement: concatenate parts exactly, without adding any
		// newline or other separator.
		b.WriteString(text)
	}
	return b.String(), nil
}

// textParts converts OpenAI message content (string or array of content parts)
// into Gemini parts. Text parts are concatenated verbatim; supported
// multimodal parts are preserved in order. Unsupported part types return an
// error instead of being silently dropped.
func textParts(content json.RawMessage) ([]GeminiPart, error) {
	if len(content) == 0 || isJSONNull(content) {
		return nil, nil
	}

	var s string
	if err := json.Unmarshal(content, &s); err == nil {
		if s == "" {
			return nil, nil
		}
		return []GeminiPart{{Text: &s}}, nil
	}

	var parts []contentPart
	if err := json.Unmarshal(content, &parts); err != nil {
		return nil, fmt.Errorf("must be a string or an array of content parts")
	}

	out := make([]GeminiPart, 0, len(parts))
	var text strings.Builder
	flushText := func() {
		if text.Len() == 0 {
			return
		}
		s := text.String()
		out = append(out, GeminiPart{Text: &s})
		text.Reset()
	}

	for _, p := range parts {
		switch p.Type {
		case "text", "input_text":
			// Adjacent text parts are accumulated linearly and emitted as a
			// single part when a non-text part or the content array ends.
			text.WriteString(p.Text)
		case "image_url":
			flushText()
			part, err := imagePart(p.ImageURL)
			if err != nil {
				return nil, err
			}
			out = append(out, part)
		case "input_audio":
			flushText()
			part, err := audioPart(p.InputAudio)
			if err != nil {
				return nil, err
			}
			out = append(out, part)
		case "file":
			flushText()
			part, err := filePart(p.File)
			if err != nil {
				return nil, err
			}
			out = append(out, part)
		default:
			return nil, fmt.Errorf("unsupported content part type %q", p.Type)
		}
	}
	flushText()
	return out, nil
}

func contentText(content json.RawMessage) (string, error) {
	parts, err := textParts(content)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for _, part := range parts {
		if part.Text == nil {
			return "", fmt.Errorf("non-text content part is not supported here")
		}
		b.WriteString(*part.Text)
	}
	return b.String(), nil
}

func imagePart(raw json.RawMessage) (GeminiPart, error) {
	if len(raw) == 0 || isJSONNull(raw) {
		return GeminiPart{}, fmt.Errorf("image_url is required")
	}

	var imageURL string
	if err := json.Unmarshal(raw, &imageURL); err != nil {
		var obj struct {
			URL string `json:"url"`
		}
		if err := json.Unmarshal(raw, &obj); err != nil {
			return GeminiPart{}, fmt.Errorf("image_url must be a string or an object with a url field")
		}
		imageURL = obj.URL
	}
	imageURL = strings.TrimSpace(imageURL)
	if imageURL == "" {
		return GeminiPart{}, fmt.Errorf("image_url.url is required")
	}

	if mimeType, data, ok, err := parseDataURL(imageURL); err != nil {
		return GeminiPart{}, err
	} else if ok {
		return GeminiPart{InlineData: &GeminiInlineData{MIMEType: mimeType, Data: data}}, nil
	}
	if isHTTPURL(imageURL) {
		return GeminiPart{FileData: &GeminiFileData{FileURI: imageURL}}, nil
	}
	return GeminiPart{}, fmt.Errorf("unsupported image_url scheme")
}

func audioPart(audio *struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}) (GeminiPart, error) {
	if audio == nil || strings.TrimSpace(audio.Data) == "" {
		return GeminiPart{}, fmt.Errorf("input_audio.data is required")
	}
	mimeType := "application/octet-stream"
	if format := strings.TrimSpace(audio.Format); format != "" {
		mimeType = "audio/" + format
	}
	return GeminiPart{InlineData: &GeminiInlineData{MIMEType: mimeType, Data: audio.Data}}, nil
}

func filePart(file *struct {
	FileData string `json:"file_data"`
	Filename string `json:"filename"`
	FileID   string `json:"file_id"`
}) (GeminiPart, error) {
	if file == nil {
		return GeminiPart{}, fmt.Errorf("file is required")
	}
	fileData := strings.TrimSpace(file.FileData)
	if fileData != "" {
		if mimeType, data, ok, err := parseDataURL(fileData); err != nil {
			return GeminiPart{}, err
		} else if ok {
			return GeminiPart{InlineData: &GeminiInlineData{MIMEType: mimeType, Data: data}}, nil
		}
		if isHTTPURL(fileData) {
			return GeminiPart{FileData: &GeminiFileData{FileURI: fileData}}, nil
		}
		return GeminiPart{}, fmt.Errorf("unsupported file.file_data value")
	}
	if file.FileID != "" {
		return GeminiPart{}, fmt.Errorf("file.file_id is not supported; provide file_data instead")
	}
	return GeminiPart{}, fmt.Errorf("file.file_data is required")
}

func parseDataURL(value string) (mimeType, data string, ok bool, err error) {
	if !strings.HasPrefix(value, "data:") {
		return "", "", false, nil
	}
	comma := strings.IndexByte(value, ',')
	if comma < 0 {
		return "", "", false, fmt.Errorf("invalid data URL")
	}
	meta := value[len("data:"):comma]
	if !strings.HasSuffix(meta, ";base64") {
		return "", "", false, fmt.Errorf("only base64 data URLs are supported")
	}
	mimeType = strings.TrimSuffix(meta, ";base64")
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}
	data = value[comma+1:]
	if data == "" {
		return "", "", false, fmt.Errorf("data URL payload is empty")
	}
	return mimeType, data, true, nil
}

func isHTTPURL(value string) bool {
	return strings.HasPrefix(value, "http://") || strings.HasPrefix(value, "https://")
}

func validateOpenAICandidateCount(n *int) error {
	if n == nil {
		return nil
	}
	if *n > 1 {
		return fmt.Errorf("n greater than 1 is not supported")
	}
	return nil
}

func validateCandidateCount(gc map[string]json.RawMessage) error {
	if len(gc) == 0 {
		return nil
	}
	raw, ok := gc["candidateCount"]
	if !ok || isJSONNull(raw) {
		return nil
	}
	var count int
	if err := json.Unmarshal(raw, &count); err != nil {
		return fmt.Errorf("candidateCount must be an integer")
	}
	if count > 1 {
		return fmt.Errorf("candidateCount greater than 1 is not supported")
	}
	return nil
}

func toolArguments(arguments string) json.RawMessage {
	trimmed := strings.TrimSpace(arguments)
	if trimmed == "" || !json.Valid([]byte(trimmed)) {
		return json.RawMessage("{}")
	}
	return json.RawMessage(trimmed)
}

func toolResponse(content json.RawMessage) (json.RawMessage, error) {
	if len(content) == 0 || isJSONNull(content) {
		return json.RawMessage("{}"), nil
	}

	trimmed := bytes.TrimSpace(content)
	if len(trimmed) == 0 {
		return json.RawMessage("{}"), nil
	}
	if trimmed[0] == '[' {
		var items []json.RawMessage
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return nil, fmt.Errorf("invalid tool result JSON array")
		}
		if looksLikeContentPartArray(items) {
			var parts []contentPart
			if err := json.Unmarshal(trimmed, &parts); err != nil {
				return nil, fmt.Errorf("must be an array of content parts")
			}
			var b strings.Builder
			for _, part := range parts {
				switch part.Type {
				case "text", "input_text":
					b.WriteString(part.Text)
				default:
					return nil, fmt.Errorf("unsupported tool result content part type %q", part.Type)
				}
			}
			return wrapToolResponseText(b.String()), nil
		}
	}
	if trimmed[0] == '{' && json.Valid(trimmed) {
		return json.RawMessage(trimmed), nil
	}

	var s string
	if err := json.Unmarshal(trimmed, &s); err == nil {
		return wrapToolResponseText(s), nil
	}

	if json.Valid(trimmed) {
		wrapped, _ := json.Marshal(map[string]json.RawMessage{"result": trimmed})
		return wrapped, nil
	}
	wrapped, _ := json.Marshal(map[string]string{"result": string(trimmed)})
	return wrapped, nil
}

func looksLikeContentPartArray(items []json.RawMessage) bool {
	if len(items) == 0 {
		return false
	}
	for _, item := range items {
		if !jsonObjectHasTopLevelKey(item, "type") {
			return false
		}
	}
	return true
}

func wrapToolResponseText(s string) json.RawMessage {
	inner := strings.TrimSpace(s)
	if strings.HasPrefix(inner, "{") && json.Valid([]byte(inner)) {
		return json.RawMessage(inner)
	}
	wrapped, _ := json.Marshal(map[string]string{"result": s})
	return wrapped
}

func buildTools(req *ChatRequest) ([]GeminiTool, *GeminiToolConfig) {
	var decls []GeminiFunctionDeclaration
	for _, t := range req.Tools {
		if t.Type != "" && t.Type != "function" {
			continue
		}
		if t.Function.Name == "" {
			continue
		}
		decl := GeminiFunctionDeclaration{
			Name:        t.Function.Name,
			Description: t.Function.Description,
		}
		if len(t.Function.Parameters) > 0 && !isJSONNull(t.Function.Parameters) {
			decl.Parameters = t.Function.Parameters
		}
		decls = append(decls, decl)
	}

	var tools []GeminiTool
	if len(decls) > 0 {
		tools = []GeminiTool{{FunctionDeclarations: decls}}
	}
	return tools, buildToolConfig(req.ToolChoice)
}

func buildToolConfig(raw json.RawMessage) *GeminiToolConfig {
	if len(raw) == 0 || isJSONNull(raw) {
		return nil
	}

	var mode string
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto":
			mode = "AUTO"
		case "none":
			mode = "NONE"
		case "required":
			mode = "ANY"
		default:
			return nil
		}
		return &GeminiToolConfig{FunctionCallingConfig: GeminiFunctionCallingConfig{Mode: mode}}
	}

	var obj struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Type == "function" && obj.Function.Name != "" {
		return &GeminiToolConfig{FunctionCallingConfig: GeminiFunctionCallingConfig{
			Mode:                 "ANY",
			AllowedFunctionNames: []string{obj.Function.Name},
		}}
	}
	return nil
}

func buildGenerationConfig(req *ChatRequest, raw map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	var gc map[string]json.RawMessage

	if v, ok := raw["generationConfig"]; ok && !isJSONNull(v) {
		if err := json.Unmarshal(v, &gc); err != nil {
			return nil, fmt.Errorf("generationConfig: %w", err)
		}
	}
	if gc == nil {
		gc = make(map[string]json.RawMessage)
	}

	// Gemini-native generation options may be sent at the top level.
	for _, key := range []string{
		"topK", "candidateCount", "presencePenalty", "frequencyPenalty",
		"seed", "responseMimeType", "responseSchema", "responseLogprobs",
		"thinkingConfig", "mediaResolution",
	} {
		if v, ok := raw[key]; ok && !isJSONNull(v) {
			gc[key] = v
		}
	}
	// "logprobs" is ambiguous: Gemini expects a number, while OpenAI uses a
	// boolean. Only pass it through when it is numeric.
	if v, ok := raw["logprobs"]; ok && !isJSONNull(v) {
		var n float64
		if err := json.Unmarshal(v, &n); err == nil {
			gc["logprobs"] = v
		}
	}

	// Reasonable OpenAI aliases for Gemini generation options.
	if v, ok := raw["presence_penalty"]; ok && !isJSONNull(v) {
		gc["presencePenalty"] = v
	}
	if v, ok := raw["frequency_penalty"]; ok && !isJSONNull(v) {
		gc["frequencyPenalty"] = v
	}
	if v, ok := raw["seed"]; ok && !isJSONNull(v) {
		gc["seed"] = v
	}

	if req.Temperature != nil {
		gc["temperature"] = mustJSON(*req.Temperature)
	}
	if req.TopP != nil {
		gc["topP"] = mustJSON(*req.TopP)
	}
	if req.MaxTokens != nil {
		gc["maxOutputTokens"] = mustJSON(*req.MaxTokens)
	} else if req.MaxCompletionTokens != nil {
		gc["maxOutputTokens"] = mustJSON(*req.MaxCompletionTokens)
	}

	if stops, ok, err := parseStop(req.Stop); err != nil {
		return nil, err
	} else if ok {
		gc["stopSequences"] = mustJSON(stops)
	}

	if len(req.ResponseFormat) > 0 && !isJSONNull(req.ResponseFormat) {
		var rf struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Schema json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		}
		if err := json.Unmarshal(req.ResponseFormat, &rf); err == nil {
			switch rf.Type {
			case "json_object":
				gc["responseMimeType"] = mustJSON("application/json")
			case "json_schema":
				if len(rf.JSONSchema.Schema) > 0 && !isJSONNull(rf.JSONSchema.Schema) {
					gc["responseMimeType"] = mustJSON("application/json")
					gc["responseSchema"] = rf.JSONSchema.Schema
				}
			}
		}
	}

	if len(gc) == 0 {
		return nil, nil
	}
	return gc, nil
}

func parseStop(raw json.RawMessage) ([]string, bool, error) {
	if len(raw) == 0 || isJSONNull(raw) {
		return nil, false, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []string{s}, true, nil
	}
	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		if len(arr) == 0 {
			return nil, false, nil
		}
		return arr, true, nil
	}
	return nil, false, fmt.Errorf("stop must be a string or an array of strings")
}

func mustJSON(v any) json.RawMessage {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return data
}

func isJSONNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}
