package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func strPtr(s string) *string { return &s }

func TestGeminiPartsConcatenateWithoutSeparator(t *testing.T) {
	resp := &GeminiResponse{
		Candidates: []GeminiCandidate{{
			Content: GeminiContent{Parts: []GeminiPart{
				{Text: strPtr("one")},
				{Text: strPtr("two")},
				{Text: strPtr("three")},
			}},
			FinishReason: "STOP",
		}},
		UsageMetadata: &GeminiUsage{
			PromptTokenCount:     3,
			CandidatesTokenCount: 4,
			TotalTokenCount:      7,
		},
	}

	out, err := GeminiResponseToOpenAI(resp, "gemini-test")
	if err != nil {
		t.Fatalf("GeminiResponseToOpenAI: %v", err)
	}
	if len(out.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(out.Choices))
	}
	content := out.Choices[0].Message.Content
	if content == nil || *content != "onetwothree" {
		t.Fatalf("content = %v, want onetwothree", content)
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %q", out.Choices[0].FinishReason)
	}
	if out.Usage == nil || out.Usage.PromptTokens != 3 || out.Usage.CompletionTokens != 4 || out.Usage.TotalTokens != 7 {
		t.Fatalf("usage = %#v", out.Usage)
	}
}

func TestMapFinishReason(t *testing.T) {
	cases := map[string]string{
		"STOP":                      "stop",
		"MAX_TOKENS":                "length",
		"SAFETY":                    "content_filter",
		"RECITATION":                "content_filter",
		"BLOCKLIST":                 "content_filter",
		"PROHIBITED_CONTENT":        "content_filter",
		"SPII":                      "content_filter",
		"IMAGE_SAFETY":              "content_filter",
		"OTHER":                     "",
		"":                          "stop",
		"FINISH_REASON_UNSPECIFIED": "stop",
		"MALFORMED_FUNCTION_CALL":   "",
		"UNEXPECTED_TOOL_CALL":      "",
	}
	for in, want := range cases {
		if got := MapFinishReason(in); got != want {
			t.Errorf("MapFinishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestFunctionCallToToolCalls(t *testing.T) {
	resp := &GeminiResponse{
		Candidates: []GeminiCandidate{{
			Content: GeminiContent{Parts: []GeminiPart{
				{FunctionCall: &GeminiFunctionCall{
					Name: "get_weather",
					Args: json.RawMessage(`{"city":"Paris"}`),
					ID:   "call_abc",
				}},
			}},
			FinishReason: "STOP",
		}},
	}

	out, err := GeminiResponseToOpenAI(resp, "gemini-test")
	if err != nil {
		t.Fatalf("GeminiResponseToOpenAI: %v", err)
	}
	msg := out.Choices[0].Message
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool_calls = %#v", msg.ToolCalls)
	}
	if msg.ToolCalls[0].ID != "call_abc" || msg.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("tool_call = %#v", msg.ToolCalls[0])
	}
	if msg.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Fatalf("arguments = %q", msg.ToolCalls[0].Function.Arguments)
	}
	if msg.Content != nil {
		t.Fatalf("content = %v, want nil when only tool calls", *msg.Content)
	}
	if out.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason = %q, want tool_calls", out.Choices[0].FinishReason)
	}
}

func TestChunkBuilderContentWithoutSeparator(t *testing.T) {
	builder := NewChunkBuilder("gemini-test")
	c1 := builder.ContentChunk("a")
	c2 := builder.ContentChunk("b")
	if c1.Choices[0].Delta.Content == nil || *c1.Choices[0].Delta.Content != "a" {
		t.Fatalf("chunk 1 = %#v", c1)
	}
	if c2.Choices[0].Delta.Content == nil || *c2.Choices[0].Delta.Content != "b" {
		t.Fatalf("chunk 2 = %#v", c2)
	}
}

func TestChunkBuilderToolCallFinishReason(t *testing.T) {
	builder := NewChunkBuilder("gemini-test")
	builder.ToolCallChunk(&GeminiFunctionCall{Name: "get_weather", Args: json.RawMessage(`{}`)})
	finish, err := builder.FinishChunk("STOP", nil)
	if err != nil {
		t.Fatalf("FinishChunk: %v", err)
	}
	if finish.Choices[0].FinishReason == nil || *finish.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason = %v, want tool_calls", finish.Choices[0].FinishReason)
	}
}

func TestUsageIncludesThoughtAndCachedTokens(t *testing.T) {
	out := UsageFromGemini(&GeminiUsage{
		PromptTokenCount:        10,
		CandidatesTokenCount:    5,
		TotalTokenCount:         35,
		ThoughtsTokenCount:      20,
		CachedContentTokenCount: 4,
	})
	if out == nil {
		t.Fatal("usage is nil")
	}
	if out.PromptTokens != 10 || out.CompletionTokens != 25 || out.TotalTokens != 35 {
		t.Fatalf("usage = %#v", out)
	}
	if out.CompletionTokensDetails == nil || out.CompletionTokensDetails.ReasoningTokens != 20 {
		t.Fatalf("completion details = %#v", out.CompletionTokensDetails)
	}
	if out.PromptTokensDetails == nil || out.PromptTokensDetails.CachedTokens != 4 {
		t.Fatalf("prompt details = %#v", out.PromptTokensDetails)
	}
}

func TestPromptFeedbackBlockMapsToContentFilter(t *testing.T) {
	for _, reason := range []string{"SAFETY", "OTHER", "IMAGE_PROHIBITED_CONTENT"} {
		out, err := GeminiResponseToOpenAI(&GeminiResponse{
			PromptFeedback: &GeminiPromptFeedback{BlockReason: reason},
		}, "gemini-test")
		if err != nil {
			t.Fatalf("GeminiResponseToOpenAI: %v", err)
		}
		if out.Choices[0].FinishReason != "content_filter" {
			t.Fatalf("blockReason %q: finish_reason = %q, want content_filter", reason, out.Choices[0].FinishReason)
		}
		if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != "" {
			t.Fatalf("blockReason %q: content = %v, want empty string", reason, out.Choices[0].Message.Content)
		}
	}
}

func TestAbnormalFinishReasonTakesPrecedenceOverToolCalls(t *testing.T) {
	for _, tc := range []struct {
		reason string
		want   string
	}{
		{"MAX_TOKENS", "length"},
		{"SAFETY", "content_filter"},
	} {
		resp := &GeminiResponse{Candidates: []GeminiCandidate{{
			Content: GeminiContent{Parts: []GeminiPart{
				{FunctionCall: &GeminiFunctionCall{Name: "get_weather", Args: json.RawMessage(`{}`)}},
			}},
			FinishReason: tc.reason,
		}}}
		out, err := GeminiResponseToOpenAI(resp, "gemini-test")
		if err != nil {
			t.Fatalf("GeminiResponseToOpenAI: %v", err)
		}
		if out.Choices[0].FinishReason != tc.want {
			t.Fatalf("finishReason %q = %q, want %q", tc.reason, out.Choices[0].FinishReason, tc.want)
		}
	}
}

func TestThoughtSignatureRoundTrip(t *testing.T) {
	resp := &GeminiResponse{Candidates: []GeminiCandidate{{
		Content: GeminiContent{Parts: []GeminiPart{{
			FunctionCall:     &GeminiFunctionCall{Name: "get_weather", Args: json.RawMessage(`{"city":"Paris"}`), ID: "call_1"},
			ThoughtSignature: "sig-1",
		}}},
		FinishReason: "STOP",
	}}}
	out, err := GeminiResponseToOpenAI(resp, "gemini-test")
	if err != nil {
		t.Fatalf("GeminiResponseToOpenAI: %v", err)
	}
	if len(out.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("tool_calls = %#v", out.Choices[0].Message.ToolCalls)
	}
	if got := out.Choices[0].Message.ToolCalls[0].ThoughtSignature; got != "sig-1" {
		t.Fatalf("thought_signature = %q, want sig-1", got)
	}

	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [
			{"role": "user", "content": "weather?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "thought_signature": "sig-1",
				 "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
		]
	}`)
	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	var got string
	for _, content := range gem.Contents {
		for _, part := range content.Parts {
			if part.FunctionCall != nil {
				got = part.ThoughtSignature
			}
		}
	}
	if got != "sig-1" {
		t.Fatalf("round-tripped thoughtSignature = %q, want sig-1", got)
	}
}

func TestChunkBuilderToolCallSignatureAndLength(t *testing.T) {
	builder := NewChunkBuilder("gemini-test")
	chunk := builder.ToolCallChunk(&GeminiFunctionCall{Name: "get_weather", Args: json.RawMessage(`{}`)}, "sig-2")
	if got := chunk.Choices[0].Delta.ToolCalls[0].ThoughtSignature; got != "sig-2" {
		t.Fatalf("thought_signature = %q, want sig-2", got)
	}
	finish, err := builder.FinishChunk("MAX_TOKENS", nil)
	if err != nil {
		t.Fatalf("FinishChunk: %v", err)
	}
	if finish.Choices[0].FinishReason == nil || *finish.Choices[0].FinishReason != "length" {
		t.Fatalf("finish_reason = %v, want length", finish.Choices[0].FinishReason)
	}
}

func TestWriteOpenAIResponseJSONAbnormalFinishWithToolCalls(t *testing.T) {
	resp := &GeminiResponse{Candidates: []GeminiCandidate{{
		Content:      GeminiContent{Parts: []GeminiPart{{FunctionCall: &GeminiFunctionCall{Name: "f", Args: json.RawMessage(`{}`)}}}},
		FinishReason: "MAX_TOKENS",
	}}}
	var buf bytes.Buffer
	if err := WriteOpenAIResponseJSON(&buf, resp, "gemini-test", nil); err != nil {
		t.Fatalf("WriteOpenAIResponseJSON: %v", err)
	}
	var out struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(buf.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].FinishReason != "length" {
		t.Fatalf("finish_reason = %q, want length", out.Choices[0].FinishReason)
	}
}

func TestWriteOpenAIResponseJSONStreamsContent(t *testing.T) {
	resp := &GeminiResponse{Candidates: []GeminiCandidate{{
		Content:      GeminiContent{Parts: []GeminiPart{{FunctionCall: &GeminiFunctionCall{Name: "f", Args: json.RawMessage(`{}`)}}}},
		FinishReason: "STOP",
	}}}
	content := strings.Repeat("你好abc", 10000)

	var buf bytes.Buffer
	if err := WriteOpenAIResponseJSON(&buf, resp, "gemini-test", strings.NewReader(content)); err != nil {
		t.Fatalf("WriteOpenAIResponseJSON: %v", err)
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content   *string           `json:"content"`
				ToolCalls []json.RawMessage `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(buf.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != content {
		t.Fatal("streamed content did not round-trip")
	}
	if len(out.Choices[0].Message.ToolCalls) != 1 || out.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("tool calls = %#v, finish_reason = %q", out.Choices[0].Message.ToolCalls, out.Choices[0].FinishReason)
	}
}

func TestGeminiImageOnlyResponseIsPreserved(t *testing.T) {
	resp := &GeminiResponse{Candidates: []GeminiCandidate{{
		Content: GeminiContent{Parts: []GeminiPart{{
			InlineData: &GeminiInlineData{MIMEType: "image/png", Data: "AAAA"},
		}}},
		FinishReason: "STOP",
	}}}

	out, err := GeminiResponseToOpenAI(resp, "gemini-test")
	if err != nil {
		t.Fatalf("GeminiResponseToOpenAI: %v", err)
	}
	msg := out.Choices[0].Message
	if msg.Content != nil {
		t.Fatalf("content = %v, want nil for image-only response", *msg.Content)
	}
	if len(msg.ContentParts) != 1 {
		t.Fatalf("content parts = %#v", msg.ContentParts)
	}
	if msg.ContentParts[0].Type != "image_url" || msg.ContentParts[0].ImageURL == nil {
		t.Fatalf("image part = %#v", msg.ContentParts[0])
	}
	if got := msg.ContentParts[0].ImageURL.URL; got != "data:image/png;base64,AAAA" {
		t.Fatalf("image URL = %q", got)
	}

	data, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	var parts []OpenAIContentPart
	if err := json.Unmarshal(parsed.Choices[0].Message.Content, &parts); err != nil {
		t.Fatalf("content is not an array: %v (%s)", err, parsed.Choices[0].Message.Content)
	}
	if len(parts) != 1 || parts[0].ImageURL == nil || parts[0].ImageURL.URL != "data:image/png;base64,AAAA" {
		t.Fatalf("serialized content = %#v", parts)
	}
}

func TestGeminiMixedTextAndImagePreservesOrder(t *testing.T) {
	resp := &GeminiResponse{Candidates: []GeminiCandidate{{
		Content: GeminiContent{Parts: []GeminiPart{
			{Text: strPtr("alpha")},
			{InlineData: &GeminiInlineData{MIMEType: "image/jpeg", Data: "BBBB"}},
			{Text: strPtr("beta")},
		}},
		FinishReason: "STOP",
	}}}

	out, err := GeminiResponseToOpenAI(resp, "gemini-test")
	if err != nil {
		t.Fatalf("GeminiResponseToOpenAI: %v", err)
	}
	parts := out.Choices[0].Message.ContentParts
	if len(parts) != 3 {
		t.Fatalf("content parts = %#v", parts)
	}
	if parts[0].Type != "text" || parts[0].Text != "alpha" {
		t.Fatalf("part 0 = %#v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil || parts[1].ImageURL.URL != "data:image/jpeg;base64,BBBB" {
		t.Fatalf("part 1 = %#v", parts[1])
	}
	if parts[2].Type != "text" || parts[2].Text != "beta" {
		t.Fatalf("part 2 = %#v", parts[2])
	}
}

func TestGeminiImageMIMETypeLengthIsBounded(t *testing.T) {
	tooLong := "image/" + strings.Repeat("a", MaxImageMIMETypeBytes)
	resp := &GeminiResponse{Candidates: []GeminiCandidate{{
		Content: GeminiContent{Parts: []GeminiPart{{
			InlineData: &GeminiInlineData{MIMEType: tooLong, Data: "AAAA"},
		}}},
		FinishReason: "STOP",
	}}}
	if _, err := GeminiResponseToOpenAI(resp, "gemini-test"); err == nil {
		t.Fatal("expected oversized image mimeType to be rejected")
	}
}

func TestGeminiUnsupportedNonTextResponseIsRejected(t *testing.T) {
	resp := &GeminiResponse{Candidates: []GeminiCandidate{{
		Content: GeminiContent{Parts: []GeminiPart{{
			InlineData: &GeminiInlineData{MIMEType: "audio/wav", Data: "AAAA"},
		}}},
		FinishReason: "STOP",
	}}}
	if _, err := GeminiResponseToOpenAI(resp, "gemini-test"); err == nil {
		t.Fatal("expected unsupported non-text response to be rejected")
	}
}

func TestAbnormalFinishReasonsAreErrors(t *testing.T) {
	for _, reason := range []string{
		"MALFORMED_FUNCTION_CALL",
		"UNEXPECTED_TOOL_CALL",
		"TOO_MANY_TOOL_CALLS",
		"OTHER",
		"LANGUAGE",
	} {
		for _, withToolCall := range []bool{false, true} {
			parts := []GeminiPart{{Text: strPtr("partial")}}
			if withToolCall {
				parts = []GeminiPart{{FunctionCall: &GeminiFunctionCall{Name: "f", Args: json.RawMessage(`{}`)}}}
			}
			resp := &GeminiResponse{Candidates: []GeminiCandidate{{
				Content:      GeminiContent{Parts: parts},
				FinishReason: reason,
			}}}
			if _, err := GeminiResponseToOpenAI(resp, "gemini-test"); err == nil {
				t.Fatalf("finishReason %q (tool call %v) was not rejected", reason, withToolCall)
			}
		}
	}
}

func TestChunkBuilderAbnormalFinishReturnsError(t *testing.T) {
	builder := NewChunkBuilder("gemini-test")
	builder.ToolCallChunk(&GeminiFunctionCall{Name: "f", Args: json.RawMessage(`{}`)})
	if _, err := builder.FinishChunk("MALFORMED_FUNCTION_CALL", nil); err == nil {
		t.Fatal("expected abnormal finish reason to return an error")
	}
}

func TestUnknownGeminiPartIsRejected(t *testing.T) {
	var resp GeminiResponse
	if err := json.Unmarshal([]byte(`{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}}]},"finishReason":"STOP"}]}`), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, err := GeminiResponseToOpenAI(&resp, "gemini-test"); err == nil {
		t.Fatal("expected unknown Gemini part to be rejected")
	} else if !strings.Contains(err.Error(), "executableCode") {
		t.Fatalf("error = %v, want it to name executableCode", err)
	}
}

func TestValidateGeminiResponseRequiresCandidateOrBlockReason(t *testing.T) {
	for _, body := range []string{`{}`, `null`, `{"candidates":[]}`, `{"candidates":[{}]}`} {
		var resp GeminiResponse
		if err := json.Unmarshal([]byte(body), &resp); err != nil {
			t.Fatalf("unmarshal %s: %v", body, err)
		}
		if err := ValidateGeminiResponse(&resp); err == nil {
			t.Fatalf("ValidateGeminiResponse(%s) unexpectedly succeeded", body)
		}
	}
	blocked := &GeminiResponse{PromptFeedback: &GeminiPromptFeedback{BlockReason: "SAFETY"}}
	if err := ValidateGeminiResponse(blocked); err != nil {
		t.Fatalf("prompt block should be valid: %v", err)
	}
}
