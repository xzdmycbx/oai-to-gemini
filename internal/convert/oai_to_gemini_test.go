package convert

import (
	"encoding/json"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func parseForTest(t *testing.T, body string) (*ChatRequest, map[string]json.RawMessage) {
	t.Helper()
	req, raw, err := ParseChatRequest([]byte(body))
	if err != nil {
		t.Fatalf("ParseChatRequest: %v", err)
	}
	return req, raw
}

func TestOpenAIContentPartsConcatenateWithoutSeparator(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "alpha"},
				{"type": "text", "text": "beta"},
				{"type": "text", "text": "gamma"}
			]}
		]
	}`)

	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	if len(gem.Contents) != 1 {
		t.Fatalf("contents = %d, want 1", len(gem.Contents))
	}
	parts := gem.Contents[0].Parts
	if len(parts) != 1 {
		t.Fatalf("parts = %d, want 1", len(parts))
	}
	if parts[0].Text == nil || *parts[0].Text != "alphabetagamma" {
		t.Fatalf("text = %v, want alphabetagamma (no separators)", parts[0].Text)
	}
}

func TestSystemUserAssistantConversion(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [
			{"role": "system", "content": "rule one"},
			{"role": "system", "content": "rule two"},
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "hi"}
		]
	}`)

	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	if gem.SystemInstruction == nil || len(gem.SystemInstruction.Parts) != 1 {
		t.Fatalf("systemInstruction = %#v", gem.SystemInstruction)
	}
	if got := *gem.SystemInstruction.Parts[0].Text; got != "rule onerule two" {
		t.Fatalf("system text = %q", got)
	}
	if len(gem.Contents) != 2 {
		t.Fatalf("contents = %d, want 2", len(gem.Contents))
	}
	if gem.Contents[0].Role != "user" || *gem.Contents[0].Parts[0].Text != "hello" {
		t.Fatalf("user content = %#v", gem.Contents[0])
	}
	if gem.Contents[1].Role != "model" || *gem.Contents[1].Parts[0].Text != "hi" {
		t.Fatalf("assistant content = %#v", gem.Contents[1])
	}
}

func TestGenerationConfigMapping(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [{"role": "user", "content": "x"}],
		"temperature": 0.7,
		"top_p": 0.9,
		"max_tokens": 128,
		"stop": ["END", "STOP"],
		"topK": 40,
		"generationConfig": {"candidateCount": 1}
	}`)

	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	gc := gem.GenerationConfig
	if gc == nil {
		t.Fatal("generationConfig is nil")
	}

	assertRawNumber(t, gc, "temperature", 0.7)
	assertRawNumber(t, gc, "topP", 0.9)
	assertRawNumber(t, gc, "maxOutputTokens", 128)
	assertRawNumber(t, gc, "topK", 40)
	assertRawNumber(t, gc, "candidateCount", 1)

	var stops []string
	if err := json.Unmarshal(gc["stopSequences"], &stops); err != nil {
		t.Fatalf("stopSequences: %v", err)
	}
	if len(stops) != 2 || stops[0] != "END" || stops[1] != "STOP" {
		t.Fatalf("stopSequences = %#v", stops)
	}
}

func TestCandidateCountGreaterThanOneIsRejected(t *testing.T) {
	for _, body := range []string{
		`{"model":"gemini-test","messages":[{"role":"user","content":"x"}],"candidateCount":2}`,
		`{"model":"gemini-test","messages":[{"role":"user","content":"x"}],"generationConfig":{"candidateCount":2}}`,
	} {
		req, raw := parseForTest(t, body)
		if _, err := ToGemini(req, raw); err == nil {
			t.Fatalf("expected candidateCount > 1 to be rejected: %s", body)
		}
	}
}

func TestOpenAINGreaterThanOneIsRejected(t *testing.T) {
	req, raw := parseForTest(t, `{"model":"gemini-test","messages":[{"role":"user","content":"x"}],"n":2}`)
	if _, err := ToGemini(req, raw); err == nil {
		t.Fatal("expected OpenAI n > 1 to be rejected")
	}

	req, raw = parseForTest(t, `{"model":"gemini-test","messages":[{"role":"user","content":"x"}],"n":1}`)
	if _, err := ToGemini(req, raw); err != nil {
		t.Fatalf("n = 1 should be accepted: %v", err)
	}
}

func TestManyAdjacentTextPartsConcatenateWithoutSeparator(t *testing.T) {
	const count = 100000
	var body strings.Builder
	body.WriteString(`{"model":"gemini-test","messages":[{"role":"user","content":[`)
	for i := 0; i < count; i++ {
		if i > 0 {
			body.WriteByte(',')
		}
		body.WriteString(`{"type":"text","text":"a"}`)
	}
	body.WriteString(`]}]}`)

	req, raw := parseForTest(t, body.String())
	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	parts := gem.Contents[0].Parts
	if len(parts) != 1 || parts[0].Text == nil {
		t.Fatalf("parts = %#v, want one merged text part", parts)
	}
	if got := len(*parts[0].Text); got != count {
		t.Fatalf("merged text length = %d, want %d", got, count)
	}
}

func TestMultimodalContentParts(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [{"role": "user", "content": [
			{"type": "text", "text": "alpha"},
			{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
			{"type": "text", "text": "beta"},
			{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
			{"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0="}}
		]}]
	}`)
	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	if len(gem.Contents) != 1 {
		t.Fatalf("contents = %d, want 1", len(gem.Contents))
	}
	parts := gem.Contents[0].Parts
	if len(parts) != 5 {
		t.Fatalf("parts = %d, want 5: %#v", len(parts), parts)
	}
	if parts[0].Text == nil || *parts[0].Text != "alpha" {
		t.Fatalf("text part 0 = %#v", parts[0])
	}
	if parts[1].InlineData == nil || parts[1].InlineData.MIMEType != "image/png" || parts[1].InlineData.Data != "iVBORw0KGgo=" {
		t.Fatalf("image part = %#v", parts[1])
	}
	if parts[2].Text == nil || *parts[2].Text != "beta" {
		t.Fatalf("text part 2 = %#v", parts[2])
	}
	if parts[3].InlineData == nil || parts[3].InlineData.MIMEType != "audio/wav" || parts[3].InlineData.Data != "AAAA" {
		t.Fatalf("audio part = %#v", parts[3])
	}
	if parts[4].InlineData == nil || parts[4].InlineData.MIMEType != "application/pdf" {
		t.Fatalf("file part = %#v", parts[4])
	}
}

func TestUnsupportedContentPartIsRejected(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [{"role": "user", "content": [
			{"type": "text", "text": "hello"},
			{"type": "video_url", "video_url": {"url": "https://example.invalid/video.mp4"}}
		]}]
	}`)
	if _, err := ToGemini(req, raw); err == nil {
		t.Fatal("expected unsupported content part to be rejected")
	}
}

func TestRemoteImageUsesFileData(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [{"role": "user", "content": [
			{"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}
		]}]
	}`)
	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	part := gem.Contents[0].Parts[0]
	if part.FileData == nil || part.FileData.FileURI != "https://example.invalid/image.png" {
		t.Fatalf("fileData = %#v", part.FileData)
	}
}

func TestGenerationConfigOmitsDefaults(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [{"role": "user", "content": "x"}]
	}`)

	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	if gem.GenerationConfig != nil {
		t.Fatalf("generationConfig = %#v, want nil", gem.GenerationConfig)
	}
}

func TestToolsAndToolCallsConversion(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [
			{"role": "user", "content": "weather?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
		],
		"tools": [{"type": "function", "function": {
			"name": "get_weather",
			"description": "Get weather",
			"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
		}}],
		"tool_choice": "required"
	}`)

	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	if len(gem.Tools) != 1 || len(gem.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("tools = %#v", gem.Tools)
	}
	if gem.Tools[0].FunctionDeclarations[0].Name != "get_weather" {
		t.Fatalf("tool name = %q", gem.Tools[0].FunctionDeclarations[0].Name)
	}
	if gem.ToolConfig == nil || gem.ToolConfig.FunctionCallingConfig.Mode != "ANY" {
		t.Fatalf("toolConfig = %#v", gem.ToolConfig)
	}

	var foundCall, foundResponse bool
	for _, content := range gem.Contents {
		for _, part := range content.Parts {
			if part.FunctionCall != nil {
				foundCall = true
				if part.FunctionCall.Name != "get_weather" || string(part.FunctionCall.Args) != `{"city":"Paris"}` {
					t.Fatalf("functionCall = %#v", part.FunctionCall)
				}
			}
			if part.FunctionResponse != nil {
				foundResponse = true
				if part.FunctionResponse.Name != "get_weather" {
					t.Fatalf("functionResponse name = %q", part.FunctionResponse.Name)
				}
				var response map[string]string
				if err := json.Unmarshal(part.FunctionResponse.Response, &response); err != nil {
					t.Fatalf("functionResponse response: %v", err)
				}
				if response["result"] != "sunny" {
					t.Fatalf("functionResponse = %#v", response)
				}
			}
		}
	}
	if !foundCall || !foundResponse {
		t.Fatalf("foundCall=%v foundResponse=%v", foundCall, foundResponse)
	}
}

func TestParallelToolResultsShareOneUserTurn(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [
			{"role": "user", "content": "weather and time?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
				{"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": "{\"city\":\"Paris\"}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
			{"role": "tool", "tool_call_id": "call_2", "content": "noon"}
		]
	}`)

	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	if len(gem.Contents) != 3 {
		t.Fatalf("contents = %d, want 3 (user, model, one merged tool turn)", len(gem.Contents))
	}
	toolTurn := gem.Contents[2]
	if toolTurn.Role != "user" || len(toolTurn.Parts) != 2 {
		t.Fatalf("tool turn = %#v, want one user turn with 2 functionResponse parts", toolTurn)
	}
	for i, want := range []struct{ id, name, result string }{
		{"call_1", "get_weather", "sunny"},
		{"call_2", "get_time", "noon"},
	} {
		fr := toolTurn.Parts[i].FunctionResponse
		if fr == nil || fr.ID != want.id || fr.Name != want.name {
			t.Fatalf("part %d = %#v", i, toolTurn.Parts[i])
		}
		var response map[string]string
		if err := json.Unmarshal(fr.Response, &response); err != nil {
			t.Fatalf("part %d response: %v", i, err)
		}
		if response["result"] != want.result {
			t.Fatalf("part %d result = %q, want %q", i, response["result"], want.result)
		}
	}
}

func TestSeparateToolRoundsUseSeparateUserTurns(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [
			{"role": "user", "content": "first"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "f1", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "one"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_2", "type": "function", "function": {"name": "f2", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_2", "content": "two"}
		]
	}`)

	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	if len(gem.Contents) != 5 {
		t.Fatalf("contents = %d, want 5", len(gem.Contents))
	}
	for _, idx := range []int{2, 4} {
		turn := gem.Contents[idx]
		if turn.Role != "user" || len(turn.Parts) != 1 || turn.Parts[0].FunctionResponse == nil {
			t.Fatalf("contents[%d] = %#v, want a single functionResponse", idx, turn)
		}
	}
	if got := gem.Contents[2].Parts[0].FunctionResponse.ID; got != "call_1" {
		t.Fatalf("first tool turn id = %q", got)
	}
	if got := gem.Contents[4].Parts[0].FunctionResponse.ID; got != "call_2" {
		t.Fatalf("second tool turn id = %q", got)
	}
}

func assertRawNumber(t *testing.T, m map[string]json.RawMessage, key string, want float64) {
	t.Helper()
	raw, ok := m[key]
	if !ok {
		t.Fatalf("missing %s", key)
	}
	var got float64
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("%s: %v", key, err)
	}
	if got != want {
		t.Fatalf("%s = %v, want %v", key, got, want)
	}
}

func TestToolResultTextPartsArrayIsConcatenated(t *testing.T) {
	for _, content := range []string{
		`"alphabeta"`,
		`[{"type":"text","text":"alpha"},{"type":"text","text":"beta"}]`,
		`[{"type":"input_text","text":"alpha"},{"type":"text","text":"beta"}]`,
	} {
		body := `{
			"model": "gemini-test",
			"messages": [
				{"role": "user", "content": "weather?"},
				{"role": "assistant", "content": null, "tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
				]},
				{"role": "tool", "tool_call_id": "call_1", "content": ` + content + `}
			]
		}`
		req, raw := parseForTest(t, body)
		gem, err := ToGemini(req, raw)
		if err != nil {
			t.Fatalf("ToGemini(%s): %v", content, err)
		}
		var found bool
		for _, gemContent := range gem.Contents {
			for _, part := range gemContent.Parts {
				if part.FunctionResponse == nil {
					continue
				}
				found = true
				var response map[string]string
				if err := json.Unmarshal(part.FunctionResponse.Response, &response); err != nil {
					t.Fatalf("functionResponse: %v", err)
				}
				if response["result"] != "alphabeta" {
					t.Fatalf("content %s -> result %q, want alphabeta", content, response["result"])
				}
			}
		}
		if !found {
			t.Fatalf("content %s: no functionResponse found", content)
		}
	}
}

func TestToolResultUnsupportedContentPartIsRejected(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [
			{"role": "user", "content": "weather?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": [
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
			]}
		]
	}`)
	if _, err := ToGemini(req, raw); err == nil {
		t.Fatal("expected unsupported tool result content part to be rejected")
	}
}

func TestToolResultJSONArrayIsWrappedAsBusinessResult(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [
			{"role": "user", "content": "weather?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": [1, 2, 3]}
		]
	}`)
	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	for _, content := range gem.Contents {
		for _, part := range content.Parts {
			if part.FunctionResponse == nil {
				continue
			}
			var response struct {
				Result []int `json:"result"`
			}
			if err := json.Unmarshal(part.FunctionResponse.Response, &response); err != nil {
				t.Fatalf("functionResponse: %v", err)
			}
			if len(response.Result) != 3 || response.Result[0] != 1 || response.Result[2] != 3 {
				t.Fatalf("functionResponse = %s", part.FunctionResponse.Response)
			}
			return
		}
	}
	t.Fatal("no functionResponse found")
}

func findToolFunctionResponse(t *testing.T, gem *GeminiRequest) *GeminiFunctionResponse {
	t.Helper()
	for _, content := range gem.Contents {
		for _, part := range content.Parts {
			if part.FunctionResponse != nil {
				return part.FunctionResponse
			}
		}
	}
	t.Fatal("no functionResponse found")
	return nil
}

func TestLooksLikeContentPartArrayUsesTopLevelTypeOnly(t *testing.T) {
	tests := []struct {
		name  string
		items []json.RawMessage
		want  bool
	}{
		{"empty", nil, false},
		{"null", []json.RawMessage{json.RawMessage(`null`)}, false},
		{"non-object", []json.RawMessage{json.RawMessage(`["type"]`)}, false},
		{"no-type", []json.RawMessage{json.RawMessage(`{"text":"x"}`)}, false},
		{"nested-type", []json.RawMessage{json.RawMessage(`{"data":{"type":"text"},"text":"x"}`)}, false},
		{"type", []json.RawMessage{json.RawMessage(`{"type":"text"}`)}, true},
		{"escaped-type", []json.RawMessage{json.RawMessage(`{"\u0074ype":"text"}`)}, true},
		{"all-have-type", []json.RawMessage{
			json.RawMessage(`{"type":"text"}`),
			json.RawMessage(`{"text":"x","type":"input_text"}`),
		}, true},
		{"mixed", []json.RawMessage{
			json.RawMessage(`{"type":"text"}`),
			json.RawMessage(`{"text":"x"}`),
		}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := looksLikeContentPartArray(tc.items); got != tc.want {
				t.Fatalf("looksLikeContentPartArray = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestLooksLikeContentPartArrayWideObjectAllocatesBounded(t *testing.T) {
	var b strings.Builder
	b.WriteByte('{')
	for i := 0; i < 100000; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`"u`)
		b.WriteString(strconv.Itoa(i))
		b.WriteString(`":null`)
	}
	b.WriteByte('}')
	items := []json.RawMessage{json.RawMessage(b.String())}

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	if looksLikeContentPartArray(items) {
		t.Fatal("wide business object was mistaken for a content part array")
	}
	runtime.ReadMemStats(&after)
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 256<<10 {
		t.Fatalf("type recognition allocated %d bytes for a wide object; want < 256 KiB", allocated)
	}
}

func TestToolResultWideObjectTypeCheckAllocatesBounded(t *testing.T) {
	for _, tc := range []struct {
		name     string
		repeated bool
	}{
		{"distinct-fields", false},
		{"repeated-fields", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const fields = 100000
			var b strings.Builder
			b.WriteString(`{"model":"gemini-test","messages":[{"role":"user","content":"weather?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":[{`)
			for i := 0; i < fields; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				if tc.repeated {
					b.WriteString(`"u":null`)
					continue
				}
				b.WriteString(`"u`)
				b.WriteString(strconv.Itoa(i))
				b.WriteString(`":null`)
			}
			b.WriteString(`}]}]}`)

			req, raw := parseForTest(t, b.String())
			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			gem, err := ToGemini(req, raw)
			runtime.ReadMemStats(&after)
			if err != nil {
				t.Fatalf("ToGemini: %v", err)
			}
			// Type recognition must not decode the wide business object into a
			// map. The remaining allocations are the bounded wrapper copy.
			if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 12<<20 {
				t.Fatalf("ToGemini allocated %d bytes for a wide tool result; want < 12 MiB", allocated)
			}

			fr := findToolFunctionResponse(t, gem)
			var wrapped struct {
				Result json.RawMessage `json:"result"`
			}
			if err := json.Unmarshal(fr.Response, &wrapped); err != nil {
				t.Fatalf("functionResponse: %v", err)
			}
			if len(wrapped.Result) == 0 || wrapped.Result[0] != '[' {
				t.Fatalf("wide object was not wrapped as a business result: %s", fr.Response[:min(len(fr.Response), 80)])
			}
		})
	}
}

func TestToolResultObjectArrayWithoutTypeIsWrappedAsBusinessResult(t *testing.T) {
	req, raw := parseForTest(t, `{
		"model": "gemini-test",
		"messages": [
			{"role": "user", "content": "weather?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": [
				{"foo": "bar"},
				{"data": {"type": "text"}, "text": "nested type must not count"}
			]}
		]
	}`)
	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	fr := findToolFunctionResponse(t, gem)
	var wrapped struct {
		Result []map[string]json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(fr.Response, &wrapped); err != nil {
		t.Fatalf("functionResponse: %v", err)
	}
	if len(wrapped.Result) != 2 {
		t.Fatalf("wrapped result length = %d, want 2", len(wrapped.Result))
	}
	if got := string(wrapped.Result[0]["foo"]); got != `"bar"` {
		t.Fatalf("first result foo = %s, want \"bar\"", got)
	}
	if _, ok := wrapped.Result[1]["data"]; !ok {
		t.Fatalf("second result = %#v, want nested data preserved", wrapped.Result[1])
	}
}
