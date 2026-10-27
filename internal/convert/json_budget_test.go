package convert

import (
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func repeatJSONNulls(count int) string {
	var b strings.Builder
	b.Grow(count * 5)
	for i := 0; i < count; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString("null")
	}
	return b.String()
}

func TestParseChatRequestRejectsWideMessagesArray(t *testing.T) {
	body := `{"model":"gemini-test","messages":[` + repeatJSONNulls(maxRequestMessages+1) + `]}`
	_, _, err := ParseChatRequest([]byte(body))
	if err == nil || !strings.Contains(err.Error(), "request messages exceed limit") {
		t.Fatalf("error = %v, want request messages count limit", err)
	}
}

func TestParseChatRequestRejectsWideContentPartsArray(t *testing.T) {
	body := `{"model":"gemini-test","messages":[{"role":"user","content":[` +
		repeatJSONNulls(maxRequestContentPartsPerMsg+1) + `]}]}`
	_, _, err := ParseChatRequest([]byte(body))
	if err == nil || !strings.Contains(err.Error(), "request content parts exceed limit") {
		t.Fatalf("error = %v, want request content parts count limit", err)
	}
}

func TestParseChatRequestRejectsWideToolCallsArray(t *testing.T) {
	body := `{"model":"gemini-test","messages":[{"role":"assistant","tool_calls":[` +
		repeatJSONNulls(maxRequestToolCallsPerMsg+1) + `]}]}`
	_, _, err := ParseChatRequest([]byte(body))
	if err == nil || !strings.Contains(err.Error(), "request tool calls exceed limit") {
		t.Fatalf("error = %v, want request tool calls count limit", err)
	}
}

func TestParseChatRequestRejectsWideToolsArray(t *testing.T) {
	body := `{"model":"gemini-test","messages":[],"tools":[` +
		repeatJSONNulls(maxRequestTools+1) + `]}`
	_, _, err := ParseChatRequest([]byte(body))
	if err == nil || !strings.Contains(err.Error(), "request tools exceed limit") {
		t.Fatalf("error = %v, want request tools count limit", err)
	}
}

func TestParseChatRequestRejectsWideStopArray(t *testing.T) {
	body := `{"model":"gemini-test","messages":[],"stop":[` +
		repeatJSONNulls(maxRequestStopSequences+1) + `]}`
	_, _, err := ParseChatRequest([]byte(body))
	if err == nil || !strings.Contains(err.Error(), "request stop sequences exceed limit") {
		t.Fatalf("error = %v, want request stop sequence count limit", err)
	}
}

func TestParseGeminiResponseJSONRejectsWideCandidatesArray(t *testing.T) {
	body := `{"candidates":[` + repeatJSONNulls(maxResponseCandidates+1) + `]}`
	_, err := ParseGeminiResponseJSON([]byte(body))
	if err == nil || !strings.Contains(err.Error(), "response candidates exceed limit") {
		t.Fatalf("error = %v, want response candidates count limit", err)
	}
}

func TestParseGeminiResponseJSONRejectsWidePartsArray(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[` +
		repeatJSONNulls(maxResponseParts+1) + `]}}]}`
	_, err := ParseGeminiResponseJSON([]byte(body))
	if err == nil || !strings.Contains(err.Error(), "response parts exceed limit") {
		t.Fatalf("error = %v, want response parts count limit", err)
	}
}

func TestBudgetRejectsUnicodeEquivalentFieldNames(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantError string
		parse     func([]byte) error
	}{
		{
			name:      "request messages",
			body:      `{"meſſages":[` + repeatJSONNulls(maxRequestMessages+1) + `]}`,
			wantError: "request messages exceed limit",
			parse: func(body []byte) error {
				_, _, err := ParseChatRequest(body)
				return err
			},
		},
		{
			name: "request tool calls",
			body: `{"messages":[{"role":"assistant","tool_callſ":[` +
				repeatJSONNulls(maxRequestToolCallsPerMsg+1) + `]}]}`,
			wantError: "request tool calls exceed limit",
			parse: func(body []byte) error {
				_, _, err := ParseChatRequest(body)
				return err
			},
		},
		{
			name:      "response candidates",
			body:      `{"candidateſ":[` + repeatJSONNulls(maxResponseCandidates+1) + `]}`,
			wantError: "response candidates exceed limit",
			parse: func(body []byte) error {
				_, err := ParseGeminiResponseJSON(body)
				return err
			},
		},
		{
			name: "response parts",
			body: `{"candidates":[{"content":{"partſ":[` +
				repeatJSONNulls(maxResponseParts+1) + `]}}]}`,
			wantError: "response parts exceed limit",
			parse: func(body []byte) error {
				_, err := ParseGeminiResponseJSON(body)
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.parse([]byte(tt.body))
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("error = %v, want %q", err, tt.wantError)
			}
		})
	}
}

func TestParseGeminiResponseJSONRejectsTooManyPartFields(t *testing.T) {
	var fields strings.Builder
	for i := 0; i < maxResponsePartFieldsPerPart+1; i++ {
		if i > 0 {
			fields.WriteByte(',')
		}
		fields.WriteString(`"unknownField":null`)
	}
	body := `{"candidates":[{"content":{"parts":[{` + fields.String() + `}]}}]}`
	_, err := ParseGeminiResponseJSON([]byte(body))
	if err == nil || !strings.Contains(err.Error(), "response part fields exceed limit") {
		t.Fatalf("error = %v, want response part field count limit", err)
	}
}

func buildTopLevelFieldsBody(prefix []string, filler int, repeated bool) []byte {
	var b strings.Builder
	b.WriteByte('{')
	first := true
	writeField := func(field string) {
		if !first {
			b.WriteByte(',')
		}
		first = false
		b.WriteString(field)
	}
	for _, field := range prefix {
		writeField(field)
	}
	for i := 0; i < filler; i++ {
		if repeated {
			writeField(`"u":null`)
			continue
		}
		writeField(`"u` + strconv.Itoa(i) + `":null`)
	}
	b.WriteByte('}')
	return []byte(b.String())
}

func TestParseChatRequestRejectsManyUnknownTopLevelFields(t *testing.T) {
	body := buildTopLevelFieldsBody([]string{`"model":"gemini-test"`, `"messages":[]`}, 100000, false)

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	_, _, err := ParseChatRequest(body)
	runtime.ReadMemStats(&after)

	if err == nil || !strings.Contains(err.Error(), "request top-level fields exceed limit") {
		t.Fatalf("error = %v, want request top-level field count limit", err)
	}
	// The budget must reject the request before decoding the top-level object
	// into a map. Without that check, 100k unknown fields allocate well over
	// 10 MiB; the structural scan itself stops at the field limit.
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 8<<20 {
		t.Fatalf("ParseChatRequest allocated %d bytes before rejecting wide top-level object; want < 8 MiB", allocated)
	}
}

func TestParseChatRequestRejectsManyRepeatedTopLevelFields(t *testing.T) {
	// Repeated keys collapse in a map, but they still consume scan time and
	// must count toward the same top-level field budget.
	body := buildTopLevelFieldsBody(
		[]string{`"model":"gemini-test"`, `"messages":[]`},
		maxRequestTopLevelFields-1,
		true,
	)
	_, _, err := ParseChatRequest(body)
	if err == nil || !strings.Contains(err.Error(), "request top-level fields exceed limit") {
		t.Fatalf("error = %v, want request top-level field count limit", err)
	}
}

func TestParseChatRequestTopLevelFieldBudgetBoundary(t *testing.T) {
	prefix := []string{
		`"model":"gemini-test"`,
		`"messages":[]`,
		`"safetySettings":[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"}]`,
	}
	body := buildTopLevelFieldsBody(prefix, maxRequestTopLevelFields-len(prefix), false)

	req, raw, err := ParseChatRequest(body)
	if err != nil {
		t.Fatalf("ParseChatRequest at field limit: %v", err)
	}
	if len(raw) != maxRequestTopLevelFields {
		t.Fatalf("raw fields = %d, want %d", len(raw), maxRequestTopLevelFields)
	}
	if _, ok := raw["safetySettings"]; !ok {
		t.Fatal("known passthrough field was dropped at the field budget boundary")
	}
	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	if len(gem.SafetySettings) == 0 || !strings.Contains(string(gem.SafetySettings), "HARM_CATEGORY_HARASSMENT") {
		t.Fatalf("safetySettings = %s, want passthrough value", gem.SafetySettings)
	}

	over := buildTopLevelFieldsBody(prefix, maxRequestTopLevelFields-len(prefix)+1, false)
	if _, _, err := ParseChatRequest(over); err == nil || !strings.Contains(err.Error(), "request top-level fields exceed limit") {
		t.Fatalf("error over boundary = %v, want request top-level field count limit", err)
	}
}

func buildGenerationConfigFieldsBody(prefix []string, filler int, repeated bool) []byte {
	var b strings.Builder
	b.WriteString(`{"model":"gemini-test","messages":[],"generationConfig":{`)
	first := true
	writeField := func(field string) {
		if !first {
			b.WriteByte(',')
		}
		first = false
		b.WriteString(field)
	}
	for _, field := range prefix {
		writeField(field)
	}
	for i := 0; i < filler; i++ {
		if repeated {
			writeField(`"u":null`)
			continue
		}
		writeField(`"u` + strconv.Itoa(i) + `":null`)
	}
	b.WriteString("}}")
	return []byte(b.String())
}

func TestParseChatRequestGenerationConfigFieldBudgetBoundary(t *testing.T) {
	prefix := []string{`"responseMimeType":"application/json"`, `"candidateCount":1`}
	body := buildGenerationConfigFieldsBody(prefix, maxRequestGenerationConfig-len(prefix), false)

	req, raw, err := ParseChatRequest(body)
	if err != nil {
		t.Fatalf("ParseChatRequest at generationConfig field limit: %v", err)
	}
	gem, err := ToGemini(req, raw)
	if err != nil {
		t.Fatalf("ToGemini: %v", err)
	}
	if got := string(gem.GenerationConfig["responseMimeType"]); got != `"application/json"` {
		t.Fatalf("responseMimeType = %s, want passthrough value", got)
	}
	if got := string(gem.GenerationConfig["candidateCount"]); got != "1" {
		t.Fatalf("candidateCount = %s, want passthrough value", got)
	}

	over := buildGenerationConfigFieldsBody(prefix, maxRequestGenerationConfig-len(prefix)+1, false)
	if _, _, err := ParseChatRequest(over); err == nil || !strings.Contains(err.Error(), "request generationConfig fields exceed limit") {
		t.Fatalf("error over boundary = %v, want request generationConfig field count limit", err)
	}
}

func TestParseChatRequestRejectsManyGenerationConfigFields(t *testing.T) {
	body := buildGenerationConfigFieldsBody(nil, 100000, false)

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	_, _, err := ParseChatRequest(body)
	runtime.ReadMemStats(&after)

	if err == nil || !strings.Contains(err.Error(), "request generationConfig fields exceed limit") {
		t.Fatalf("error = %v, want request generationConfig field count limit", err)
	}
	// The budget must reject the request before decoding generationConfig into
	// a map. Without that check, 100k unknown fields allocate well over 10 MiB;
	// the structural scan itself stops at the nested field limit.
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 8<<20 {
		t.Fatalf("ParseChatRequest allocated %d bytes before rejecting wide generationConfig; want < 8 MiB", allocated)
	}
}

func TestParseChatRequestRejectsManyRepeatedGenerationConfigFields(t *testing.T) {
	// Repeated keys collapse in a map, but they still consume scan time and
	// must count toward the same nested field budget.
	body := buildGenerationConfigFieldsBody(nil, maxRequestGenerationConfig+1, true)
	_, _, err := ParseChatRequest(body)
	if err == nil || !strings.Contains(err.Error(), "request generationConfig fields exceed limit") {
		t.Fatalf("error = %v, want request generationConfig field count limit", err)
	}
}
