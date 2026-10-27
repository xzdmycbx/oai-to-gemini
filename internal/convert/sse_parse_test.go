package convert

import (
	"bytes"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"
)

type testPayloadSink struct {
	bytes.Buffer
}

func (s *testPayloadSink) Size() int64 { return int64(s.Len()) }

func TestParseGeminiSSEPayloadStoresLargeFieldsIncrementally(t *testing.T) {
	event := `{"candidates":[{"content":{"parts":[` +
		`{"text":"\u4f60\u597d\ud83d\ude00\n\"\\"},` +
		`{"inlineData":{"mimeType":"image/png","data":"QUJD"}},` +
		`{"executableCode":{"language":"PYTHON"}}` +
		`]},"finishReason":"STOP"}],` +
		`"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}`

	var payload testPayloadSink
	resp, err := ParseGeminiSSEPayload(bytes.NewReader([]byte(event)), &payload)
	if err != nil {
		t.Fatalf("ParseGeminiSSEPayload: %v", err)
	}
	if len(resp.Candidates) != 1 || len(resp.Candidates[0].Content.Parts) != 3 {
		t.Fatalf("unexpected response shape: %#v", resp)
	}

	parts := resp.Candidates[0].Content.Parts
	if parts[0].TextRange == nil {
		t.Fatal("text was not stored as a range")
	}
	text := payload.Bytes()[parts[0].TextRange.Offset : parts[0].TextRange.Offset+parts[0].TextRange.Length]
	if string(text) != "你好😀\n\"\\" {
		t.Fatalf("decoded text = %q", text)
	}

	if parts[1].InlineData == nil || parts[1].InlineData.MIMEType != "image/png" {
		t.Fatalf("inline metadata = %#v", parts[1].InlineData)
	}
	if parts[1].InlineDataRange == nil {
		t.Fatal("inline data was not stored as a range")
	}
	data := payload.Bytes()[parts[1].InlineDataRange.Offset : parts[1].InlineDataRange.Offset+parts[1].InlineDataRange.Length]
	if string(data) != "QUJD" {
		t.Fatalf("decoded inline data = %q", data)
	}

	var unsupported *UnsupportedContentError
	if err := parts[2].UnsupportedContentError(); !errors.As(err, &unsupported) {
		t.Fatalf("unknown part error = %v, want UnsupportedContentError", err)
	}

	if resp.UsageMetadata == nil || resp.UsageMetadata.TotalTokenCount != 3 {
		t.Fatalf("usage = %#v", resp.UsageMetadata)
	}
}

func TestParseGeminiSSEPayloadRejectsDeepNesting(t *testing.T) {
	nested := strings.Repeat("[", maxJSONNestingDepth+1) + strings.Repeat("]", maxJSONNestingDepth+1)
	for _, event := range []string{
		`{"ignored":` + nested + `}`,
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":` + nested + `}}]}}]}`,
	} {
		var payload testPayloadSink
		_, err := ParseGeminiSSEPayload(strings.NewReader(event), &payload)
		if err == nil || !strings.Contains(err.Error(), "nesting depth") {
			t.Fatalf("deep nesting error = %v, want nesting depth error", err)
		}
	}
}

func TestParseGeminiSSEPayloadRejectsInvalidUTF8(t *testing.T) {
	event := append([]byte(`{"candidates":[{"content":{"parts":[{"text":"`), 0xff)
	event = append(event, []byte(`"}]}}]}`)...)

	var payload testPayloadSink
	_, err := ParseGeminiSSEPayload(bytes.NewReader(event), &payload)
	if err == nil || !strings.Contains(err.Error(), "UTF-8") {
		t.Fatalf("invalid UTF-8 error = %v, want UTF-8 error", err)
	}
}

func TestParseGeminiSSEPayloadNullAndWhitespace(t *testing.T) {
	var payload testPayloadSink
	resp, err := ParseGeminiSSEPayload(bytes.NewReader([]byte("  \n\t")), &payload)
	if err != nil {
		t.Fatalf("whitespace: %v", err)
	}
	if resp != nil {
		t.Fatalf("whitespace response = %#v, want nil", resp)
	}

	resp, err = ParseGeminiSSEPayload(bytes.NewReader([]byte("null")), &payload)
	if err != nil {
		t.Fatalf("null: %v", err)
	}
	if resp == nil || len(resp.Candidates) != 0 {
		t.Fatalf("null response = %#v", resp)
	}
}

func TestParseGeminiSSEPayloadRejectsTrailingDataAfterNull(t *testing.T) {
	for _, event := range []string{
		`null {"error":{"message":"lost"}}`,
		`null{}`,
		`null garbage`,
		`nullx`,
	} {
		var payload testPayloadSink
		_, err := ParseGeminiSSEPayload(strings.NewReader(event), &payload)
		if err == nil || !strings.Contains(err.Error(), "unexpected data after JSON value") {
			t.Fatalf("%q: error = %v, want trailing data error", event, err)
		}
	}
}

func TestParseGeminiSSEPayloadAcceptsWhitespaceAfterNull(t *testing.T) {
	var payload testPayloadSink
	resp, err := ParseGeminiSSEPayload(strings.NewReader("null \n\t"), &payload)
	if err != nil {
		t.Fatalf("null with trailing whitespace: %v", err)
	}
	if resp == nil || len(resp.Candidates) != 0 {
		t.Fatalf("null response = %#v", resp)
	}
}

func TestParseGeminiSSEPayloadRejectsWidePartsArray(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"candidates":[{"content":{"parts":[`)
	for i := 0; i < maxSSEParts+1; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`{}`)
	}
	b.WriteString(`]}}]}`)

	var payload testPayloadSink
	_, err := ParseGeminiSSEPayload(strings.NewReader(b.String()), &payload)
	if err == nil || !strings.Contains(err.Error(), "too many parts") {
		t.Fatalf("wide parts error = %v, want parts count limit", err)
	}
}

func TestParseGeminiSSEPayloadRejectsWideUnknownFields(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"candidates":[{"content":{"parts":[{"text":"ok"`)
	for i := 0; i < maxSSEPartUnknownFields+1; i++ {
		b.WriteString(`,"u":0`)
	}
	b.WriteString(`}]}}]}`)

	var payload testPayloadSink
	_, err := ParseGeminiSSEPayload(strings.NewReader(b.String()), &payload)
	if err == nil || !strings.Contains(err.Error(), "too many unknown fields") {
		t.Fatalf("wide unknown fields error = %v, want unknown field count limit", err)
	}
}

type repeatingLiteralReader struct {
	prefix    []byte
	digit     byte
	remaining int64
	read      int64
}

func (r *repeatingLiteralReader) Read(p []byte) (int, error) {
	if len(r.prefix) > 0 {
		n := copy(p, r.prefix)
		r.prefix = r.prefix[n:]
		r.read += int64(n)
		return n, nil
	}
	if r.remaining <= 0 {
		return 0, io.EOF
	}
	if int64(len(p)) > r.remaining {
		p = p[:r.remaining]
	}
	for i := range p {
		p[i] = r.digit
	}
	r.remaining -= int64(len(p))
	r.read += int64(len(p))
	return len(p), nil
}

func TestParseGeminiSSEPayloadRejectsOversizedLiteralInSkippedField(t *testing.T) {
	r := &repeatingLiteralReader{
		prefix:    []byte(`{"ignored":`),
		digit:     '1',
		remaining: 64 << 20,
	}
	var payload testPayloadSink
	_, err := ParseGeminiSSEPayload(r, &payload)
	if err == nil || !strings.Contains(err.Error(), "literal exceeds") {
		t.Fatalf("oversized literal error = %v, want literal limit", err)
	}
	if r.read > maxSSELiteralBytes+(2<<20) {
		t.Fatalf("parser read %d bytes, want a bounded read", r.read)
	}
}

func TestParseGeminiSSEPayloadRejectsOversizedInteger(t *testing.T) {
	r := &repeatingLiteralReader{
		prefix:    []byte(`{"usageMetadata":{"totalTokenCount":`),
		digit:     '9',
		remaining: 64 << 20,
	}
	var payload testPayloadSink
	_, err := ParseGeminiSSEPayload(r, &payload)
	if err == nil || !strings.Contains(err.Error(), "literal exceeds") {
		t.Fatalf("oversized integer error = %v, want literal limit", err)
	}
	if r.read > maxSSEIntegerBytes+(2<<20) {
		t.Fatalf("parser read %d bytes, want a bounded read", r.read)
	}
}

func TestParseGeminiSSEPayloadRejectsOversizedRetainedLiteral(t *testing.T) {
	r := &repeatingLiteralReader{
		prefix:    []byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":`),
		digit:     '7',
		remaining: 96 << 20,
	}
	var payload testPayloadSink
	_, err := ParseGeminiSSEPayload(r, &payload)
	if err == nil || !strings.Contains(err.Error(), "metadata exceeds") {
		t.Fatalf("oversized retained literal error = %v, want metadata budget", err)
	}
	if r.read > maxSSEMetadataBytes+(2<<20) {
		t.Fatalf("parser read %d bytes, want a bounded read", r.read)
	}
}

func TestParseGeminiSSEPayloadRejectsLongInvalidLiteral(t *testing.T) {
	event := `{"ignored":` + strings.Repeat("1", maxSSELiteralBytes+1) + `x}`
	var payload testPayloadSink
	_, err := ParseGeminiSSEPayload(strings.NewReader(event), &payload)
	if err == nil || !strings.Contains(err.Error(), "literal exceeds") {
		t.Fatalf("long invalid literal error = %v, want literal limit", err)
	}
}

func TestSSEIntegerSyntaxMatchesJSONPath(t *testing.T) {
	tests := []struct {
		token string
		ok    bool
		want  int
	}{
		{token: "+1", ok: false},
		{token: "01", ok: false},
		{token: "-01", ok: false},
		{token: "1.0", ok: false},
		{token: "1e3", ok: false},
		{token: "99999999999999999999", ok: false},
		{token: "0", ok: true, want: 0},
		{token: "-0", ok: true, want: 0},
		{token: "1", ok: true, want: 1},
		{token: "-1", ok: true, want: -1},
	}
	for _, tt := range tests {
		t.Run(tt.token, func(t *testing.T) {
			jsonBody := []byte(`{"usageMetadata":{"totalTokenCount":` + tt.token + `}}`)
			jsonResp, jsonErr := ParseGeminiResponseJSON(jsonBody)

			var payload testPayloadSink
			event := `{"usageMetadata":{"totalTokenCount":` + tt.token + `}}`
			sseResp, sseErr := ParseGeminiSSEPayload(strings.NewReader(event), &payload)

			if (jsonErr == nil) != (sseErr == nil) {
				t.Fatalf("JSON error = %v, SSE error = %v; paths must agree", jsonErr, sseErr)
			}
			if !tt.ok {
				if sseErr == nil {
					t.Fatalf("SSE accepted invalid JSON integer %q", tt.token)
				}
				return
			}
			if sseErr != nil {
				t.Fatalf("SSE rejected valid integer %q: %v", tt.token, sseErr)
			}
			if sseResp == nil || sseResp.UsageMetadata == nil || sseResp.UsageMetadata.TotalTokenCount != tt.want {
				t.Fatalf("SSE usage = %#v, want totalTokenCount %d", sseResp, tt.want)
			}
			if jsonResp == nil || jsonResp.UsageMetadata == nil || jsonResp.UsageMetadata.TotalTokenCount != tt.want {
				t.Fatalf("JSON usage = %#v, want totalTokenCount %d", jsonResp, tt.want)
			}
		})
	}
}

// normalizeSSERanges replaces spill-backed ranges with the actual text and
// inline data so an SSE-decoded response can be compared with the regular JSON
// decoder.
func normalizeSSERanges(t *testing.T, resp *GeminiResponse, payload *testPayloadSink) {
	t.Helper()
	for ci := range resp.Candidates {
		for pi := range resp.Candidates[ci].Content.Parts {
			part := &resp.Candidates[ci].Content.Parts[pi]
			if part.TextRange != nil {
				text := string(payload.Bytes()[part.TextRange.Offset : part.TextRange.Offset+part.TextRange.Length])
				part.Text = &text
				part.TextRange = nil
			}
			if part.InlineDataRange != nil {
				data := string(payload.Bytes()[part.InlineDataRange.Offset : part.InlineDataRange.Offset+part.InlineDataRange.Length])
				if part.InlineData == nil {
					part.InlineData = &GeminiInlineData{}
				}
				part.InlineData.Data = data
				part.InlineDataRange = nil
			}
		}
	}
}

func TestSSEDuplicateNullMatchesJSONPath(t *testing.T) {
	tests := []struct {
		name              string
		event             string
		wantInlineDataNil bool
	}{
		{
			name:              "inlineData-object-then-null",
			event:             `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"AAAA"},"inlineData":null}]},"finishReason":"STOP"}]}`,
			wantInlineDataNil: true,
		},
		{
			name:  "functionCall-object-then-null",
			event: `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{"x":1}},"functionCall":null}]},"finishReason":"STOP"}]}`,
		},
		{
			name:  "functionResponse-object-then-null",
			event: `{"candidates":[{"content":{"parts":[{"functionResponse":{"name":"f","response":{"ok":true}},"functionResponse":null}]},"finishReason":"STOP"}]}`,
		},
		{
			name:  "fileData-object-then-null",
			event: `{"candidates":[{"content":{"parts":[{"fileData":{"mimeType":"image/png","fileUri":"gs://x"},"fileData":null}]},"finishReason":"STOP"}]}`,
		},
		{
			name:  "usageMetadata-object-then-null",
			event: `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":7},"usageMetadata":null}`,
		},
		{
			name:  "promptFeedback-object-then-null",
			event: `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"promptFeedback":{"blockReason":"SAFETY"},"promptFeedback":null}`,
		},
		{
			name:  "candidates-array-then-null",
			event: `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"candidates":null}`,
		},
		{
			name:  "parts-array-then-null",
			event: `{"candidates":[{"content":{"parts":[{"text":"ok"}],"parts":null}}]}`,
		},
		{
			name:  "text-string-then-null",
			event: `{"candidates":[{"content":{"parts":[{"text":"ok","text":null}]}}]}`,
		},
		{
			name:  "thought-bool-then-null",
			event: `{"candidates":[{"content":{"parts":[{"thought":true,"thought":null}]}}]}`,
		},
		{
			name:  "modelVersion-string-then-null",
			event: `{"modelVersion":"v1","modelVersion":null}`,
		},
		{
			name:  "responseId-string-then-null",
			event: `{"responseId":"r1","responseId":null}`,
		},
		{
			name:  "finishReason-string-then-null",
			event: `{"candidates":[{"finishReason":"STOP","finishReason":null}]}`,
		},
		{
			name:  "index-integer-then-null",
			event: `{"candidates":[{"index":3,"index":null}]}`,
		},
		{
			name:  "role-string-then-null",
			event: `{"candidates":[{"content":{"role":"model","role":null}}]}`,
		},
		{
			name:  "thoughtSignature-string-then-null",
			event: `{"candidates":[{"content":{"parts":[{"thoughtSignature":"sig","thoughtSignature":null}]}}]}`,
		},
		{
			name:  "functionCall-fields-then-null",
			event: `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","id":"i","name":null,"id":null}}]}}]}`,
		},
		{
			name:  "functionResponse-fields-then-null",
			event: `{"candidates":[{"content":{"parts":[{"functionResponse":{"name":"f","id":"i","response":{"ok":true},"name":null,"id":null}}]}}]}`,
		},
		{
			name:  "inlineData-fields-then-null",
			event: `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"AAAA","mimeType":null,"data":null}}]}}]}`,
		},
		{
			name:  "fileData-fields-then-null",
			event: `{"candidates":[{"content":{"parts":[{"fileData":{"mimeType":"image/png","fileUri":"gs://x","mimeType":null,"fileUri":null}}]}}]}`,
		},
		{
			name:  "usage-fields-then-null",
			event: `{"usageMetadata":{"totalTokenCount":7,"promptTokenCount":1,"totalTokenCount":null,"promptTokenCount":null}}`,
		},
		{
			name:  "promptFeedback-blockReason-then-null",
			event: `{"promptFeedback":{"blockReason":"SAFETY","blockReason":null}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jsonResp, err := ParseGeminiResponseJSON([]byte(tt.event))
			if err != nil {
				t.Fatalf("ParseGeminiResponseJSON: %v", err)
			}
			var payload testPayloadSink
			sseResp, err := ParseGeminiSSEPayload(strings.NewReader(tt.event), &payload)
			if err != nil {
				t.Fatalf("ParseGeminiSSEPayload: %v", err)
			}
			if tt.wantInlineDataNil {
				part := sseResp.Candidates[0].Content.Parts[0]
				if part.InlineData != nil || part.InlineDataRange != nil {
					t.Fatalf("inlineData = %#v, range = %#v; want both nil", part.InlineData, part.InlineDataRange)
				}
			}
			normalizeSSERanges(t, sseResp, &payload)
			if !reflect.DeepEqual(jsonResp, sseResp) {
				t.Fatalf("SSE response differs from JSON path\nJSON: %#v\nSSE:  %#v", jsonResp, sseResp)
			}
		})
	}
}
