package proxy

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/xzdmycbx/oai-to-gemini/internal/config"
	"github.com/xzdmycbx/oai-to-gemini/internal/convert"
)

func newTestServer(t *testing.T, upstreamURL string) *Server {
	t.Helper()
	return newTestServerWithCacheMax(t, upstreamURL, 1<<20)
}

func newTestServerWithCacheMax(t *testing.T, upstreamURL string, cacheMax int64) *Server {
	t.Helper()
	return newTestServerWithLogger(t, upstreamURL, cacheMax, log.New(io.Discard, "", 0))
}

func newTestServerWithLogger(t *testing.T, upstreamURL string, cacheMax int64, logger *log.Logger) *Server {
	t.Helper()
	cfg := &config.Config{
		Listen:                 ":0",
		SSESpillThresholdBytes: 1, // force the file-spill path in aggregate tests
		SSECacheMaxBytes:       cacheMax,
		SSECacheDir:            t.TempDir(),
		Providers: []config.Provider{{
			Name:    "newapi1",
			APIBase: upstreamURL + "/v1beta",
		}},
	}
	server, err := NewServer(cfg, logger)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	t.Cleanup(func() { _ = server.Close() })
	return server
}

func TestUnknownProviderReturnsOpenAI404(t *testing.T) {
	server := newTestServer(t, "https://example.invalid")
	req := httptest.NewRequest(http.MethodPost, "/missing/v1/chat/completions", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
	var envelope errorEnvelope
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("response is not JSON: %v", err)
	}
	if envelope.Error.Message == "" {
		t.Fatal("expected OpenAI error message")
	}
}

func TestNonStreamConversionAndAuthForwarding(t *testing.T) {
	const (
		authSecret  = "test-auth-secret"
		googSecret  = "test-goog-header-secret"
		querySecret = "test-query-secret"
	)
	var gotAuth, gotGoogKey, gotQueryKey string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotGoogKey = r.Header.Get("X-Goog-Api-Key")
		gotQueryKey = r.URL.Query().Get("key")
		if r.URL.Path != "/v1beta/models/gemini-test:generateContent" {
			t.Errorf("upstream path = %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"candidates": [{"content": {"parts": [{"text": "alpha"}, {"text": "beta"}]}, "finishReason": "STOP"}],
			"usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3}
		}`)
	}))
	defer upstream.Close()

	var logs bytes.Buffer
	server := newTestServerWithLogger(t, upstream.URL, 1<<20, log.New(&logs, "", 0))
	body := `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions?key="+querySecret, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+authSecret)
	req.Header.Set("X-Goog-Api-Key", googSecret)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if gotAuth != "Bearer "+authSecret {
		t.Fatalf("upstream Authorization = %q", gotAuth)
	}
	if gotGoogKey != googSecret {
		t.Fatalf("upstream X-Goog-Api-Key = %q", gotGoogKey)
	}
	if gotQueryKey != querySecret {
		t.Fatalf("upstream key query = %q", gotQueryKey)
	}
	assertSecretsNotLogged(t, logs.String(), authSecret, googSecret, querySecret)

	var out convertResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != "alphabeta" {
		t.Fatalf("content = %v, want alphabeta (no newline)", out.Choices[0].Message.Content)
	}
	if out.Usage == nil || out.Usage.TotalTokens != 3 {
		t.Fatalf("usage = %#v", out.Usage)
	}
}

func TestAggregateSSEConcatenatesWithoutNewline(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1beta/models/gemini-test:streamGenerateContent" {
			t.Errorf("upstream path = %q", r.URL.Path)
		}
		if r.URL.Query().Get("alt") != "sse" {
			t.Errorf("alt = %q", r.URL.Query().Get("alt"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, chunk := range []string{
			`{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b"},{"text":"c"}]}}]}`,
			`{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":3,"totalTokenCount":4}}`,
		} {
			_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out convertResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != "abc" {
		t.Fatalf("content = %v, want abc", out.Choices[0].Message.Content)
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %q", out.Choices[0].FinishReason)
	}
}

func TestAggregateSSEQueryAndHeaderEntriesEndToEnd(t *testing.T) {
	for _, tc := range []struct {
		name   string
		target string
		header string
	}{
		{"query-only", "/newapi1/v1/chat/completions?aggregate_sse=true", ""},
		{"header-only", "/newapi1/v1/chat/completions", "true"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath, gotAlt string
			var gotQuery url.Values
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				gotQuery = r.URL.Query()
				gotAlt = gotQuery.Get("alt")
				w.Header().Set("Content-Type", "text/event-stream")
				for _, chunk := range []string{
					`{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}`,
					`{"candidates":[{"content":{"parts":[{"text":"b"}]}}]}`,
					`{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}`,
				} {
					_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
				}
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
			}))
			defer upstream.Close()

			server := newTestServer(t, upstream.URL)
			body := `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`
			req := httptest.NewRequest(http.MethodPost, tc.target, strings.NewReader(body))
			if tc.header != "" {
				req.Header.Set("X-Aggregate-SSE", tc.header)
			}
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
			if gotPath != "/v1beta/models/gemini-test:streamGenerateContent" {
				t.Fatalf("upstream path = %q", gotPath)
			}
			if gotAlt != "sse" {
				t.Fatalf("upstream alt = %q, want sse", gotAlt)
			}
			if _, ok := gotQuery["aggregate_sse"]; ok {
				t.Fatalf("aggregate_sse was forwarded upstream: %v", gotQuery)
			}
			if contentType := rec.Header().Get("Content-Type"); !strings.HasPrefix(contentType, "application/json") {
				t.Fatalf("response content type = %q, want application/json", contentType)
			}
			var out convertResponse
			if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
				t.Fatalf("response is not a single OpenAI JSON response: %v, body = %s", err, rec.Body.String())
			}
			if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != "ab" {
				t.Fatalf("content = %v, want ab", out.Choices[0].Message.Content)
			}
			if out.Choices[0].FinishReason != "stop" {
				t.Fatalf("finish_reason = %q", out.Choices[0].FinishReason)
			}
		})
	}
}

func TestAggregateSSEEmptyTextEventsAreAccepted(t *testing.T) {
	tests := []struct {
		name      string
		events    []string
		wantUsage bool
	}{
		{
			name: "standalone",
			events: []string{
				`{"candidates":[{"content":{"parts":[{"text":""}]}}]}`,
				`{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}`,
			},
		},
		{
			name: "terminal",
			events: []string{
				`{"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}]}`,
			},
		},
		{
			name: "usage-tail",
			events: []string{
				`{"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}]}`,
				`{"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}`,
			},
			wantUsage: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				for _, event := range tc.events {
					_, _ = io.WriteString(w, "data: "+event+"\n\n")
				}
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
			}))
			defer upstream.Close()

			server := newTestServer(t, upstream.URL)
			body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200, body = %s", rec.Code, rec.Body.String())
			}
			var out convertResponse
			if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
				t.Fatalf("response JSON: %v", err)
			}
			if len(out.Choices) != 1 {
				t.Fatalf("choices = %d, want 1", len(out.Choices))
			}
			if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != "" {
				t.Fatalf("content = %v, want empty string", out.Choices[0].Message.Content)
			}
			if out.Choices[0].FinishReason != "stop" {
				t.Fatalf("finish_reason = %q, want stop", out.Choices[0].FinishReason)
			}
			if tc.wantUsage {
				if out.Usage == nil || out.Usage.TotalTokens != 3 {
					t.Fatalf("usage = %#v, want total_tokens 3", out.Usage)
				}
			}
		})
	}
}

func TestStreamingSSEConcatenatesWithoutNewline(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, chunk := range []string{
			`{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b"}]}}]}`,
			`{"candidates":[{"content":{"parts":[]},"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}`,
		} {
			_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "data: [DONE]") {
		t.Fatalf("missing [DONE]: %s", rec.Body.String())
	}

	var content strings.Builder
	var finishReason string
	scanner := bufio.NewScanner(strings.NewReader(rec.Body.String()))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content *string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatalf("chunk JSON: %v", err)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		if chunk.Choices[0].Delta.Content != nil {
			content.WriteString(*chunk.Choices[0].Delta.Content)
		}
		if chunk.Choices[0].FinishReason != nil {
			finishReason = *chunk.Choices[0].FinishReason
		}
	}
	if content.String() != "ab" {
		t.Fatalf("streamed content = %q, want ab", content.String())
	}
	if finishReason != "length" {
		t.Fatalf("finish_reason = %q, want length", finishReason)
	}
}

type convertResponse struct {
	Choices []struct {
		Message struct {
			Content *string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func TestCrossOriginRedirectDoesNotForwardAPIKey(t *testing.T) {
	const (
		googSecret  = "test-redirect-goog-secret"
		querySecret = "test-redirect-query-secret"
	)
	var targetCalled bool
	var targetKey, targetQueryKey string
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		targetCalled = true
		targetKey = r.Header.Get("X-Goog-Api-Key")
		targetQueryKey = r.URL.Query().Get("key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"models":[]}`)
	}))
	defer target.Close()

	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer source.Close()

	var logs bytes.Buffer
	server := newTestServerWithLogger(t, source.URL, 1<<20, log.New(&logs, "", 0))
	req := httptest.NewRequest(http.MethodGet, "/newapi1/v1/models?key="+querySecret, nil)
	req.Header.Set("X-Goog-Api-Key", googSecret)
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if targetCalled {
		t.Fatal("cross-origin redirect target was called")
	}
	if targetKey != "" {
		t.Fatalf("redirect target received API key %q", targetKey)
	}
	if targetQueryKey != "" {
		t.Fatalf("redirect target received key query %q", targetQueryKey)
	}
	if rec.Code == http.StatusOK {
		t.Fatalf("status = %d, want redirect not followed as success", rec.Code)
	}
	assertSecretsNotLogged(t, logs.String(), googSecret, querySecret)
}

func assertSecretsNotLogged(t *testing.T, logs string, secrets ...string) {
	t.Helper()
	for _, secret := range secrets {
		if strings.Contains(logs, secret) {
			t.Fatalf("log contains secret %q: %s", secret, logs)
		}
	}
	for _, marker := range []string{"Authorization", "X-Goog-Api-Key", "key="} {
		if strings.Contains(logs, marker) {
			t.Fatalf("log contains credential marker %q: %s", marker, logs)
		}
	}
}

func TestAggregateSSEDeepNestingFailsAndCleansUp(t *testing.T) {
	const depth = 5000
	nested := strings.Repeat("[", depth) + strings.Repeat("]", depth)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"prefix\"}]}}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"ignored\":"+nested+"}\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "nesting depth") {
		t.Fatalf("expected nesting depth error, body = %s", rec.Body.String())
	}
	assertDirEmpty(t, server.store.Dir())
}

func TestAggregateSSEMultiLineDataEvent(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":\n")
		_, _ = io.WriteString(w, "data: [{\"content\":{\"parts\":[{\"text\":\"ab\"}]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out convertResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != "ab" {
		t.Fatalf("content = %v, want ab", out.Choices[0].Message.Content)
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %q", out.Choices[0].FinishReason)
	}
}

func TestAggregateSSEIncompleteStreamFails(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "upstream_incomplete") {
		t.Fatalf("expected upstream_incomplete error, body = %s", rec.Body.String())
	}
}

func TestStreamingSSEIncompleteStreamFails(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "upstream_incomplete") {
		t.Fatalf("expected upstream_incomplete error, body = %s", rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), `"finish_reason":"stop"`) {
		t.Fatalf("incomplete stream was reported as a normal stop: %s", rec.Body.String())
	}
}

func TestPromptFeedbackBlockReturnsContentFilter(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"promptFeedback":{"blockReason":"SAFETY"}}`)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out convertResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].FinishReason != "content_filter" {
		t.Fatalf("finish_reason = %q, want content_filter", out.Choices[0].FinishReason)
	}
}

func TestStreamingIgnoresInvalidAggregateSSE(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","stream":true,"aggregate_sse":"invalid","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "ok") {
		t.Fatalf("expected streamed content, body = %s", rec.Body.String())
	}
}

func TestAggregateSSEStreamsLargeUTF8ContentFromSpill(t *testing.T) {
	content := strings.Repeat("你好abc", 20000)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk, err := json.Marshal(map[string]any{
			"candidates": []any{map[string]any{
				"content":      map[string]any{"parts": []any{map[string]any{"text": content}}},
				"finishReason": "STOP",
			}},
		})
		if err != nil {
			t.Errorf("marshal upstream chunk: %v", err)
			return
		}
		_, _ = io.WriteString(w, "data: "+string(chunk)+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out convertResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != content {
		t.Fatal("large UTF-8 content did not round-trip exactly")
	}
}

func TestCandidateCountGreaterThanOneIsRejectedBeforeUpstream(t *testing.T) {
	var called bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[]}`)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","candidateCount":2,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400, body = %s", rec.Code, rec.Body.String())
	}
	if called {
		t.Fatal("upstream was called for an unsupported candidateCount")
	}
}

func TestOpenAINGreaterThanOneIsRejectedBeforeUpstream(t *testing.T) {
	var called bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[]}`)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","n":2,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400, body = %s", rec.Code, rec.Body.String())
	}
	if called {
		t.Fatal("upstream was called for an unsupported OpenAI n")
	}
}

func TestSSELineEndingsAreEquivalent(t *testing.T) {
	for _, eol := range []struct {
		name string
		sep  string
	}{
		{"LF", "\n"},
		{"CRLF", "\r\n"},
		{"CR", "\r"},
	} {
		eol := eol
		t.Run(eol.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}"+eol.sep+eol.sep)
				_, _ = io.WriteString(w, "data: [DONE]"+eol.sep+eol.sep)
			}))
			defer upstream.Close()

			server := newTestServer(t, upstream.URL)
			for _, tc := range []struct {
				name string
				body string
			}{
				{"stream", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
				{"aggregate", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`},
			} {
				t.Run(tc.name, func(t *testing.T) {
					req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
					rec := httptest.NewRecorder()
					server.ServeHTTP(rec, req)

					if rec.Code != http.StatusOK {
						t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
					}
					response := rec.Body.String()
					if !strings.Contains(response, `"content":"ok"`) {
						t.Fatalf("missing text content, body = %s", response)
					}
					if !strings.Contains(response, `"finish_reason":"stop"`) {
						t.Fatalf("missing normal finish reason, body = %s", response)
					}
					if tc.name == "stream" && !strings.Contains(response, "data: [DONE]") {
						t.Fatalf("missing completion marker, body = %s", response)
					}
				})
			}
		})
	}
}

func TestEmbeddedGeminiErrorMessagesAreUnwrapped(t *testing.T) {
	const geminiError = `{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: "+geminiError+"\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, geminiError)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	for _, tc := range []struct {
		name string
		body string
	}{
		{"json", `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`},
		{"sse", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			response := rec.Body.String()
			if !strings.Contains(response, "quota exceeded") {
				t.Fatalf("missing unwrapped error message, body = %s", response)
			}
			if strings.Contains(response, `\"code\"`) || strings.Contains(response, `\"message\"`) {
				t.Fatalf("error message contains nested JSON, body = %s", response)
			}
		})
	}
}

func TestPromptBlockReasonOtherReturnsContentFilter(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"promptFeedback\":{\"blockReason\":\"OTHER\"}}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"promptFeedback":{"blockReason":"OTHER"}}`)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	for _, tc := range []struct {
		name string
		body string
	}{
		{"non-stream", `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`},
		{"aggregate", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"stream", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), `"content_filter"`) {
				t.Fatalf("expected content_filter, body = %s", rec.Body.String())
			}
		})
	}
}

func TestModelsPaginationMergesAllPages(t *testing.T) {
	var tokens []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.URL.Query().Get("pageToken")
		tokens = append(tokens, token)
		w.Header().Set("Content-Type", "application/json")
		switch token {
		case "":
			_, _ = io.WriteString(w, `{"models":[{"name":"models/model-a"}],"nextPageToken":"page-2"}`)
		case "page-2":
			_, _ = io.WriteString(w, `{"models":[{"name":"models/model-b"}],"nextPageToken":"page-3"}`)
		case "page-3":
			_, _ = io.WriteString(w, `{"models":[{"name":"models/model-c"}]}`)
		default:
			t.Errorf("unexpected pageToken %q", token)
			_, _ = io.WriteString(w, `{"models":[]}`)
		}
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	req := httptest.NewRequest(http.MethodGet, "/newapi1/v1/models", nil)
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	var ids []string
	for _, m := range out.Data {
		ids = append(ids, m.ID)
	}
	if strings.Join(ids, ",") != "model-a,model-b,model-c" {
		t.Fatalf("models = %v, want all three pages", ids)
	}
	if strings.Join(tokens, ",") != ",page-2,page-3" {
		t.Fatalf("page tokens = %v", tokens)
	}
}

func TestModelsRejectsCumulativeNameBytesOverLimit(t *testing.T) {
	const perPage = 140
	name := "models/" + strings.Repeat("a", maxModelNameBytes-len("models/"))
	buildPage := func(next string) string {
		var b strings.Builder
		b.WriteString(`{"models":[`)
		for i := 0; i < perPage; i++ {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(`{"name":"`)
			b.WriteString(name)
			b.WriteString(`"}`)
		}
		b.WriteByte(']')
		if next != "" {
			b.WriteString(`,"nextPageToken":"`)
			b.WriteString(next)
			b.WriteString(`"`)
		}
		b.WriteByte('}')
		return b.String()
	}
	pageBytes := int64(perPage * len(name))
	if pageBytes > maxModelsTotalBytes {
		t.Fatalf("test page alone exceeds budget: %d > %d", pageBytes, maxModelsTotalBytes)
	}
	if 2*pageBytes <= maxModelsTotalBytes {
		t.Fatalf("test pages do not exceed budget: %d <= %d", 2*pageBytes, maxModelsTotalBytes)
	}
	page1 := buildPage("page-2")
	page2 := buildPage("")

	var requests int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("pageToken") == "" {
			_, _ = io.WriteString(w, page1)
			return
		}
		_, _ = io.WriteString(w, page2)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	req := httptest.NewRequest(http.MethodGet, "/newapi1/v1/models", nil)
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502, body = %s", rec.Code, rec.Body.String())
	}
	var envelope errorEnvelope
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("response is not JSON: %v, body = %s", err, rec.Body.String())
	}
	if envelope.Error.Message == "" {
		t.Fatalf("expected an OpenAI error message, body = %s", rec.Body.String())
	}
	if requests != 2 {
		t.Fatalf("upstream requests = %d, want 2", requests)
	}
}

func TestModelsRejectsBudgetOverruns(t *testing.T) {
	buildList := func(key, field string, count int, value string) string {
		var b strings.Builder
		b.WriteString(`{"`)
		b.WriteString(key)
		b.WriteString(`":[`)
		for i := 0; i < count; i++ {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(`{"`)
			b.WriteString(field)
			b.WriteString(`":"`)
			b.WriteString(value)
			b.WriteString(`"}`)
		}
		b.WriteString(`]}`)
		return b.String()
	}
	oversizedID := strings.Repeat("a", maxModelNameBytes)
	overByteCount := maxModelsTotalBytes/maxModelNameBytes + 1
	for _, tc := range []struct {
		name string
		body string
	}{
		{"openai-too-many-entries", buildList("data", "id", maxModels+1, "x")},
		{"openai-too-many-id-bytes", buildList("data", "id", overByteCount, oversizedID)},
		{"gemini-too-many-entries", buildList("models", "name", maxModels+1, "x")},
		{"gemini-too-many-name-bytes", buildList("models", "name", overByteCount, oversizedID)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.body)
			}))
			defer upstream.Close()

			server := newTestServer(t, upstream.URL)
			req := httptest.NewRequest(http.MethodGet, "/newapi1/v1/models", nil)
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusBadGateway {
				t.Fatalf("status = %d, want 502, body = %s", rec.Code, rec.Body.String())
			}
			var envelope errorEnvelope
			if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
				t.Fatalf("response is not JSON: %v, body = %s", err, rec.Body.String())
			}
			if envelope.Error.Code != "models_too_large" {
				t.Fatalf("code = %v, want models_too_large, body = %s", envelope.Error.Code, rec.Body.String())
			}
		})
	}
}

func TestScanModelArrayRejectsExtraItemsBeforeDecoding(t *testing.T) {
	for _, tc := range []struct {
		name   string
		raw    string
		handle func(index int, item json.RawMessage) (int, error)
	}{
		{"openai", `[{"id":"model-a"},not-json]`, parseOpenAIModelEntry},
		{"gemini", `[{"name":"model-a"},not-json]`, func(index int, item json.RawMessage) (int, error) {
			name, err := parseGeminiModelEntry(index, item)
			return len(name), err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := scanModelArray(json.RawMessage(tc.raw), 1, maxModelsTotalBytes, tc.handle); !errors.Is(err, errModelListTooLarge) {
				t.Fatalf("err = %v, want errModelListTooLarge", err)
			}
		})
	}
}

func TestModelsRejectsInvalidSuccessResponses(t *testing.T) {
	for _, tc := range []struct {
		name        string
		body        string
		wantMessage string
	}{
		{"empty-object", `{}`, ""},
		{"null", `null`, ""},
		{"openai-null-data", `{"data":null}`, ""},
		{"openai-non-array-data", `{"data":{}}`, ""},
		{"openai-non-object-entry", `{"data":["model-a"]}`, ""},
		{"openai-missing-id", `{"data":[{}]}`, ""},
		{"openai-empty-id", `{"data":[{"id":""}]}`, ""},
		{"openai-whitespace-id", `{"data":[{"id":"   "}]}`, ""},
		{"openai-id-not-string", `{"data":[{"id":123}]}`, ""},
		{"openai-valid-and-invalid", `{"data":[{"id":"model-a"},{}]}`, ""},
		{"openai-id-too-long", `{"data":[{"id":"` + strings.Repeat("a", maxModelNameBytes+1) + `"}]}`, ""},
		{"gemini-null-models", `{"models":null}`, ""},
		{"gemini-non-array-models", `{"models":{}}`, ""},
		{"gemini-non-object-entry", `{"models":["model-a"]}`, ""},
		{"gemini-missing-name", `{"models":[{}]}`, ""},
		{"gemini-empty-name", `{"models":[{"name":""}]}`, ""},
		{"gemini-prefix-only-name", `{"models":[{"name":"models/"}]}`, ""},
		{"gemini-name-not-string", `{"models":[{"name":123}]}`, ""},
		{"gemini-name-too-long", `{"models":[{"name":"` + strings.Repeat("a", maxModelNameBytes+1) + `"}]}`, ""},
		{"gemini-valid-and-invalid", `{"models":[{"name":"models/model-a"},{}]}`, ""},
		{"embedded-error", `{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`, "quota exceeded"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.body)
			}))
			defer upstream.Close()

			server := newTestServer(t, upstream.URL)
			req := httptest.NewRequest(http.MethodGet, "/newapi1/v1/models", nil)
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusBadGateway {
				t.Fatalf("status = %d, want 502, body = %s", rec.Code, rec.Body.String())
			}
			var envelope errorEnvelope
			if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
				t.Fatalf("response is not JSON: %v, body = %s", err, rec.Body.String())
			}
			if envelope.Error.Message == "" {
				t.Fatalf("expected an OpenAI error message, body = %s", rec.Body.String())
			}
			if tc.wantMessage != "" && !strings.Contains(envelope.Error.Message, tc.wantMessage) {
				t.Fatalf("message = %q, want %q", envelope.Error.Message, tc.wantMessage)
			}
		})
	}
}

func TestModelsAcceptsValidAndEmptyLists(t *testing.T) {
	for _, tc := range []struct {
		name      string
		body      string
		wantIDs   []string
		wantEmpty bool
	}{
		{"openai-list", `{"object":"list","data":[{"id":"model-a","object":"model"}]}`, []string{"model-a"}, false},
		{"openai-empty", `{"data":[]}`, nil, true},
		{"gemini-empty", `{"models":[]}`, nil, true},
		{"gemini-bare-name", `{"models":[{"name":"model-a"}]}`, []string{"model-a"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.body)
			}))
			defer upstream.Close()

			server := newTestServer(t, upstream.URL)
			req := httptest.NewRequest(http.MethodGet, "/newapi1/v1/models", nil)
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
			var out struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
				t.Fatalf("response JSON: %v, body = %s", err, rec.Body.String())
			}
			if tc.wantEmpty && len(out.Data) != 0 {
				t.Fatalf("data = %#v, want empty", out.Data)
			}
			var ids []string
			for _, model := range out.Data {
				ids = append(ids, model.ID)
			}
			if strings.Join(ids, ",") != strings.Join(tc.wantIDs, ",") {
				t.Fatalf("ids = %v, want %v", ids, tc.wantIDs)
			}
		})
	}
}

func TestModelsEmptyListWithManyUnknownTopLevelFieldsAllocatesBounded(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"models":[]`)
	for i := 0; i < 100000; i++ {
		b.WriteString(`,"k`)
		b.WriteString(strconv.Itoa(i))
		b.WriteString(`":null`)
	}
	b.WriteString(`}`)
	body := b.String()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	req := httptest.NewRequest(http.MethodGet, "/newapi1/v1/models", nil)
	rec := httptest.NewRecorder()

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	server.ServeHTTP(rec, req)
	runtime.ReadMemStats(&after)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if len(out.Data) != 0 {
		t.Fatalf("data = %#v, want empty", out.Data)
	}
	// Unknown top-level fields must be skipped during typed decoding rather
	// than expanded into one map entry and raw value copy per field.
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 8<<20 {
		t.Fatalf("models handler allocated %d bytes for a wide top-level object; want < 8 MiB", allocated)
	}
}

func TestModelsOpenAIPassthroughPreservesOriginalBytes(t *testing.T) {
	body := `{"object":"list","data":[{"id":"model-a","object":"model","created":1,"owned_by":"x","extra":{"k":null}}],"custom":"keep"}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	req := httptest.NewRequest(http.MethodGet, "/newapi1/v1/models", nil)
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if got := rec.Body.String(); got != body {
		t.Fatalf("OpenAI-style response was not passed through byte-for-byte:\n got: %s\nwant: %s", got, body)
	}
}

func assertDirEmpty(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("cache dir still has %d entries after request", len(entries))
	}
}

func TestAggregateOverSoftLimitReturnsFullAndCleansUp(t *testing.T) {
	content := strings.Repeat("完整内容abc", 40000)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		half := len(content) / 2
		for half > 0 && content[half]&0xC0 == 0x80 {
			half--
		}
		for _, piece := range []string{content[:half], content[half:]} {
			chunk, err := json.Marshal(map[string]any{
				"candidates": []any{map[string]any{
					"content": map[string]any{"parts": []any{map[string]any{"text": piece}}},
				}},
			})
			if err != nil {
				t.Errorf("marshal chunk: %v", err)
				return
			}
			_, _ = io.WriteString(w, "data: "+string(chunk)+"\n\n")
		}
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServerWithCacheMax(t, upstream.URL, 1024)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out convertResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != content {
		t.Fatal("content over the cache soft limit was not returned in full")
	}
	assertDirEmpty(t, server.store.Dir())
}

func TestAggregateErrorCleansUpTempFiles(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n")
	}))
	defer upstream.Close()

	server := newTestServerWithCacheMax(t, upstream.URL, 1024)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	assertDirEmpty(t, server.store.Dir())
}

func TestAggregateCancelCleansUpTempFiles(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	defer upstream.Close()

	server := newTestServerWithCacheMax(t, upstream.URL, 1024)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body)).WithContext(ctx)
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	assertDirEmpty(t, server.store.Dir())
}

func TestShutdownWaitsForSpilledAggregateRequest(t *testing.T) {
	content := strings.Repeat("shutdown-content", 1024)
	sent := make(chan struct{})
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk, err := json.Marshal(map[string]any{
			"candidates": []any{map[string]any{
				"content": map[string]any{"parts": []any{map[string]any{"text": content}}},
			}},
		})
		if err != nil {
			t.Errorf("marshal upstream chunk: %v", err)
			return
		}
		_, _ = io.WriteString(w, "data: "+string(chunk)+"\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		close(sent)
		<-release
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	requestDone := make(chan struct{})
	go func() {
		server.ServeHTTP(rec, req)
		close(requestDone)
	}()

	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("upstream did not send the first aggregate event")
	}
	waitForSpillFile(t, server.store.Dir())

	shutdownDone := make(chan error, 1)
	go func() { shutdownDone <- server.Shutdown(context.Background()) }()

	select {
	case err := <-shutdownDone:
		t.Fatalf("Shutdown returned while an aggregate request was still active: %v", err)
	case <-time.After(100 * time.Millisecond):
	}

	close(release)
	select {
	case <-requestDone:
	case <-time.After(10 * time.Second):
		t.Fatal("aggregate request did not finish")
	}
	select {
	case err := <-shutdownDone:
		if err != nil {
			t.Fatalf("Shutdown: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Shutdown did not finish after the request completed")
	}

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out convertResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != content {
		t.Fatal("spilled aggregate response was not returned in full during shutdown")
	}
	if _, err := os.Stat(server.store.Dir()); !os.IsNotExist(err) {
		t.Fatalf("cache instance dir still exists after shutdown: %v", err)
	}
}

func TestShutdownCancelsSpilledAggregateAfterDeadline(t *testing.T) {
	sent := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		close(sent)
		<-r.Context().Done()
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	requestDone := make(chan struct{})
	go func() {
		server.ServeHTTP(rec, req)
		close(requestDone)
	}()

	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("upstream did not send the first aggregate event")
	}
	waitForSpillFile(t, server.store.Dir())

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	select {
	case <-requestDone:
	case <-time.After(10 * time.Second):
		t.Fatal("canceled aggregate request did not release its resources")
	}
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502, body = %s", rec.Code, rec.Body.String())
	}
	if _, err := os.Stat(server.store.Dir()); !os.IsNotExist(err) {
		t.Fatalf("cache instance dir still exists after forced shutdown: %v", err)
	}
}

func waitForSpillFile(t *testing.T, dir string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("ReadDir: %v", err)
		}
		if len(entries) > 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("aggregate payload did not spill to a temp file")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func parseContentParts(t *testing.T, raw json.RawMessage) []convert.OpenAIContentPart {
	t.Helper()
	var parts []convert.OpenAIContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		t.Fatalf("content is not an array: %v (%s)", err, raw)
	}
	return parts
}

func TestNonStreamMixedTextAndImagePreserved(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"candidates": [{"content": {"parts": [
				{"text": "alpha"},
				{"inlineData": {"mimeType": "image/png", "data": "AAAA"}},
				{"text": "beta"}
			]}, "finishReason": "STOP"}]
		}`)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content json.RawMessage `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	parts := parseContentParts(t, out.Choices[0].Message.Content)
	if len(parts) != 3 {
		t.Fatalf("content parts = %#v", parts)
	}
	if parts[0].Type != "text" || parts[0].Text != "alpha" {
		t.Fatalf("part 0 = %#v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil || parts[1].ImageURL.URL != "data:image/png;base64,AAAA" {
		t.Fatalf("part 1 = %#v", parts[1])
	}
	if parts[2].Type != "text" || parts[2].Text != "beta" {
		t.Fatalf("part 2 = %#v", parts[2])
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %q", out.Choices[0].FinishReason)
	}
}

func TestAggregateMixedTextAndImagePreserved(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"alpha\"}]}}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"AAAA\"}},{\"text\":\"beta\"}]}}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	parts := parseContentParts(t, out.Choices[0].Message.Content)
	if len(parts) != 3 {
		t.Fatalf("content parts = %#v", parts)
	}
	if parts[0].Text != "alpha" || parts[1].ImageURL == nil || parts[1].ImageURL.URL != "data:image/png;base64,AAAA" || parts[2].Text != "beta" {
		t.Fatalf("content parts = %#v", parts)
	}
}

func TestStreamingImageResponsePreserved(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"AAAA\"}}]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var foundImage bool
	scanner := bufio.NewScanner(strings.NewReader(rec.Body.String()))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content json.RawMessage `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatalf("chunk JSON: %v", err)
		}
		if len(chunk.Choices) == 0 || len(chunk.Choices[0].Delta.Content) == 0 {
			continue
		}
		var parts []convert.OpenAIContentPart
		if err := json.Unmarshal(chunk.Choices[0].Delta.Content, &parts); err != nil {
			continue
		}
		for _, part := range parts {
			if part.Type == "image_url" && part.ImageURL != nil && part.ImageURL.URL == "data:image/png;base64,AAAA" {
				foundImage = true
			}
		}
	}
	if !foundImage {
		t.Fatalf("image was not streamed: %s", rec.Body.String())
	}
}

func TestAbnormalFinishReasonsAreErrorsAcrossPaths(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"MALFORMED_FUNCTION_CALL\"}]}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"parts":[]},"finishReason":"MALFORMED_FUNCTION_CALL"}]}`)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	for _, tc := range []struct {
		name string
		body string
	}{
		{"non-stream", `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`},
		{"aggregate", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"stream", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)
			body := rec.Body.String()
			if !strings.Contains(body, "abnormal_finish") {
				t.Fatalf("expected abnormal_finish error, status = %d, body = %s", rec.Code, body)
			}
			if strings.Contains(body, `"finish_reason":"stop"`) || strings.Contains(body, `"finish_reason":"tool_calls"`) {
				t.Fatalf("abnormal finish was reported as success: %s", body)
			}
		})
	}
}

func TestStalledUpstreamIdleTimeout(t *testing.T) {
	old := upstreamResponseIdleTimeout
	upstreamResponseIdleTimeout = 100 * time.Millisecond
	defer func() { upstreamResponseIdleTimeout = old }()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	for _, tc := range []struct {
		name string
		body string
	}{
		{"aggregate", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"stream", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body)).WithContext(ctx)
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)
			if !strings.Contains(rec.Body.String(), "idle timeout") {
				t.Fatalf("expected idle timeout error, status = %d, body = %s", rec.Code, rec.Body.String())
			}
			if tc.name == "aggregate" {
				assertDirEmpty(t, server.store.Dir())
			}
		})
	}
}

func TestSlowDownstreamWriteDoesNotTripUpstreamIdleTimeout(t *testing.T) {
	old := upstreamResponseIdleTimeout
	upstreamResponseIdleTimeout = 200 * time.Millisecond
	defer func() { upstreamResponseIdleTimeout = old }()

	release := make(chan struct{}, 1)
	releaseUpstream := func() {
		select {
		case release <- struct{}{}:
		default:
		}
	}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"first\"}]}}]}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"second\"}]}}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":2,\"totalTokenCount\":3}}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	slow := &slowSecondWriteRecorder{
		ResponseWriter: rec,
		delay:          300 * time.Millisecond,
		release:        releaseUpstream,
	}
	done := make(chan struct{})
	go func() {
		server.ServeHTTP(slow, req)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("request did not finish")
	}

	out := rec.Body.String()
	for _, want := range []string{`"first"`, `"second"`, `"finish_reason":"stop"`, `"total_tokens":3`, `[DONE]`} {
		if !strings.Contains(out, want) {
			t.Fatalf("response missing %q: %s", want, out)
		}
	}
}

type slowSecondWriteRecorder struct {
	http.ResponseWriter
	writes  int
	delay   time.Duration
	release func()
}

func (w *slowSecondWriteRecorder) Write(p []byte) (int, error) {
	w.writes++
	if w.writes == 2 {
		time.Sleep(w.delay)
	}
	n, err := w.ResponseWriter.Write(p)
	if w.writes == 2 && w.release != nil {
		w.release()
		w.release = nil
	}
	return n, err
}

func (w *slowSecondWriteRecorder) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func TestOversizedSSELiteralIsRejectedAndCleanedUp(t *testing.T) {
	event := `{"ignored":` + strings.Repeat("1", 2<<20) + `}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: "+event+"\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	for _, tc := range []struct {
		name string
		body string
	}{
		{"aggregate", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"stream", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)
			if !strings.Contains(rec.Body.String(), "literal exceeds") {
				t.Fatalf("expected literal limit error, status = %d, body = %s", rec.Code, rec.Body.String())
			}
			assertDirEmpty(t, server.store.Dir())
		})
	}
}

func TestUpstreamSendTimeoutWhenBodyIsNotRead(t *testing.T) {
	old := upstreamRequestSendTimeout
	upstreamRequestSendTimeout = 200 * time.Millisecond
	defer func() { upstreamRequestSendTimeout = old }()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	defer ln.Close()

	accepted := make(chan net.Conn, 1)
	go func() {
		conn, err := ln.Accept()
		if err == nil {
			accepted <- conn
		}
	}()
	defer func() {
		select {
		case conn := <-accepted:
			_ = conn.Close()
		default:
		}
	}()

	server := newTestServer(t, "http://"+ln.Addr().String())
	payload, err := json.Marshal(map[string]any{
		"model": "gemini-test",
		"messages": []any{map[string]any{
			"role":    "user",
			"content": strings.Repeat("a", 16<<20),
		}},
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		server.ServeHTTP(rec, req)
		close(done)
	}()

	// Wait for the upstream listener to accept the connection before timing
	// the send phase. Request preprocessing (reading, scanning, converting and
	// marshaling a 16 MiB body) is not part of the send timeout and can take
	// longer than 10 seconds under -race.
	select {
	case conn := <-accepted:
		defer conn.Close()
	case <-done:
		t.Fatalf("request exited before the upstream connection was accepted: status = %d, body = %s", rec.Code, rec.Body.String())
	case <-time.After(60 * time.Second):
		t.Fatal("upstream connection was not accepted before the preprocessing deadline")
	}
	sendPhaseStart := time.Now()
	select {
	case <-done:
		if elapsed := time.Since(sendPhaseStart); elapsed > 5*time.Second {
			t.Fatalf("request took %s after the upstream connection was accepted, want the send timeout to cancel it promptly", elapsed)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("request did not exit after upstream send timeout")
	}
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "send timed out") {
		t.Fatalf("expected send timeout error, body = %s", rec.Body.String())
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if !server.waitRequests(ctx) {
		t.Fatal("request resources were not released")
	}
}

func TestSameOriginRedirectReplaySendTimeout(t *testing.T) {
	old := upstreamRequestSendTimeout
	upstreamRequestSendTimeout = 500 * time.Millisecond
	defer func() { upstreamRequestSendTimeout = old }()

	for _, tc := range []struct {
		name   string
		status int
	}{
		{"307", http.StatusTemporaryRedirect},
		{"308", http.StatusPermanentRedirect},
	} {
		t.Run(tc.name, func(t *testing.T) {
			release := make(chan struct{})
			targetStarted := make(chan struct{})
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1beta/models/gemini-test:generateContent" {
					_, _ = io.Copy(io.Discard, r.Body)
					http.Redirect(w, r, "/retry", tc.status)
					return
				}
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					return
				}
				close(targetStarted)
				<-release
				_ = conn.Close()
			}))
			defer upstream.Close()
			defer func() {
				select {
				case <-release:
				default:
					close(release)
				}
			}()

			server := newTestServer(t, upstream.URL)
			payload, err := json.Marshal(map[string]any{
				"model": "gemini-test",
				"messages": []any{map[string]any{
					"role":    "user",
					"content": strings.Repeat("a", 16<<20),
				}},
			})
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", bytes.NewReader(payload))
			rec := httptest.NewRecorder()

			done := make(chan struct{})
			go func() {
				server.ServeHTTP(rec, req)
				close(done)
			}()

			// Request preprocessing (reading, scanning, converting and
			// marshaling a 16 MiB body) is not part of the send timeout and can
			// take longer than 10 seconds under -race. Wait until the redirect
			// target has accepted the replayed request before timing the send
			// phase, so this test measures the per-hop send timer rather than
			// total handler duration.
			select {
			case <-targetStarted:
			case <-time.After(60 * time.Second):
				t.Fatal("redirected request did not reach the target")
			}
			sendPhaseStart := time.Now()
			select {
			case <-done:
				if elapsed := time.Since(sendPhaseStart); elapsed > 5*time.Second {
					t.Fatalf("request took %s after the target started receiving, want the send timeout to cancel it promptly", elapsed)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("request did not exit after redirected send timeout")
			}
			if rec.Code != http.StatusBadGateway {
				t.Fatalf("status = %d, want 502, body = %s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), "send timed out") {
				t.Fatalf("expected send timeout error, body = %s", rec.Body.String())
			}

			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			if !server.waitRequests(ctx) {
				t.Fatal("request resources were not released")
			}
		})
	}
}

func TestSameOriginRedirectLongSSECompletes(t *testing.T) {
	old := upstreamRequestSendTimeout
	upstreamRequestSendTimeout = 200 * time.Millisecond
	defer func() { upstreamRequestSendTimeout = old }()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/retry" {
			_, _ = io.Copy(io.Discard, r.Body)
			http.Redirect(w, r, "/retry", http.StatusTemporaryRedirect)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, chunk := range []string{
			`{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b"}]}}]}`,
			`{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}`,
		} {
			time.Sleep(300 * time.Millisecond)
			_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	out := rec.Body.String()
	for _, want := range []string{`"content":"a"`, `"content":"b"`, `"finish_reason":"stop"`, `"total_tokens":3`, "data: [DONE]"} {
		if !strings.Contains(out, want) {
			t.Fatalf("response missing %q: %s", want, out)
		}
	}
}

func TestSlowDownstreamWriteTimesOut(t *testing.T) {
	old := downstreamWriteIdleTimeout
	downstreamWriteIdleTimeout = 100 * time.Millisecond
	defer func() { downstreamWriteIdleTimeout = old }()

	done := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(done)
		sw := &statusWriter{
			ResponseWriter: w,
			controller:     http.NewResponseController(w),
			writeIdle:      downstreamWriteIdleTimeout,
		}
		chunk := bytes.Repeat([]byte("x"), 1<<20)
		for i := 0; i < 100; i++ {
			if _, err := sw.Write(chunk); err != nil {
				return
			}
			sw.Flush()
		}
	}))
	defer ts.Close()

	conn, err := net.Dial("tcp", strings.TrimPrefix(ts.URL, "http://"))
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()
	_, _ = io.WriteString(conn, "GET / HTTP/1.1\r\nHost: test\r\n\r\n")

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("downstream write did not time out")
	}
}

func TestAggregateLargeImageSpillsAndReturnsFull(t *testing.T) {
	imageData := strings.Repeat("QUJD", 512*1024)
	sent := make(chan struct{})
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk, err := json.Marshal(map[string]any{
			"candidates": []any{map[string]any{
				"content": map[string]any{"parts": []any{map[string]any{
					"inlineData": map[string]any{"mimeType": "image/png", "data": imageData},
				}}},
			}},
		})
		if err != nil {
			t.Errorf("marshal image chunk: %v", err)
			return
		}
		_, _ = io.WriteString(w, "data: "+string(chunk)+"\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		close(sent)
		<-release
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		server.ServeHTTP(rec, req)
		close(done)
	}()

	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("upstream did not send the image event")
	}

	deadline := time.Now().Add(3 * time.Second)
	for {
		entries, err := os.ReadDir(server.store.Dir())
		if err != nil {
			t.Fatalf("ReadDir: %v", err)
		}
		if len(entries) > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("image payload did not spill to a temp file")
		}
		time.Sleep(10 * time.Millisecond)
	}

	close(release)
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("aggregate request did not finish")
	}

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	parts := parseContentParts(t, out.Choices[0].Message.Content)
	if len(parts) != 1 || parts[0].ImageURL == nil {
		t.Fatalf("content parts = %#v", parts)
	}
	wantURL := "data:image/png;base64," + imageData
	if parts[0].ImageURL.URL != wantURL {
		t.Fatal("spilled image did not round-trip exactly")
	}
	assertDirEmpty(t, server.store.Dir())
}

func TestAggregateSingleLargeEventSpillsBeforeCompletion(t *testing.T) {
	content := strings.Repeat("single-event-large-content-", 200000)
	chunk, err := json.Marshal(map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{"parts": []any{map[string]any{"text": content}}},
		}},
	})
	if err != nil {
		t.Fatalf("marshal chunk: %v", err)
	}
	half := len(chunk) / 2
	sent := make(chan struct{})
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, "data: "+string(chunk[:half]))
		if flusher != nil {
			flusher.Flush()
		}
		close(sent)
		<-release
		_, _ = io.WriteString(w, string(chunk[half:])+"\n\n")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		server.ServeHTTP(rec, req)
		close(done)
	}()

	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("upstream did not send the first half of the large event")
	}
	waitForSpillFile(t, server.store.Dir())

	close(release)
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("aggregate request did not finish")
	}

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out convertResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response JSON: %v", err)
	}
	if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content != content {
		t.Fatal("single large event did not round-trip exactly")
	}
	assertDirEmpty(t, server.store.Dir())
}

func TestUnsupportedGeminiPartAcrossPaths(t *testing.T) {
	payload := `{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}}]},"finishReason":"STOP"}]}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: "+payload+"\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, payload)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	for _, tc := range []struct {
		name       string
		body       string
		wantStatus int
	}{
		{"non-stream", `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`, http.StatusBadGateway},
		{"aggregate", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`, http.StatusBadGateway},
		{"stream", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`, http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)
			if rec.Code != tc.wantStatus {
				t.Fatalf("status = %d, want %d, body = %s", rec.Code, tc.wantStatus, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), "executableCode") {
				t.Fatalf("expected an explicit executableCode error, body = %s", rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), `"code":"unsupported_response"`) {
				t.Fatalf("expected unsupported_response error code, body = %s", rec.Body.String())
			}
			if strings.Contains(rec.Body.String(), `"code":"upstream_stream"`) {
				t.Fatalf("unsupported content was reported as a stream failure: %s", rec.Body.String())
			}
			if strings.Contains(rec.Body.String(), `"finish_reason":"stop"`) {
				t.Fatalf("unsupported part was reported as a normal stop: %s", rec.Body.String())
			}
		})
	}
}

func TestDuplicateNullPartAcrossPaths(t *testing.T) {
	payload := `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"AAAA"},"inlineData":null}]},"finishReason":"STOP"}]}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: "+payload+"\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, payload)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	for _, tc := range []struct {
		name       string
		body       string
		wantStatus int
	}{
		{"non-stream", `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`, http.StatusBadGateway},
		{"aggregate", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`, http.StatusBadGateway},
		{"stream", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`, http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)
			if rec.Code != tc.wantStatus {
				t.Fatalf("status = %d, want %d, body = %s", rec.Code, tc.wantStatus, rec.Body.String())
			}
			body := rec.Body.String()
			if !strings.Contains(body, `"code":"unsupported_response"`) {
				t.Fatalf("expected unsupported_response error code, body = %s", body)
			}
			if strings.Contains(body, "data:image/png;base64,AAAA") {
				t.Fatalf("cleared inlineData was still returned, body = %s", body)
			}
			if strings.Contains(body, `"finish_reason":"stop"`) {
				t.Fatalf("cleared inlineData was reported as a normal stop, body = %s", body)
			}
			if tc.name == "stream" && !strings.Contains(body, "data: [DONE]") {
				t.Fatalf("stream error must terminate with [DONE], body = %s", body)
			}
		})
	}
}

func TestInvalidUpstreamJSONStructureAcrossJSONPaths(t *testing.T) {
	paths := []struct {
		name string
		body string
	}{
		{"non-stream", `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`},
		{"aggregate-json-fallback", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"stream-json-fallback", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
	}
	for _, invalid := range []string{`{}`, `null`, `{"candidates":[]}`} {
		for _, tc := range paths {
			t.Run(tc.name+"/"+invalid, func(t *testing.T) {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, invalid)
				}))
				defer upstream.Close()

				server := newTestServer(t, upstream.URL)
				req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
				rec := httptest.NewRecorder()
				server.ServeHTTP(rec, req)

				if rec.Code != http.StatusBadGateway {
					t.Fatalf("status = %d, want 502, body = %s", rec.Code, rec.Body.String())
				}
				if !strings.Contains(rec.Body.String(), "invalid_upstream_response") {
					t.Fatalf("expected invalid_upstream_response, body = %s", rec.Body.String())
				}
				if strings.Contains(rec.Body.String(), `"finish_reason":"stop"`) || strings.Contains(rec.Body.String(), "[DONE]") {
					t.Fatalf("invalid upstream response was reported as a normal completion: %s", rec.Body.String())
				}
			})
		}
	}
}

func TestInvalidUpstreamSSEStructureAcrossStreamPaths(t *testing.T) {
	events := []struct {
		name  string
		event string
	}{
		{"empty-object", `{}`},
		{"null", `null`},
		{"empty-candidates", `{"candidates":[]}`},
		{"empty-candidate", `{"candidates":[{"content":{"parts":[]}}]}`},
		{"only-done", ""},
	}
	paths := []struct {
		name       string
		body       string
		wantStatus int
	}{
		{"aggregate", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`, http.StatusBadGateway},
		{"stream", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`, http.StatusOK},
	}
	for _, invalid := range events {
		for _, tc := range paths {
			t.Run(invalid.name+"/"+tc.name, func(t *testing.T) {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					if invalid.event != "" {
						_, _ = io.WriteString(w, "data: "+invalid.event+"\n\n")
					}
					_, _ = io.WriteString(w, "data: [DONE]\n\n")
				}))
				defer upstream.Close()

				server := newTestServer(t, upstream.URL)
				req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
				rec := httptest.NewRecorder()
				server.ServeHTTP(rec, req)

				if rec.Code != tc.wantStatus {
					t.Fatalf("status = %d, want %d, body = %s", rec.Code, tc.wantStatus, rec.Body.String())
				}
				body := rec.Body.String()
				if !strings.Contains(body, "invalid_upstream_response") {
					t.Fatalf("expected invalid_upstream_response, body = %s", body)
				}
				if strings.Contains(body, `"finish_reason":"stop"`) || strings.Contains(body, `"finish_reason":"tool_calls"`) {
					t.Fatalf("invalid upstream SSE response was reported as success: %s", body)
				}
				if tc.name == "stream" {
					if !strings.Contains(body, "data: [DONE]") {
						t.Fatalf("stream error must terminate with [DONE], body = %s", body)
					}
				} else if strings.Contains(body, "[DONE]") {
					t.Fatalf("aggregate error must not contain a stream terminator: %s", body)
				}
			})
		}
	}
}

func TestSSEUsageOnlyTrailingEventIsAccepted(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":2,\"totalTokenCount\":3}}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	for _, tc := range []struct {
		name string
		body string
	}{
		{"aggregate", `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"stream", `{"model":"gemini-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}

			type usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}
			assertUsage := func(t *testing.T, got *usage) {
				t.Helper()
				if got == nil {
					t.Fatal("usage is missing")
				}
				if got.PromptTokens != 1 || got.CompletionTokens != 2 || got.TotalTokens != 3 {
					t.Fatalf("usage = %#v, want prompt=1 completion=2 total=3", got)
				}
			}

			if tc.name == "aggregate" {
				var out convertResponse
				if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
					t.Fatalf("response JSON: %v, body = %s", err, rec.Body.String())
				}
				if len(out.Choices) == 0 || out.Choices[0].FinishReason != "stop" {
					t.Fatalf("expected a normal stop, body = %s", rec.Body.String())
				}
				var got *usage
				if out.Usage != nil {
					got = &usage{
						PromptTokens:     out.Usage.PromptTokens,
						CompletionTokens: out.Usage.CompletionTokens,
						TotalTokens:      out.Usage.TotalTokens,
					}
				}
				assertUsage(t, got)
				return
			}

			var finishUsage *usage
			for _, line := range strings.Split(rec.Body.String(), "\n") {
				line = strings.TrimSpace(line)
				if !strings.HasPrefix(line, "data:") {
					continue
				}
				payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				if payload == "" || payload == "[DONE]" {
					continue
				}
				var chunk struct {
					Choices []struct {
						FinishReason *string `json:"finish_reason"`
					} `json:"choices"`
					Usage *usage `json:"usage"`
				}
				if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
					t.Fatalf("invalid stream chunk %q: %v", payload, err)
				}
				for _, choice := range chunk.Choices {
					if choice.FinishReason != nil && *choice.FinishReason == "stop" {
						finishUsage = chunk.Usage
					}
				}
			}
			if finishUsage == nil {
				t.Fatalf("expected a finish chunk with usage, body = %s", rec.Body.String())
			}
			assertUsage(t, finishUsage)
		})
	}
}

func TestAggregateOverDefaultSoftLimitReturnsFullAndCleansUp(t *testing.T) {
	const (
		chunks    = 65
		chunkSize = 1 << 20
	)
	alphabet := []byte("abcdefghijklmnopqrstuvwxyz0123456789")
	makeChunk := func(i int) []byte {
		b := make([]byte, chunkSize)
		for j := range b {
			b[j] = alphabet[(i*17+j*7)%len(alphabet)]
		}
		return b
	}

	wantHash := sha256.New()
	for i := 0; i < chunks; i++ {
		_, _ = wantHash.Write(makeChunk(i))
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i := 0; i < chunks; i++ {
			chunk, err := json.Marshal(map[string]any{
				"candidates": []any{map[string]any{
					"content": map[string]any{"parts": []any{map[string]any{"text": string(makeChunk(i))}}},
				}},
			})
			if err != nil {
				t.Errorf("marshal chunk %d: %v", i, err)
				return
			}
			_, _ = io.WriteString(w, "data: "+string(chunk)+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServerWithCacheMax(t, upstream.URL, config.DefaultSSECacheMaxBytes)
	body := `{"model":"gemini-test","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	raw := rec.Body.Bytes()
	marker := []byte(`"content":"`)
	start := bytes.Index(raw, marker)
	if start < 0 {
		t.Fatal("response has no content field")
	}
	start += len(marker)
	end := bytes.IndexByte(raw[start:], '"')
	if end < 0 {
		t.Fatal("response content is not terminated")
	}
	content := raw[start : start+end]
	wantSize := chunks * chunkSize
	if len(content) != wantSize {
		t.Fatalf("content length = %d, want %d", len(content), wantSize)
	}
	if wantSize <= int(config.DefaultSSECacheMaxBytes) {
		t.Fatalf("test content %d must exceed the default soft limit %d", wantSize, config.DefaultSSECacheMaxBytes)
	}
	gotHash := sha256.Sum256(content)
	if !bytes.Equal(gotHash[:], wantHash.Sum(nil)) {
		t.Fatal("content over the default soft limit did not round-trip exactly")
	}
	assertDirEmpty(t, server.store.Dir())
}

func TestAggregateImageMetadataBudgetIsEnforced(t *testing.T) {
	parts := make([]map[string]any, maxAggregatedImages+1)
	for i := range parts {
		parts[i] = map[string]any{
			"inlineData": map[string]any{"mimeType": "image/png", "data": "A"},
		}
	}
	event, err := json.Marshal(map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"parts": parts},
			"finishReason": "STOP",
		}},
	})
	if err != nil {
		t.Fatalf("marshal event: %v", err)
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: "+string(event)+"\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}],"aggregate_sse":true}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "image metadata exceeds limit") {
		t.Fatalf("expected image metadata limit error, got: %s", rec.Body.String())
	}
	assertDirEmpty(t, server.store.Dir())
}

func TestAggregateImageMetadataByteBudgetIsEnforced(t *testing.T) {
	mimeType := "image/" + strings.Repeat("a", 249)
	const imageCount = 3800
	if imageCount >= maxAggregatedImages {
		t.Fatal("test must stay below the image count limit")
	}
	if imageCount*(len(mimeType)+3*8) <= maxAggregatedImageMetadataBytes {
		t.Fatal("test images must exceed the metadata byte limit")
	}
	parts := make([]map[string]any, imageCount)
	for i := range parts {
		parts[i] = map[string]any{
			"inlineData": map[string]any{"mimeType": mimeType, "data": "A"},
		}
	}
	event, err := json.Marshal(map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"parts": parts},
			"finishReason": "STOP",
		}},
	})
	if err != nil {
		t.Fatalf("marshal event: %v", err)
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: "+string(event)+"\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}],"aggregate_sse":true}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "image metadata exceeds limit") {
		t.Fatalf("expected image metadata byte limit error, got: %s", rec.Body.String())
	}
	assertDirEmpty(t, server.store.Dir())
}

func TestAggregateToolCallBudgetsAreEnforced(t *testing.T) {
	for _, tc := range []struct {
		name  string
		parts []map[string]any
	}{
		{
			name: "count",
			parts: func() []map[string]any {
				parts := make([]map[string]any, maxAggregatedToolCalls+1)
				for i := range parts {
					parts[i] = map[string]any{
						"functionCall": map[string]any{"name": "f", "args": map[string]any{}},
					}
				}
				return parts
			}(),
		},
		{
			name: "bytes-including-id-and-signature",
			parts: []map[string]any{{
				"functionCall": map[string]any{
					"name": "f",
					"id":   strings.Repeat("i", 8<<20),
					"args": map[string]any{},
				},
				"thoughtSignature": strings.Repeat("s", 8<<20),
			}},
		},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			event, err := json.Marshal(map[string]any{
				"candidates": []any{map[string]any{
					"content":      map[string]any{"parts": tc.parts},
					"finishReason": "STOP",
				}},
			})
			if err != nil {
				t.Fatalf("marshal event: %v", err)
			}

			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				// Spill a small text prefix so the limit failure also has a
				// temporary file to clean up.
				_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"prefix\"}]}}]}\n\n")
				_, _ = io.WriteString(w, "data: "+string(event)+"\n\n")
			}))
			defer upstream.Close()

			server := newTestServer(t, upstream.URL)
			body := `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}],"aggregate_sse":true}`
			req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusBadGateway {
				t.Fatalf("status = %d, want 502; body = %s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), "aggregated tool call data exceeds limit") {
				t.Fatalf("expected tool call limit error, got: %s", rec.Body.String())
			}
			assertDirEmpty(t, server.store.Dir())
		})
	}
}

func TestEncodedPercentDoesNotRouteToWrongProvider(t *testing.T) {
	called := false
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}]}`
	// The HTTP layer decodes %25 to a literal %, leaving "%6eewapi1". A second
	// decode would turn that into "newapi1" and send credentials to the wrong
	// provider; Resolve must not do that.
	req := httptest.NewRequest(http.MethodPost, "/%256eewapi1/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer should-not-be-forwarded")
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body = %s", rec.Code, rec.Body.String())
	}
	if called {
		t.Fatal("request was forwarded to the wrong provider after a second URL decode")
	}
}

func TestModelsRejectsOversizedPageToken(t *testing.T) {
	token := strings.Repeat("t", maxModelPageTokenBytes+1)
	var requests int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"models":[],"nextPageToken":"`+token+`"}`)
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	req := httptest.NewRequest(http.MethodGet, "/newapi1/v1/models", nil)
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502, body = %s", rec.Code, rec.Body.String())
	}
	var envelope errorEnvelope
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("response is not JSON: %v, body = %s", err, rec.Body.String())
	}
	if envelope.Error.Code != "models_page_token_too_large" {
		t.Fatalf("error code = %q, want models_page_token_too_large", envelope.Error.Code)
	}
	if requests != 1 {
		t.Fatalf("upstream requests = %d, want 1", requests)
	}
}

func TestAggregateWidePartsArrayIsRejectedAndCleansUp(t *testing.T) {
	const wideParts = 20000
	event, err := json.Marshal(map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{"parts": make([]map[string]any, wideParts)},
		}},
	})
	if err != nil {
		t.Fatalf("marshal event: %v", err)
	}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"prefix\"}]}}]}\n\n")
		_, _ = io.WriteString(w, "data: "+string(event)+"\n\n")
	}))
	defer upstream.Close()

	server := newTestServer(t, upstream.URL)
	body := `{"model":"gemini-test","messages":[{"role":"user","content":"hi"}],"aggregate_sse":true}`
	req := httptest.NewRequest(http.MethodPost, "/newapi1/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "too many parts") {
		t.Fatalf("expected parts count limit error, got: %s", rec.Body.String())
	}
	assertDirEmpty(t, server.store.Dir())
}
