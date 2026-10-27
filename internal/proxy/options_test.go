package proxy

import (
	"encoding/json"
	"net/http"
	"net/url"
	"testing"
)

func TestParseAggregateSSEPriority(t *testing.T) {
	t.Run("body true overrides query and header false", func(t *testing.T) {
		raw := map[string]json.RawMessage{"aggregate_sse": json.RawMessage("true")}
		got, err := parseAggregateSSE(raw, url.Values{"aggregate_sse": {"false"}}, http.Header{"X-Aggregate-Sse": {"false"}})
		if err != nil || !got {
			t.Fatalf("got=%v err=%v", got, err)
		}
	})

	t.Run("body false overrides query true", func(t *testing.T) {
		raw := map[string]json.RawMessage{"aggregate_sse": json.RawMessage("false")}
		got, err := parseAggregateSSE(raw, url.Values{"aggregate_sse": {"true"}}, http.Header{"X-Aggregate-Sse": {"true"}})
		if err != nil || got {
			t.Fatalf("got=%v err=%v", got, err)
		}
	})

	t.Run("query overrides header", func(t *testing.T) {
		got, err := parseAggregateSSE(nil, url.Values{"aggregate_sse": {"true"}}, http.Header{"X-Aggregate-Sse": {"false"}})
		if err != nil || !got {
			t.Fatalf("got=%v err=%v", got, err)
		}
	})

	t.Run("empty query overrides header", func(t *testing.T) {
		got, err := parseAggregateSSE(nil, url.Values{"aggregate_sse": {""}}, http.Header{"X-Aggregate-Sse": {"true"}})
		if err != nil || got {
			t.Fatalf("got=%v err=%v, want false without falling back to header", got, err)
		}
	})

	t.Run("header fallback", func(t *testing.T) {
		got, err := parseAggregateSSE(nil, url.Values{}, http.Header{"X-Aggregate-Sse": {"true"}})
		if err != nil || !got {
			t.Fatalf("got=%v err=%v", got, err)
		}
	})

	t.Run("absent", func(t *testing.T) {
		got, err := parseAggregateSSE(nil, url.Values{}, http.Header{})
		if err != nil || got {
			t.Fatalf("got=%v err=%v", got, err)
		}
	})

	t.Run("invalid", func(t *testing.T) {
		if _, err := parseAggregateSSE(nil, url.Values{"aggregate_sse": {"maybe"}}, http.Header{}); err == nil {
			t.Fatal("expected error")
		}
	})
}
