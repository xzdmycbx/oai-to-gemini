package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

// parseAggregateSSE applies the documented priority:
// request body field > ?aggregate_sse= > X-Aggregate-SSE header.
// Presence, not non-emptiness, decides which layer wins. An explicitly empty
// query value means false and does not fall through to the header.
func parseAggregateSSE(raw map[string]json.RawMessage, query url.Values, header http.Header) (bool, error) {
	if v, ok := raw["aggregate_sse"]; ok {
		if len(v) == 0 || string(v) == "null" {
			return false, nil
		}
		var b bool
		if err := json.Unmarshal(v, &b); err == nil {
			return b, nil
		}
		var s string
		if err := json.Unmarshal(v, &s); err == nil {
			if s == "" {
				return false, nil
			}
			parsed, err := strconv.ParseBool(s)
			if err != nil {
				return false, fmt.Errorf("invalid aggregate_sse value %q", s)
			}
			return parsed, nil
		}
		return false, fmt.Errorf("aggregate_sse must be a boolean")
	}

	if values, ok := query["aggregate_sse"]; ok {
		if len(values) == 0 || values[0] == "" {
			return false, nil
		}
		parsed, err := strconv.ParseBool(values[0])
		if err != nil {
			return false, fmt.Errorf("invalid aggregate_sse query value %q", values[0])
		}
		return parsed, nil
	}

	if v := header.Get("X-Aggregate-SSE"); v != "" {
		parsed, err := strconv.ParseBool(v)
		if err != nil {
			return false, fmt.Errorf("invalid X-Aggregate-SSE value %q", v)
		}
		return parsed, nil
	}
	return false, nil
}
