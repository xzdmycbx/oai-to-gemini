package proxy

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/xzdmycbx/oai-to-gemini/internal/config"
)

func (s *Server) newUpstreamRequest(
	ctx context.Context,
	incoming *http.Request,
	provider config.Provider,
	method, path string,
	body []byte,
	extraQuery url.Values,
) (*http.Request, error) {
	endpoint := strings.TrimRight(provider.APIBase, "/") + path
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, fmt.Errorf("parse upstream url: %w", err)
	}

	merged := make(url.Values)
	for key, values := range incoming.URL.Query() {
		if key == "aggregate_sse" {
			continue
		}
		merged[key] = append([]string(nil), values...)
	}
	for key, values := range extraQuery {
		merged[key] = append([]string(nil), values...)
	}
	u.RawQuery = merged.Encode()

	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, u.String(), reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	// Forward credentials exactly as received. They are never logged.
	if v := incoming.Header.Get("Authorization"); v != "" {
		req.Header.Set("Authorization", v)
	}
	if v := incoming.Header.Get("X-Goog-Api-Key"); v != "" {
		req.Header.Set("X-Goog-Api-Key", v)
	}
	if v := incoming.Header.Get("Accept"); v != "" {
		req.Header.Set("Accept", v)
	}
	return req, nil
}
