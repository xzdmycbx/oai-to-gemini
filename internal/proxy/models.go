package proxy

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/xzdmycbx/oai-to-gemini/internal/config"
)

const (
	maxModelPages              = 100
	maxModels                  = 100000
	maxModelNameBytes          = 8 << 10
	maxModelsTotalBytes        = 2 << 20
	maxModelPageTokenBytes     = 8 << 10
	maxModelPageTokensTotalLen = 1 << 20
)

var errModelListTooLarge = errors.New("upstream model list is too large")

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request, provider config.Provider) {
	var models []struct {
		Name string `json:"name"`
	}
	seenTokens := make(map[[sha256.Size]byte]struct{})
	pageToken := ""
	var totalNameBytes int64
	var totalTokenBytes int64

	for page := 0; ; page++ {
		if page >= maxModelPages {
			writeOpenAIError(w, http.StatusBadGateway, "upstream model list has too many pages", "upstream_error", "models_too_many_pages")
			return
		}

		var extraQuery url.Values
		if pageToken != "" {
			extraQuery = url.Values{"pageToken": []string{pageToken}}
		}
		upstream, err := s.newUpstreamRequest(r.Context(), r, provider, http.MethodGet, "/models", nil, extraQuery)
		if err != nil {
			writeOpenAIError(w, http.StatusInternalServerError, "failed to build upstream request", "server_error", "upstream_request")
			return
		}

		resp, err := s.doUpstream(r.Context(), upstream)
		if err != nil {
			writeOpenAIError(w, http.StatusBadGateway, "upstream request failed: "+err.Error(), "upstream_error", "upstream_unreachable")
			return
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			writeUpstreamError(w, resp)
			return
		}

		data, tooLarge, err := readAllLimited(resp.Body, maxModelsJSONBytes)
		_ = resp.Body.Close()
		if err != nil {
			writeOpenAIError(w, http.StatusBadGateway, "failed to read upstream response", "upstream_error", "upstream_read")
			return
		}
		if tooLarge {
			writeOpenAIError(w, http.StatusBadGateway, "upstream response too large", "upstream_error", "upstream_response_too_large")
			return
		}

		// Decode only the fields this handler reads. Unknown top-level fields
		// must not expand into map entries before the model list budgets run.
		var payload *struct {
			Models        json.RawMessage `json:"models"`
			Data          json.RawMessage `json:"data"`
			Error         json.RawMessage `json:"error"`
			NextPageToken json.RawMessage `json:"nextPageToken"`
		}
		if err := json.Unmarshal(data, &payload); err != nil || payload == nil {
			writeOpenAIError(w, http.StatusBadGateway, "invalid upstream response", "upstream_error", "invalid_upstream_json")
			return
		}

		// Some upstreams report failures inside an HTTP 200 response.
		if msg := embeddedGeminiError(payload.Error); msg != "" {
			writeOpenAIError(w, http.StatusBadGateway, msg, "upstream_error", "upstream_error")
			return
		}

		// Already an OpenAI-style list: validate it and pass the first page
		// through unchanged.
		if payload.Data != nil {
			if _, _, err := scanModelArray(payload.Data, maxModels, maxModelsTotalBytes, parseOpenAIModelEntry); err != nil {
				if errors.Is(err, errModelListTooLarge) {
					writeOpenAIError(w, http.StatusBadGateway, errModelListTooLarge.Error(), "upstream_error", "models_too_large")
					return
				}
				writeOpenAIError(w, http.StatusBadGateway, "invalid OpenAI-style model list: "+err.Error(), "upstream_error", "invalid_upstream_json")
				return
			}
			if page == 0 {
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(data)
				return
			}
			writeOpenAIError(w, http.StatusBadGateway, "unexpected OpenAI-style model list on later page", "upstream_error", "invalid_upstream_json")
			return
		}

		if payload.Models == nil {
			writeOpenAIError(w, http.StatusBadGateway, "invalid upstream model list", "upstream_error", "invalid_upstream_json")
			return
		}
		var pageModels []struct {
			Name string `json:"name"`
		}
		_, pageNameBytes, err := scanModelArray(payload.Models, maxModels-len(models), maxModelsTotalBytes-totalNameBytes, func(i int, item json.RawMessage) (int, error) {
			name, err := parseGeminiModelEntry(i, item)
			if err != nil {
				return 0, err
			}
			pageModels = append(pageModels, struct {
				Name string `json:"name"`
			}{Name: name})
			return len(name), nil
		})
		if err != nil {
			if errors.Is(err, errModelListTooLarge) {
				writeOpenAIError(w, http.StatusBadGateway, errModelListTooLarge.Error(), "upstream_error", "models_too_large")
				return
			}
			writeOpenAIError(w, http.StatusBadGateway, "invalid upstream model list: "+err.Error(), "upstream_error", "invalid_upstream_json")
			return
		}

		totalNameBytes += pageNameBytes
		models = append(models, pageModels...)

		nextPageToken := ""
		if payload.NextPageToken != nil {
			if err := json.Unmarshal(payload.NextPageToken, &nextPageToken); err != nil {
				writeOpenAIError(w, http.StatusBadGateway, "invalid upstream model list", "upstream_error", "invalid_upstream_json")
				return
			}
		}
		if nextPageToken == "" {
			break
		}
		if len(nextPageToken) > maxModelPageTokenBytes {
			writeOpenAIError(w, http.StatusBadGateway, "upstream model page token is too large", "upstream_error", "models_page_token_too_large")
			return
		}
		if totalTokenBytes+int64(len(nextPageToken)) > maxModelPageTokensTotalLen {
			writeOpenAIError(w, http.StatusBadGateway, "upstream model pagination state is too large", "upstream_error", "models_pagination_state_too_large")
			return
		}
		tokenDigest := sha256.Sum256([]byte(nextPageToken))
		if _, seen := seenTokens[tokenDigest]; seen {
			writeOpenAIError(w, http.StatusBadGateway, "upstream model pagination loop", "upstream_error", "models_pagination_loop")
			return
		}
		seenTokens[tokenDigest] = struct{}{}
		totalTokenBytes += int64(len(nextPageToken))
		pageToken = nextPageToken
	}

	now := time.Now().Unix()
	type modelEntry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	out := struct {
		Object string       `json:"object"`
		Data   []modelEntry `json:"data"`
	}{
		Object: "list",
		Data:   make([]modelEntry, 0, len(models)),
	}
	for _, m := range models {
		id := strings.TrimPrefix(strings.TrimSpace(m.Name), "models/")
		if id == "" {
			continue
		}
		out.Data = append(out.Data, modelEntry{
			ID:      id,
			Object:  "model",
			Created: now,
			OwnedBy: "upstream",
		})
	}
	writeJSON(w, http.StatusOK, out)
}

func scanModelArray(raw json.RawMessage, maxItems int, maxBytes int64, handle func(index int, item json.RawMessage) (int, error)) (int, int64, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return 0, 0, fmt.Errorf("model list is not an array")
	}

	dec := json.NewDecoder(bytes.NewReader(trimmed))
	token, err := dec.Token()
	if err != nil {
		return 0, 0, err
	}
	if delim, ok := token.(json.Delim); !ok || delim != '[' {
		return 0, 0, fmt.Errorf("model list is not an array")
	}

	var count int
	var totalBytes int64
	for dec.More() {
		if count >= maxItems {
			return 0, 0, errModelListTooLarge
		}
		var item json.RawMessage
		if err := dec.Decode(&item); err != nil {
			return 0, 0, err
		}
		identifierBytes, err := handle(count, item)
		if err != nil {
			return 0, 0, err
		}
		count++
		totalBytes += int64(identifierBytes)
		if totalBytes > maxBytes {
			return 0, 0, errModelListTooLarge
		}
	}

	if _, err := dec.Token(); err != nil {
		return 0, 0, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return 0, 0, fmt.Errorf("unexpected trailing data after model list")
	}
	return count, totalBytes, nil
}

func parseOpenAIModelEntry(i int, item json.RawMessage) (int, error) {
	trimmed := bytes.TrimSpace(item)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return 0, fmt.Errorf("entry %d is not an object", i)
	}
	var entry struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(trimmed, &entry); err != nil {
		return 0, fmt.Errorf("entry %d is invalid: %w", i, err)
	}
	if strings.TrimSpace(entry.ID) == "" {
		return 0, fmt.Errorf("entry %d has no model id", i)
	}
	if len(entry.ID) > maxModelNameBytes {
		return 0, fmt.Errorf("entry %d has an oversized model id", i)
	}
	return len(entry.ID), nil
}

func parseGeminiModelEntry(i int, item json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(item)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return "", fmt.Errorf("entry %d is not an object", i)
	}
	var entry struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(trimmed, &entry); err != nil {
		return "", fmt.Errorf("entry %d is invalid: %w", i, err)
	}
	name := strings.TrimSpace(entry.Name)
	if name == "" {
		return "", fmt.Errorf("entry %d has no model name", i)
	}
	if strings.TrimPrefix(name, "models/") == "" {
		return "", fmt.Errorf("entry %d has an empty model name", i)
	}
	if len(entry.Name) > maxModelNameBytes {
		return "", fmt.Errorf("entry %d has an oversized model name", i)
	}
	return entry.Name, nil
}
