package proxy

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/xzdmycbx/oai-to-gemini/internal/convert"
)

type errorEnvelope struct {
	Error errorBody `json:"error"`
}

type errorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

func writeOpenAIError(w http.ResponseWriter, status int, message, typ string, code any) {
	if typ == "" {
		typ = "upstream_error"
	}
	writeJSON(w, status, errorEnvelope{Error: errorBody{
		Message: message,
		Type:    typ,
		Code:    code,
	}})
}

// writeUpstreamError preserves the upstream status code and wraps the upstream
// error body in an OpenAI-style envelope. The body is never written to logs.
func writeUpstreamError(w http.ResponseWriter, resp *http.Response) {
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	status := resp.StatusCode
	message := strings.TrimSpace(string(body))
	typ := "upstream_error"
	var code any = status

	var envelope struct {
		Error struct {
			Code    any    `json:"code"`
			Message string `json:"message"`
			Status  string `json:"status"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err == nil {
		if envelope.Error.Message != "" {
			message = envelope.Error.Message
		}
		if envelope.Error.Type != "" {
			typ = envelope.Error.Type
		} else if envelope.Error.Status != "" {
			typ = strings.ToLower(envelope.Error.Status)
		}
		if envelope.Error.Code != nil {
			code = envelope.Error.Code
		} else if envelope.Error.Status != "" {
			code = envelope.Error.Status
		}
	}

	if message == "" {
		message = http.StatusText(status)
	}
	writeOpenAIError(w, status, message, typ, code)
}

func embeddedGeminiError(raw json.RawMessage) string {
	if len(raw) == 0 || strings.TrimSpace(string(raw)) == "null" {
		return ""
	}
	// GeminiResponse.Error already contains the inner error object, not an
	// {"error": ...} envelope.
	var errObj struct {
		Message string          `json:"message"`
		Status  string          `json:"status"`
		Code    json.RawMessage `json:"code"`
	}
	if err := json.Unmarshal(raw, &errObj); err == nil {
		if errObj.Message != "" {
			return errObj.Message
		}
		if errObj.Status != "" {
			return errObj.Status
		}
	}
	return strings.TrimSpace(string(raw))
}

func writeConversionError(w http.ResponseWriter, err error) {
	writeOpenAIError(w, http.StatusBadGateway, err.Error(), "upstream_error", conversionErrorCode(err))
}

func conversionErrorCode(err error) string {
	if abnormalFinish(err) {
		return "abnormal_finish"
	}
	var invalid *convert.InvalidUpstreamResponseError
	if errors.As(err, &invalid) {
		return "invalid_upstream_response"
	}
	var unsupported *convert.UnsupportedContentError
	if errors.As(err, &unsupported) {
		return "unsupported_response"
	}
	return "unsupported_response"
}

// streamErrorCode keeps transport/protocol stream failures distinct from
// conversion failures. Unsupported content must use the same code as the
// non-stream path.
func streamErrorCode(err error) string {
	if abnormalFinish(err) {
		return "abnormal_finish"
	}
	var invalid *convert.InvalidUpstreamResponseError
	if errors.As(err, &invalid) {
		return "invalid_upstream_response"
	}
	var unsupported *convert.UnsupportedContentError
	if errors.As(err, &unsupported) {
		return "unsupported_response"
	}
	return "upstream_stream"
}

func abnormalFinish(err error) bool {
	var abnormal *convert.AbnormalFinishError
	return errors.As(err, &abnormal)
}
