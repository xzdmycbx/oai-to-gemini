package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptrace"
	"strings"
	"sync"
	"time"

	"github.com/xzdmycbx/oai-to-gemini/internal/aggregate"
	"github.com/xzdmycbx/oai-to-gemini/internal/config"
	"github.com/xzdmycbx/oai-to-gemini/internal/router"
)

const (
	maxRequestBodyBytes  = 64 << 20
	maxUpstreamJSONBytes = 128 << 20
	maxModelsJSONBytes   = 16 << 20
)

// These are idle timeouts, not total request deadlines, so a healthy long
// stream can continue as long as data keeps flowing. The upstream send timeout
// only covers writing the request (including its body) and is cleared once the
// transport reports that the request has been fully written.
var (
	upstreamRequestSendTimeout  = 60 * time.Second
	upstreamResponseIdleTimeout = 60 * time.Second
	downstreamWriteIdleTimeout  = 60 * time.Second
)

type Server struct {
	cfg    *config.Config
	client *http.Client
	store  *aggregate.Store
	logger *log.Logger

	forceCtx    context.Context
	forceCancel context.CancelFunc

	mu       sync.Mutex
	draining bool
	requests sync.WaitGroup
}

func NewServer(cfg *config.Config, logger *log.Logger) (*Server, error) {
	if logger == nil {
		logger = log.Default()
	}
	store, err := aggregate.NewStore(cfg.SSECacheDir, cfg.SSESpillThresholdBytes, cfg.SSECacheMaxBytes, logger)
	if err != nil {
		return nil, err
	}
	dialer := &net.Dialer{
		Timeout:   10 * time.Second,
		KeepAlive: 30 * time.Second,
	}
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           dialer.DialContext,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   20,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
		ExpectContinueTimeout: time.Second,
	}
	forceCtx, forceCancel := context.WithCancel(context.Background())
	return &Server{
		cfg: cfg,
		client: &http.Client{
			Transport:     transport,
			CheckRedirect: checkRedirect,
		},
		store:       store,
		logger:      logger,
		forceCtx:    forceCtx,
		forceCancel: forceCancel,
	}, nil
}

// checkRedirect allows only same-origin redirects and never downgrades HTTPS.
// Cross-origin redirects are not followed at all, so credentials such as
// x-goog-api-key can never be forwarded to another host.
func checkRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= 10 {
		return fmt.Errorf("stopped after 10 redirects")
	}
	if len(via) == 0 {
		return nil
	}
	prev := via[len(via)-1].URL
	next := req.URL
	if !strings.EqualFold(prev.Scheme, next.Scheme) || !strings.EqualFold(prev.Host, next.Host) {
		return http.ErrUseLastResponse
	}
	if prev.Scheme == "https" && next.Scheme != "https" {
		return http.ErrUseLastResponse
	}
	return nil
}

// BeginShutdown prevents new requests from being admitted. Existing requests
// keep running until Shutdown's context expires or CancelActiveRequests is
// called.
func (s *Server) BeginShutdown() {
	s.mu.Lock()
	s.draining = true
	s.mu.Unlock()
}

// CancelActiveRequests cancels the request contexts of all in-flight handlers.
// It is used after a graceful shutdown deadline expires so handlers stop using
// the aggregate store and release their buffers.
func (s *Server) CancelActiveRequests() {
	s.forceCancel()
}

func (s *Server) beginRequest() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.draining {
		return false
	}
	s.requests.Add(1)
	return true
}

func (s *Server) waitRequests(ctx context.Context) bool {
	done := make(chan struct{})
	go func() {
		s.requests.Wait()
		close(done)
	}()
	select {
	case <-done:
		return true
	case <-ctx.Done():
		return false
	}
}

// Shutdown stops admitting new requests and waits for active handlers until
// ctx is done. If the deadline expires, it cancels the remaining requests and
// waits for their handlers to release aggregate buffers before closing the
// store. The instance cache directory is never removed while a request still
// owns a buffer.
func (s *Server) Shutdown(ctx context.Context) error {
	s.BeginShutdown()
	if !s.waitRequests(ctx) {
		s.CancelActiveRequests()
		s.requests.Wait()
	}
	return s.store.Close()
}

// Close is the forced lifecycle path used by tests and as a final safety net.
// It cancels active requests, waits for handlers to release their resources,
// and only then closes the store.
func (s *Server) Close() error {
	s.BeginShutdown()
	s.CancelActiveRequests()
	s.requests.Wait()
	return s.store.Close()
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !s.beginRequest() {
		w.Header().Set("Connection", "close")
		writeOpenAIError(w, http.StatusServiceUnavailable, "server is shutting down", "server_error", "shutting_down")
		return
	}
	defer s.requests.Done()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	stop := context.AfterFunc(s.forceCtx, cancel)
	defer stop()
	r = r.WithContext(ctx)

	if r.URL.Path == "/healthz" {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			writeOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error", "method_not_allowed")
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
		return
	}

	sw := &statusWriter{
		ResponseWriter: w,
		controller:     http.NewResponseController(w),
		writeIdle:      downstreamWriteIdleTimeout,
	}
	start := time.Now()
	s.route(sw, r)
	status := sw.status
	if status == 0 {
		status = http.StatusOK
	}
	// Deliberately log only the path. Query strings may contain ?key=.
	s.logger.Printf("%s %s -> %d (%s)", r.Method, r.URL.Path, status, time.Since(start).Round(time.Millisecond))
}

func (s *Server) route(w http.ResponseWriter, r *http.Request) {
	match, ok := router.Resolve(r.URL.Path, s.cfg.Providers)
	if !ok {
		writeOpenAIError(w, http.StatusNotFound,
			fmt.Sprintf("unknown provider in path %q", r.URL.Path),
			"invalid_request_error", "provider_not_found")
		return
	}

	switch {
	case r.Method == http.MethodPost && (match.Path == "/v1/chat/completions" || match.Path == "/chat/completions"):
		s.handleChat(w, r, match.Provider)
	case r.Method == http.MethodGet && (match.Path == "/v1/models" || match.Path == "/models"):
		s.handleModels(w, r, match.Provider)
	default:
		writeOpenAIError(w, http.StatusNotFound,
			fmt.Sprintf("unsupported route %s %s", r.Method, match.Path),
			"invalid_request_error", "not_found")
	}
}

type statusWriter struct {
	http.ResponseWriter
	status     int
	controller *http.ResponseController
	writeIdle  time.Duration
}

func (w *statusWriter) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	w.refreshWriteDeadline()
	return w.ResponseWriter.Write(p)
}

func (w *statusWriter) Flush() {
	w.refreshWriteDeadline()
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (w *statusWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

func (w *statusWriter) refreshWriteDeadline() {
	if w.controller != nil && w.writeIdle > 0 {
		_ = w.controller.SetWriteDeadline(time.Now().Add(w.writeIdle))
	}
}

func (s *Server) doUpstream(ctx context.Context, req *http.Request) (*http.Response, error) {
	ctx, cancel := context.WithCancel(ctx)
	req = req.WithContext(ctx)

	// Bound the time spent sending each request. Dial and response-header
	// timeouts do not cover a connection that accepts TCP but stops reading
	// the request body, so a stalled send could otherwise hold the handler
	// forever. GetConn fires once per actual request, including same-origin
	// 307/308 redirects that replay the body; WroteRequest fires after that
	// hop has been written. Starting and clearing the timer per hop keeps
	// redirect replays bounded without imposing a deadline on long SSE
	// responses.
	send := newSendTimeout(upstreamRequestSendTimeout, cancel)
	trace := &httptrace.ClientTrace{
		GetConn: func(string) {
			send.start()
		},
		WroteRequest: func(httptrace.WroteRequestInfo) {
			send.stopSend()
		},
	}
	req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))

	resp, err := s.client.Do(req)
	if err != nil {
		send.stop()
		cancel()
		if send.didTimeout() {
			return nil, fmt.Errorf("upstream request send timed out after %s: %w", upstreamRequestSendTimeout, err)
		}
		return nil, err
	}
	resp.Body = newIdleTimeoutBody(resp.Body, cancel, upstreamResponseIdleTimeout)
	return resp, nil
}

type sendTimeout struct {
	mu         sync.Mutex
	timeout    time.Duration
	cancel     context.CancelFunc
	timer      *time.Timer
	generation uint64
	timedOut   bool
	stopped    bool
}

func newSendTimeout(timeout time.Duration, cancel context.CancelFunc) *sendTimeout {
	return &sendTimeout{timeout: timeout, cancel: cancel}
}

// start begins the send timer for one request hop. A redirect that reuses the
// client creates a new hop, so the timer is restarted even if the previous hop
// already finished writing.
func (s *sendTimeout) start() {
	if s.timeout <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped || s.timedOut {
		return
	}
	s.generation++
	generation := s.generation
	if s.timer != nil {
		s.timer.Stop()
	}
	s.timer = time.AfterFunc(s.timeout, func() {
		s.mu.Lock()
		if s.stopped || s.timedOut || s.generation != generation {
			s.mu.Unlock()
			return
		}
		s.timedOut = true
		s.mu.Unlock()
		s.cancel()
	})
}

// stopSend clears the timer after the current hop has been written. It does
// not prevent a later hop from starting a new timer.
func (s *sendTimeout) stopSend() {
	if s.timeout <= 0 {
		return
	}
	s.mu.Lock()
	s.generation++
	if s.timer != nil {
		s.timer.Stop()
	}
	s.mu.Unlock()
}

func (s *sendTimeout) stop() {
	if s.timeout <= 0 {
		return
	}
	s.mu.Lock()
	s.stopped = true
	s.generation++
	if s.timer != nil {
		s.timer.Stop()
	}
	s.mu.Unlock()
}

func (s *sendTimeout) didTimeout() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.timedOut
}

type idleTimeoutBody struct {
	body       io.ReadCloser
	cancel     context.CancelFunc
	timeout    time.Duration
	mu         sync.Mutex
	timer      *time.Timer
	generation uint64
	active     bool
	timedOut   bool
	closed     bool
}

func newIdleTimeoutBody(body io.ReadCloser, cancel context.CancelFunc, timeout time.Duration) *idleTimeoutBody {
	return &idleTimeoutBody{body: body, cancel: cancel, timeout: timeout}
}

func (b *idleTimeoutBody) Read(p []byte) (int, error) {
	b.mu.Lock()
	if b.timeout > 0 && !b.timedOut && !b.closed {
		b.generation++
		generation := b.generation
		b.active = true
		if b.timer != nil {
			b.timer.Stop()
		}
		b.timer = time.AfterFunc(b.timeout, func() {
			b.mu.Lock()
			if !b.active || b.closed || b.timedOut || b.generation != generation {
				b.mu.Unlock()
				return
			}
			b.timedOut = true
			b.active = false
			b.mu.Unlock()
			b.cancel()
		})
	}
	b.mu.Unlock()

	n, err := b.body.Read(p)

	b.mu.Lock()
	if b.timeout > 0 {
		b.generation++
		b.active = false
		if b.timer != nil {
			b.timer.Stop()
		}
	}
	timedOut := b.timedOut
	b.mu.Unlock()

	if err != nil && timedOut {
		return n, fmt.Errorf("upstream response idle timeout after %s: %w", b.timeout, err)
	}
	return n, err
}

func (b *idleTimeoutBody) Close() error {
	b.mu.Lock()
	b.closed = true
	b.active = false
	if b.timer != nil {
		b.timer.Stop()
	}
	b.mu.Unlock()
	b.cancel()
	return b.body.Close()
}

func readAllLimited(r io.Reader, limit int64) ([]byte, bool, error) {
	data, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, false, err
	}
	if int64(len(data)) > limit {
		return nil, true, nil
	}
	return data, false, nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
