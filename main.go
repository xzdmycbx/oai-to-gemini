package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/xzdmycbx/oai-to-gemini/internal/config"
	"github.com/xzdmycbx/oai-to-gemini/internal/proxy"
)

func main() {
	var configPath string
	var listen string
	flag.StringVar(&configPath, "config", "", "path to config.yaml or config.json (default: CONFIG_PATH, config.yaml, then config.json)")
	flag.StringVar(&listen, "listen", "", "listen address (default: config listen or :8080)")
	flag.Parse()

	logger := log.New(os.Stdout, "oai-to-gemini ", log.LstdFlags|log.LUTC)

	cfg, err := config.Load(configPath)
	if err != nil {
		logger.Fatalf("load config: %v", err)
	}
	if listen != "" {
		cfg.Listen = listen
	}
	if cfg.Listen == "" {
		cfg.Listen = ":8080"
	}

	server, err := proxy.NewServer(cfg, logger)
	if err != nil {
		logger.Fatalf("create server: %v", err)
	}
	defer func() { _ = server.Close() }()

	httpServer := &http.Server{
		Addr:              cfg.Listen,
		Handler:           server,
		ReadHeaderTimeout: 15 * time.Second,
		ReadTimeout:       5 * time.Minute,
		IdleTimeout:       120 * time.Second,
	}

	go func() {
		logger.Printf("listening on %s with %d provider(s)", cfg.Listen, len(cfg.Providers))
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatalf("serve: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	// Stop admitting new work before waiting for the HTTP server to drain.
	server.BeginShutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	shutdownErr := httpServer.Shutdown(ctx)
	cancel()
	if shutdownErr != nil {
		logger.Printf("graceful shutdown incomplete: %v; canceling active requests", shutdownErr)
		server.CancelActiveRequests()
		_ = httpServer.Close()
	}

	// Wait for all handlers to release aggregate buffers before the store
	// removes the instance cache directory. If the graceful deadline expired,
	// the active request contexts were already canceled above.
	if err := server.Shutdown(context.Background()); err != nil {
		logger.Printf("shutdown: %v", err)
	}
}
