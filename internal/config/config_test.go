package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadYAMLAliasesAndDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	data := []byte(`
providers:
  - name: newapi1
    api-base: https://example.invalid/v1beta/
  - name: newapi2
    base_url: https://example.invalid/v1beta
`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Listen != ":8080" {
		t.Fatalf("listen = %q", cfg.Listen)
	}
	if cfg.SSESpillThresholdBytes != DefaultSSESpillThresholdBytes {
		t.Fatalf("threshold = %d", cfg.SSESpillThresholdBytes)
	}
	if cfg.SSECacheMaxBytes != DefaultSSECacheMaxBytes {
		t.Fatalf("max = %d", cfg.SSECacheMaxBytes)
	}
	if len(cfg.Providers) != 2 {
		t.Fatalf("providers = %d", len(cfg.Providers))
	}
	if cfg.Providers[0].APIBase != "https://example.invalid/v1beta" {
		t.Fatalf("api_base = %q", cfg.Providers[0].APIBase)
	}
	if cfg.Providers[1].APIBase != "https://example.invalid/v1beta" {
		t.Fatalf("base_url alias = %q", cfg.Providers[1].APIBase)
	}
}

func TestLoadJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	data := []byte(`{
		"listen": ":9090",
		"sse_cache_max_bytes": 1234,
		"providers": [{"name": "p1", "api_base": "https://example.invalid"}]
	}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Listen != ":9090" || cfg.SSECacheMaxBytes != 1234 {
		t.Fatalf("cfg = %#v", cfg)
	}
}

func TestLoadRejectsAPIBaseQueryOrFragment(t *testing.T) {
	for _, base := range []string{
		"https://example.invalid/v1beta?region=test",
		"https://example.invalid/v1beta#fragment",
	} {
		dir := t.TempDir()
		path := filepath.Join(dir, "config.yaml")
		data := []byte("providers:\n  - name: p1\n    api_base: " + base + "\n")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(path); err == nil {
			t.Fatalf("expected api_base %q to be rejected", base)
		}
	}
}

func TestLoadRejectsProviderNameWithPercent(t *testing.T) {
	for _, name := range []string{"%61", "api%20one"} {
		dir := t.TempDir()
		path := filepath.Join(dir, "config.yaml")
		data := []byte("providers:\n  - name: \"" + name + "\"\n    api_base: https://example.invalid\n")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(path); err == nil {
			t.Fatalf("expected provider name %q to be rejected", name)
		}
	}
}

func TestLoadRejectsDuplicateProvider(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	data := []byte(`
providers:
  - name: same
    api_base: https://example.invalid
  - name: same
    api_base: https://example.invalid
`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected duplicate provider error")
	}
}
