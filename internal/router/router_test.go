package router

import (
	"testing"

	"github.com/xzdmycbx/oai-to-gemini/internal/config"
)

func TestResolveExistingProvider(t *testing.T) {
	providers := []config.Provider{
		{Name: "newapi1", APIBase: "https://example.invalid/v1beta"},
		{Name: "newapi2", APIBase: "https://example.invalid/v1beta"},
	}

	match, ok := Resolve("/newapi2/v1/chat/completions", providers)
	if !ok {
		t.Fatal("expected provider to resolve")
	}
	if match.Provider.Name != "newapi2" {
		t.Fatalf("provider = %q, want newapi2", match.Provider.Name)
	}
	if match.Path != "/v1/chat/completions" {
		t.Fatalf("path = %q, want /v1/chat/completions", match.Path)
	}
}

func TestResolveProviderRoot(t *testing.T) {
	providers := []config.Provider{{Name: "newapi1", APIBase: "https://example.invalid"}}
	match, ok := Resolve("/newapi1", providers)
	if !ok {
		t.Fatal("expected provider root to resolve")
	}
	if match.Path != "/" {
		t.Fatalf("path = %q, want /", match.Path)
	}
}

func TestResolveDoesNotDecodeProviderAgain(t *testing.T) {
	providers := []config.Provider{
		{Name: "a", APIBase: "https://example.invalid"},
		{Name: "%61", APIBase: "https://example.invalid"},
	}

	// net/http already decoded the request path once. A literal "%61" segment
	// must therefore match the provider literally named "%61", not "a".
	match, ok := Resolve("/%61/v1/chat/completions", providers)
	if !ok {
		t.Fatal("expected literal %61 provider to resolve")
	}
	if match.Provider.Name != "%61" {
		t.Fatalf("provider = %q, want literal %%61", match.Provider.Name)
	}

	// With only provider "a", a literal "%61" segment must not be decoded a
	// second time into "a".
	if _, ok := Resolve("/%61/v1/chat/completions", providers[:1]); ok {
		t.Fatal("literal %61 must not resolve to provider a")
	}
}

func TestResolveMissingProvider(t *testing.T) {
	providers := []config.Provider{{Name: "newapi1", APIBase: "https://example.invalid"}}
	if _, ok := Resolve("/missing/v1/chat/completions", providers); ok {
		t.Fatal("expected missing provider not to resolve")
	}
	if _, ok := Resolve("/", providers); ok {
		t.Fatal("expected empty path not to resolve")
	}
}
