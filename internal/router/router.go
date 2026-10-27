package router

import (
	"strings"

	"github.com/xzdmycbx/oai-to-gemini/internal/config"
)

// Match is the result of resolving a request path against the configured
// providers.
type Match struct {
	Provider config.Provider
	Path     string // remaining path after removing /<provider>
}

// Resolve extracts the first path segment as the provider name. The provider
// must exist in the configuration; there is deliberately no default fallback.
//
// path must already be decoded exactly once, as net/http does for
// Request.URL.Path. This function deliberately does not call PathUnescape:
// doing so would decode a literal '%' in a configured provider name a second
// time and could route credentials to the wrong upstream.
func Resolve(path string, providers []config.Provider) (Match, bool) {
	if !strings.HasPrefix(path, "/") {
		return Match{}, false
	}
	rest := strings.TrimPrefix(path, "/")
	if rest == "" {
		return Match{}, false
	}

	name, tail, found := strings.Cut(rest, "/")
	if !found {
		tail = ""
	}
	if tail == "" {
		tail = "/"
	} else {
		tail = "/" + tail
	}

	for _, p := range providers {
		if p.Name == name {
			return Match{Provider: p, Path: tail}, true
		}
	}
	return Match{}, false
}
