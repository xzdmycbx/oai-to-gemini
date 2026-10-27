package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

const (
	DefaultSSESpillThresholdBytes int64 = 1 << 20  // 1 MiB
	DefaultSSECacheMaxBytes       int64 = 64 << 20 // 64 MiB
)

// Provider describes one upstream Gemini-compatible endpoint.
//
// api_base is the canonical field. api-base and base_url are accepted as
// aliases for convenience. name is always required and is used as the first
// path segment of incoming requests.
type Provider struct {
	Name        string `yaml:"name" json:"name"`
	APIBase     string `yaml:"api_base" json:"api_base"`
	APIBaseDash string `yaml:"api-base" json:"api-base"`
	BaseURL     string `yaml:"base_url" json:"base_url"`
}

type Config struct {
	Listen                 string     `yaml:"listen" json:"listen"`
	SSESpillThresholdBytes int64      `yaml:"sse_spill_threshold_bytes" json:"sse_spill_threshold_bytes"`
	SSECacheMaxBytes       int64      `yaml:"sse_cache_max_bytes" json:"sse_cache_max_bytes"`
	SSECacheDir            string     `yaml:"sse_cache_dir" json:"sse_cache_dir"`
	Providers              []Provider `yaml:"providers" json:"providers"`
}

// Load reads a YAML or JSON config file.
//
// An empty path means: CONFIG_PATH if set, otherwise config.yaml, falling back
// to config.json when config.yaml does not exist.
func Load(path string) (*Config, error) {
	if path == "" {
		if env := os.Getenv("CONFIG_PATH"); env != "" {
			path = env
		} else {
			path = "config.yaml"
			if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
				if _, jsonErr := os.Stat("config.json"); jsonErr == nil {
					path = "config.json"
				}
			}
		}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}

	var cfg Config
	switch strings.ToLower(filepath.Ext(path)) {
	case ".json":
		err = json.Unmarshal(data, &cfg)
	default:
		err = yaml.Unmarshal(data, &cfg)
	}
	if err != nil {
		return nil, fmt.Errorf("parse config %q: %w", path, err)
	}
	if err := cfg.normalize(); err != nil {
		return nil, fmt.Errorf("invalid config %q: %w", path, err)
	}
	return &cfg, nil
}

func (c *Config) normalize() error {
	if c.Listen == "" {
		c.Listen = ":8080"
	}
	if c.SSESpillThresholdBytes <= 0 {
		c.SSESpillThresholdBytes = DefaultSSESpillThresholdBytes
	}
	if c.SSECacheMaxBytes <= 0 {
		c.SSECacheMaxBytes = DefaultSSECacheMaxBytes
	}

	seen := make(map[string]struct{}, len(c.Providers))
	for i := range c.Providers {
		p := &c.Providers[i]
		p.Name = strings.TrimSpace(p.Name)
		if p.Name == "" {
			return fmt.Errorf("providers[%d]: name is required", i)
		}
		if strings.ContainsAny(p.Name, "/?# \t\r\n") {
			return fmt.Errorf("providers[%d]: name %q must not contain whitespace, '/', '?' or '#'", i, p.Name)
		}
		if strings.Contains(p.Name, "%") {
			return fmt.Errorf("providers[%d]: name %q must not contain '%%'; request path segments are percent-decoded exactly once by the HTTP layer", i, p.Name)
		}
		if _, ok := seen[p.Name]; ok {
			return fmt.Errorf("providers[%d]: duplicate provider name %q", i, p.Name)
		}
		seen[p.Name] = struct{}{}

		base := p.APIBase
		if base == "" {
			base = p.APIBaseDash
		}
		if base == "" {
			base = p.BaseURL
		}
		base = strings.TrimRight(strings.TrimSpace(base), "/")
		if base == "" {
			return fmt.Errorf("providers[%d] (%s): api_base is required", i, p.Name)
		}
		u, err := url.Parse(base)
		if err != nil || u.Scheme == "" || u.Host == "" {
			return fmt.Errorf("providers[%d] (%s): invalid api_base %q", i, p.Name, base)
		}
		if u.Scheme != "http" && u.Scheme != "https" {
			return fmt.Errorf("providers[%d] (%s): api_base scheme must be http or https", i, p.Name)
		}
		if u.RawQuery != "" || u.Fragment != "" {
			return fmt.Errorf("providers[%d] (%s): api_base must not contain a query or fragment", i, p.Name)
		}
		p.APIBase = base
		p.APIBaseDash = ""
		p.BaseURL = ""
	}
	return nil
}
