package prompts

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Prompt struct {
	RequiredVariables []string `yaml:"required_variables" json:"required_variables"`
	Text              string   `yaml:"text" json:"text"`
}

type File struct {
	Version string            `yaml:"version" json:"version"`
	Prompts map[string]Prompt `yaml:"prompts" json:"prompts"`
}

type Store struct {
	version string
	items   map[string]Prompt
	source  string
}

func Load(defaultPath, overridePath string) (*Store, error) {
	path := firstNonBlank(overridePath, defaultPath)
	if path == "" {
		return nil, errors.New("prompt file path is required")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read prompts %s: %w", path, err)
	}
	var f File
	if err := yaml.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("parse prompts %s: %w", path, err)
	}
	if len(f.Prompts) == 0 {
		return nil, fmt.Errorf("prompts %s did not define any prompts", path)
	}
	return &Store{version: f.Version, items: f.Prompts, source: path}, nil
}

func (s *Store) Source() string {
	if s == nil {
		return ""
	}
	return s.source
}

func (s *Store) Version() string {
	if s == nil {
		return ""
	}
	return s.version
}

func (s *Store) Get(key string) (Prompt, bool) {
	if s == nil {
		return Prompt{}, false
	}
	p, ok := s.items[key]
	return p, ok
}

func (s *Store) Render(key string, vars map[string]string) (string, error) {
	p, ok := s.Get(key)
	if !ok {
		return "", fmt.Errorf("prompt %q not found", key)
	}
	out := p.Text
	for _, required := range p.RequiredVariables {
		if _, ok := vars[required]; !ok {
			return "", fmt.Errorf("prompt %q missing variable %q", key, required)
		}
	}
	for key, value := range vars {
		out = strings.ReplaceAll(out, "{{"+key+"}}", value)
	}
	return strings.TrimSpace(out), nil
}

func firstNonBlank(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}
