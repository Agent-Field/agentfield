package templates

import (
	"bytes"
	"slices"
	"strings"
	"testing"
)

func TestGetTemplate(t *testing.T) {
	t.Parallel()

	data := TemplateData{
		NodeID: "agent-123",
	}

	tests := []struct {
		name        string
		template    string
		wantText    string
		wantErrText string
	}{
		{
			name:     "parses and executes python template",
			template: "python/main.py.tmpl",
			wantText: `"agent-123"`,
		},
		{
			name:     "parses and executes go template",
			template: "go/main.go.tmpl",
			wantText: `"agent-123"`,
		},
		{
			name:        "missing template returns error",
			template:    "python/does-not-exist.tmpl",
			wantErrText: "pattern matches no files",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			tmpl, err := GetTemplate(tt.template)
			if tt.wantErrText != "" {
				if err == nil {
					t.Fatalf("GetTemplate(%q) error = nil, want %q", tt.template, tt.wantErrText)
				}
				if !strings.Contains(err.Error(), tt.wantErrText) {
					t.Fatalf("GetTemplate(%q) error = %q, want substring %q", tt.template, err.Error(), tt.wantErrText)
				}
				return
			}

			if err != nil {
				t.Fatalf("GetTemplate(%q) error = %v", tt.template, err)
			}

			var buf bytes.Buffer
			if err := tmpl.Execute(&buf, data); err != nil {
				t.Fatalf("Execute() error = %v", err)
			}
			if !strings.Contains(buf.String(), tt.wantText) {
				t.Fatalf("rendered template missing %q in output:\n%s", tt.wantText, buf.String())
			}
		})
	}
}

func TestGetTemplateFiles(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		language    string
		want        map[string]string
		wantErrText string
	}{
		{
			name:     "python templates",
			language: "python",
			want: map[string]string{
				"python/.env.example.tmpl":     ".env.example",
				"python/.gitignore.tmpl":       ".gitignore",
				"python/README.md.tmpl":        "README.md",
				"python/main.py.tmpl":          "main.py",
				"python/reasoners.py.tmpl":     "reasoners.py",
				"python/requirements.txt.tmpl": "requirements.txt",
			},
		},
		{
			name:     "go templates",
			language: "go",
			want: map[string]string{
				"go/.env.example.tmpl": ".env.example",
				"go/.gitignore.tmpl":   ".gitignore",
				"go/README.md.tmpl":    "README.md",
				"go/go.mod.tmpl":       "go.mod",
				"go/main.go.tmpl":      "main.go",
				"go/reasoners.go.tmpl": "reasoners.go",
			},
		},
		{
			name:     "typescript templates",
			language: "typescript",
			want: map[string]string{
				"typescript/.env.example.tmpl":  ".env.example",
				"typescript/.gitignore.tmpl":    ".gitignore",
				"typescript/README.md.tmpl":     "README.md",
				"typescript/main.ts.tmpl":       "main.ts",
				"typescript/package.json.tmpl":  "package.json",
				"typescript/reasoners.ts.tmpl":  "reasoners.ts",
				"typescript/tsconfig.json.tmpl": "tsconfig.json",
			},
		},
		{
			name:        "unsupported language",
			language:    "ruby",
			wantErrText: "unsupported language: ruby",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := GetTemplateFiles(tt.language)
			if tt.wantErrText != "" {
				if err == nil {
					t.Fatalf("GetTemplateFiles(%q) error = nil, want %q", tt.language, tt.wantErrText)
				}
				if !strings.Contains(err.Error(), tt.wantErrText) {
					t.Fatalf("GetTemplateFiles(%q) error = %q, want substring %q", tt.language, err.Error(), tt.wantErrText)
				}
				return
			}

			if err != nil {
				t.Fatalf("GetTemplateFiles(%q) error = %v", tt.language, err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("GetTemplateFiles(%q) returned %d files, want %d: %#v", tt.language, len(got), len(tt.want), got)
			}
			for wantPath, wantDest := range tt.want {
				if got[wantPath] != wantDest {
					t.Fatalf("GetTemplateFiles(%q)[%q] = %q, want %q", tt.language, wantPath, got[wantPath], wantDest)
				}
			}
		})
	}
}

func TestReadTemplateContent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		path        string
		wantText    string
		wantErrText string
	}{
		{
			name:     "reads embedded content",
			path:     "typescript/package.json.tmpl",
			wantText: `"@agentfield/sdk"`,
		},
		{
			name:        "missing content returns error",
			path:        "typescript/missing.tmpl",
			wantErrText: "file does not exist",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := ReadTemplateContent(tt.path)
			if tt.wantErrText != "" {
				if err == nil {
					t.Fatalf("ReadTemplateContent(%q) error = nil, want %q", tt.path, tt.wantErrText)
				}
				if !strings.Contains(err.Error(), tt.wantErrText) {
					t.Fatalf("ReadTemplateContent(%q) error = %q, want substring %q", tt.path, err.Error(), tt.wantErrText)
				}
				return
			}

			if err != nil {
				t.Fatalf("ReadTemplateContent(%q) error = %v", tt.path, err)
			}
			if !strings.Contains(string(got), tt.wantText) {
				t.Fatalf("ReadTemplateContent(%q) missing %q in output:\n%s", tt.path, tt.wantText, string(got))
			}
		})
	}
}

func TestGetSupportedLanguages(t *testing.T) {
	t.Parallel()

	got := GetSupportedLanguages()
	want := []string{"python", "go", "typescript"}

	if !slices.Equal(got, want) {
		t.Fatalf("GetSupportedLanguages() = %v, want %v", got, want)
	}
}
