package templates

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

// These are intentionally rendered through the embedded templates and parsed
// through the public package reader. That catches drift between `af init` and
// the local-install manifest contract.
func TestRenderedScaffoldManifestIsInstallable(t *testing.T) {
	if packages.CurrentConfigVersion != 1 {
		t.Fatalf("CurrentConfigVersion = %d, want 1; scaffold manifests target v1", packages.CurrentConfigVersion)
	}

	tests := []struct {
		language string
		start    string
	}{
		{language: "python", start: "python main.py"},
		{language: "go", start: "go run ."},
		{language: "typescript", start: "npm run start"},
	}

	for _, tt := range tests {
		t.Run(tt.language, func(t *testing.T) {
			data := TemplateData{
				ProjectName: `project: "quoted" & punctuated`,
				NodeID:      "representative-node",
				AuthorName:  `A. Author: "The Builder"`,
				Language:    tt.language,
				AgentPort:   8173,
			}
			tmpl, err := GetTemplate(tt.language + "/agentfield-package.yaml.tmpl")
			if err != nil {
				t.Fatalf("GetTemplate: %v", err)
			}
			var rendered bytes.Buffer
			if err := tmpl.Execute(&rendered, data); err != nil {
				t.Fatalf("render manifest: %v", err)
			}
			if !strings.Contains(rendered.String(), "user_environment:\n  required: []") {
				t.Fatalf("rendered manifest must explicitly declare an empty required environment list:\n%s", rendered.String())
			}

			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), rendered.Bytes(), 0o644); err != nil {
				t.Fatal(err)
			}
			metadata, err := packages.ParsePackageMetadata(dir)
			if err != nil {
				t.Fatalf("ParsePackageMetadata(rendered manifest): %v\n%s", err, rendered.String())
			}
			if metadata.ConfigVersion != "v1" || metadata.ConfigVersionNumber() != 1 {
				t.Errorf("config version = %q (%d), want v1 (1)", metadata.ConfigVersion, metadata.ConfigVersionNumber())
			}
			if metadata.Name != data.NodeID || metadata.AgentNode.NodeID != data.NodeID {
				t.Errorf("name/node_id = %q/%q, want %q", metadata.Name, metadata.AgentNode.NodeID, data.NodeID)
			}
			if metadata.Version != "1.0.0" || metadata.Language != tt.language {
				t.Errorf("version/language = %q/%q, want 1.0.0/%q", metadata.Version, metadata.Language, tt.language)
			}
			if metadata.Author != data.AuthorName {
				t.Errorf("author = %q, want %q", metadata.Author, data.AuthorName)
			}
			if !strings.Contains(metadata.Description, data.ProjectName) {
				t.Errorf("description = %q, want it to retain project name %q", metadata.Description, data.ProjectName)
			}
			if metadata.AgentNode.DefaultPort != data.AgentPort {
				t.Errorf("default port = %d, want %d", metadata.AgentNode.DefaultPort, data.AgentPort)
			}
			if metadata.HealthcheckPath() != "/health" || metadata.Entrypoint.Start != tt.start {
				t.Errorf("healthcheck/start = %q/%q, want /health/%q", metadata.HealthcheckPath(), metadata.Entrypoint.Start, tt.start)
			}
			if len(metadata.UserEnvironment.Required) != 0 {
				t.Errorf("required environment = %#v, want explicit empty list", metadata.UserEnvironment.Required)
			}
		})
	}
}

func TestTypeScriptScaffoldPackageScripts(t *testing.T) {
	tmpl, err := GetTemplate("typescript/package.json.tmpl")
	if err != nil {
		t.Fatal(err)
	}
	var rendered bytes.Buffer
	if err := tmpl.Execute(&rendered, TemplateData{ProjectName: `typescript: "quoted" & punctuated`}); err != nil {
		t.Fatal(err)
	}
	var pkg struct {
		Name    string            `json:"name"`
		Scripts map[string]string `json:"scripts"`
	}
	if err := json.Unmarshal(rendered.Bytes(), &pkg); err != nil {
		t.Fatalf("rendered package.json is invalid: %v\n%s", err, rendered.String())
	}
	if pkg.Name != `typescript: "quoted" & punctuated` {
		t.Errorf("package name = %q, want punctuation preserved", pkg.Name)
	}
	for script, want := range map[string]string{
		"start": "tsx main.ts",
		"dev":   "tsx main.ts",
		"lint":  "tsc --noEmit",
	} {
		if got := pkg.Scripts[script]; got != want {
			t.Errorf("scripts.%s = %q, want %q", script, got, want)
		}
	}
}

func TestRenderedScaffoldManifestRejectsEmptyNodeID(t *testing.T) {
	tmpl, err := GetTemplate("python/agentfield-package.yaml.tmpl")
	if err != nil {
		t.Fatal(err)
	}
	var rendered bytes.Buffer
	if err := tmpl.Execute(&rendered, TemplateData{ProjectName: "empty", AuthorName: "Nobody", AgentPort: 1}); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), rendered.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := packages.ParsePackageMetadata(dir); err == nil {
		t.Fatal("ParsePackageMetadata accepted an empty rendered node ID")
	}
}

// String template values may come from user-entered project metadata. Exercise
// control characters as well as punctuation so the YAML quoting helper cannot
// accidentally turn a newline or tab into YAML structure.
func TestRenderedScaffoldManifestEscapesControlCharacters(t *testing.T) {
	data := TemplateData{
		ProjectName: "project: first line\nsecond line # &",
		NodeID:      "control-character-node",
		AuthorName:  "Author\t\"quoted\"",
		AgentPort:   1,
	}

	for _, language := range GetSupportedLanguages() {
		t.Run(language, func(t *testing.T) {
			tmpl, err := GetTemplate(language + "/agentfield-package.yaml.tmpl")
			if err != nil {
				t.Fatal(err)
			}
			var rendered bytes.Buffer
			if err := tmpl.Execute(&rendered, data); err != nil {
				t.Fatal(err)
			}

			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), rendered.Bytes(), 0o644); err != nil {
				t.Fatal(err)
			}
			metadata, err := packages.ParsePackageMetadata(dir)
			if err != nil {
				t.Fatalf("ParsePackageMetadata: %v\n%s", err, rendered.String())
			}
			if metadata.Author != data.AuthorName {
				t.Errorf("author = %q, want %q", metadata.Author, data.AuthorName)
			}
			if !strings.Contains(metadata.Description, data.ProjectName) {
				t.Errorf("description = %q, want it to retain %q", metadata.Description, data.ProjectName)
			}
		})
	}
}
