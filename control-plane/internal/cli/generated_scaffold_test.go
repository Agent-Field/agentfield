package cli_test

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/templates"
)

func TestGeneratedScaffoldsHaveInstallableV1Manifests(t *testing.T) {
	if packages.CurrentConfigVersion != 1 {
		t.Fatalf("CurrentConfigVersion = %d, want 1", packages.CurrentConfigVersion)
	}

	data := templates.TemplateData{
		ProjectName: "Project: punctuation & more",
		NodeID:      "representative-node",
		AuthorName:  "O'Reilly: Jane \"JJ\"",
		AgentPort:   8123,
	}
	wantStarts := map[string]string{
		"python":     "python main.py",
		"go":         "go run .",
		"typescript": "npm run start",
	}

	for language, wantStart := range wantStarts {
		t.Run(language, func(t *testing.T) {
			root := t.TempDir()
			files, err := templates.GetTemplateFiles(language)
			if err != nil {
				t.Fatal(err)
			}
			manifestCount := 0
			for templatePath, destination := range files {
				if destination == "agentfield-package.yaml" {
					manifestCount++
				}
				tmpl, err := templates.GetTemplate(templatePath)
				if err != nil {
					t.Fatal(err)
				}
				var rendered bytes.Buffer
				if err := tmpl.Execute(&rendered, data); err != nil {
					t.Fatal(err)
				}
				path := filepath.Join(root, destination)
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, rendered.Bytes(), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if manifestCount != 1 {
				t.Fatalf("rendered %d root manifests, want 1", manifestCount)
			}

			metadata, err := packages.ParsePackageMetadata(root)
			if err != nil {
				t.Fatalf("ParsePackageMetadata: %v", err)
			}
			if metadata.ConfigVersionNumber() != 1 || metadata.Name != data.NodeID || metadata.Version != "1.0.0" || metadata.Language != language || metadata.Author != data.AuthorName {
				t.Fatalf("unexpected manifest metadata: %#v", metadata)
			}
			if metadata.AgentNode.NodeID != data.NodeID || metadata.AgentNode.DefaultPort != data.AgentPort || metadata.Entrypoint.Start != wantStart || metadata.HealthcheckPath() != "/health" {
				t.Fatalf("unexpected executable metadata: %#v", metadata)
			}
			if len(metadata.UserEnvironment.Required) != 0 {
				t.Fatalf("required environment = %#v, want empty", metadata.UserEnvironment.Required)
			}

			if language == "typescript" {
				var packageJSON struct {
					Scripts map[string]string `json:"scripts"`
				}
				contents, err := os.ReadFile(filepath.Join(root, "package.json"))
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(contents, &packageJSON); err != nil {
					t.Fatalf("parse package.json: %v", err)
				}
				wantScripts := map[string]string{"start": "tsx main.ts", "dev": "tsx main.ts", "lint": "tsc --noEmit"}
				if len(packageJSON.Scripts) != len(wantScripts) {
					t.Fatalf("scripts = %#v, want %#v", packageJSON.Scripts, wantScripts)
				}
				for name, want := range wantScripts {
					if packageJSON.Scripts[name] != want {
						t.Fatalf("scripts[%q] = %q, want %q", name, packageJSON.Scripts[name], want)
					}
				}
			}
		})
	}
}
