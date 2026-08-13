package templates

import "testing"

func TestGetDockerTemplateFiles(t *testing.T) {
	t.Parallel()

	// docker-compose.yml.tmpl builds the agent service from `dockerfile:
	// Dockerfile`, so every supported language must map a Dockerfile template.
	tests := []struct {
		name     string
		language string
		wantFile string
	}{
		{
			name:     "python includes language dockerfile",
			language: "python",
			wantFile: "docker/python.Dockerfile.tmpl",
		},
		{
			name:     "go includes language dockerfile",
			language: "go",
			wantFile: "docker/go.Dockerfile.tmpl",
		},
		{
			name:     "typescript includes language dockerfile",
			language: "typescript",
			wantFile: "docker/typescript.Dockerfile.tmpl",
		},
	}

	common := map[string]string{
		"docker/docker-compose.yml.tmpl": "docker-compose.yml",
		"docker/.env.example.tmpl":       ".env.example",
		"docker/.dockerignore.tmpl":      ".dockerignore",
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := GetDockerTemplateFiles(tt.language)

			for path, dest := range common {
				if got[path] != dest {
					t.Fatalf("GetDockerTemplateFiles(%q)[%q] = %q, want %q", tt.language, path, got[path], dest)
				}
			}

			if got[tt.wantFile] != "Dockerfile" {
				t.Fatalf("GetDockerTemplateFiles(%q)[%q] = %q, want %q", tt.language, tt.wantFile, got[tt.wantFile], "Dockerfile")
			}

			dockerfiles := 0
			for path := range got {
				if got[path] == "Dockerfile" {
					dockerfiles++
				}
			}
			if dockerfiles != 1 {
				t.Fatalf("GetDockerTemplateFiles(%q) maps %d Dockerfile templates, want exactly 1", tt.language, dockerfiles)
			}
		})
	}
}
