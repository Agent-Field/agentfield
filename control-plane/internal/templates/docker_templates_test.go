package templates

import "testing"

func TestGetDockerTemplateFiles(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		language       string
		wantFile       string
		wantDest       string
		wantDockerfile bool
	}{
		{
			name:           "python includes language dockerfile",
			language:       "python",
			wantFile:       "docker/python.Dockerfile.tmpl",
			wantDest:       "Dockerfile",
			wantDockerfile: true,
		},
		{
			name:           "go omits language dockerfile",
			language:       "go",
			wantDockerfile: false,
		},
		{
			name:           "typescript omits language dockerfile",
			language:       "typescript",
			wantDockerfile: false,
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

			_, hasDockerfile := got["docker/python.Dockerfile.tmpl"]
			if hasDockerfile != tt.wantDockerfile {
				t.Fatalf("GetDockerTemplateFiles(%q) python dockerfile present = %v, want %v", tt.language, hasDockerfile, tt.wantDockerfile)
			}

			if tt.wantDockerfile && got[tt.wantFile] != tt.wantDest {
				t.Fatalf("GetDockerTemplateFiles(%q)[%q] = %q, want %q", tt.language, tt.wantFile, got[tt.wantFile], tt.wantDest)
			}
		})
	}
}
