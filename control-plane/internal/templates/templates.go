package templates

import (
	"embed"
	"fmt"
	"io/fs"
	"strings"
	"text/template"
)

//go:embed python/*.tmpl go/*.tmpl typescript/*.tmpl docker/*.tmpl
var content embed.FS

// TemplateData holds the data to be passed to the templates.
type TemplateData struct {
	ProjectName       string // "my-awesome-agent"
	NodeID            string // "my-awesome-agent" (same as ProjectName)
	GoModule          string // "my-awesome-agent" (Go module name)
	AuthorName        string // "John Doe"
	AuthorEmail       string // "john@example.com"
	CurrentYear       int    // 2025
	CreatedAt         string // "2025-01-05 10:30:00 EST"
	Language          string // "python", "go", or "typescript"
	// Docker scaffold fields (used only when --docker is set on `af init`)
	ControlPlaneImage string // "agentfield/control-plane:latest"
	ControlPlanePort  int    // 8080
	AgentPort         int    // 8001
	DefaultModel      string // "openrouter/google/gemini-2.5-flash"
}

// GetTemplate retrieves a specific template by its path.
func GetTemplate(name string) (*template.Template, error) {
	tmpl, err := template.ParseFS(content, name)
	if err != nil {
		return nil, err
	}
	return tmpl, nil
}

// GetTemplateFiles returns a map of template file paths for the specified language.
// The map keys are the template paths in the embed.FS, and values are the destination paths.
func GetTemplateFiles(language string) (map[string]string, error) {
	files := make(map[string]string)

	// Determine the language directory
	langDir := language
	if language != "python" && language != "go" && language != "typescript" {
		return nil, fmt.Errorf("unsupported language: %s (supported: python, go, typescript)", language)
	}

	// Walk the language-specific directory
	err := fs.WalkDir(content, langDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(path, ".tmpl") {
			// Remove the language prefix and .tmpl suffix
			// e.g., "python/main.py.tmpl" -> "main.py"
			relativePath := strings.TrimPrefix(path, langDir+"/")
			relativePath = strings.TrimSuffix(relativePath, ".tmpl")
			files[path] = relativePath
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return files, nil
}

// ReadTemplateContent reads the content of an embedded template file.
func ReadTemplateContent(path string) ([]byte, error) {
	return content.ReadFile(path)
}

// GetSupportedLanguages returns the list of supported languages.
func GetSupportedLanguages() []string {
	return []string{"python", "go", "typescript"}
}

// GetDockerTemplateFiles returns the minimal Docker infrastructure scaffold for a
// given language. Deliberately scoped to the four files an agent will NEVER need
// to customize: Dockerfile, docker-compose.yml, .env.example, .dockerignore.
//
// CLAUDE.md and README.md are NOT generated here — those are produced by the
// agentfield-multi-reasoner-builder skill AFTER the agent has written the real
// reasoner architecture in main.py, so they can contain real reasoner names,
// real curl examples, and a real architectural justification instead of
// placeholders that ship with TODO markers.
func GetDockerTemplateFiles(language string) map[string]string {
	files := map[string]string{
		"docker/docker-compose.yml.tmpl": "docker-compose.yml",
		"docker/.env.example.tmpl":       ".env.example",
		"docker/.dockerignore.tmpl":      ".dockerignore",
	}
	switch language {
	case "python":
		files["docker/python.Dockerfile.tmpl"] = "Dockerfile"
	}
	return files
}
