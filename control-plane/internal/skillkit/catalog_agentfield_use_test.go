package skillkit

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

type skillFrontmatter struct {
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
}

func skillSource(t *testing.T, skillName string) []byte {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate source test file")
	}
	path := filepath.Join(filepath.Dir(file), "..", "..", "..", "skills", skillName, "SKILL.md")
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read canonical %s skill %q: %v", skillName, path, err)
	}
	return content
}

func parseSkillFrontmatter(content []byte) (skillFrontmatter, error) {
	text := string(content)
	if !strings.HasPrefix(text, "---\n") {
		return skillFrontmatter{}, fmt.Errorf("missing opening frontmatter delimiter")
	}
	rest := strings.TrimPrefix(text, "---\n")
	end := strings.Index(rest, "\n---\n")
	if end < 0 {
		return skillFrontmatter{}, fmt.Errorf("missing closing frontmatter delimiter")
	}
	var frontmatter skillFrontmatter
	if err := yaml.Unmarshal([]byte(rest[:end]), &frontmatter); err != nil {
		return skillFrontmatter{}, fmt.Errorf("parse frontmatter: %w", err)
	}
	return frontmatter, nil
}

// Contract: the agentfield-use skill is registered and its content is really
// embedded — a missing go:embed directive would only surface at install time
// as "embedded skill is empty" without this guard.
func TestAgentfieldUseSkillEmbedded(t *testing.T) {
	skill, err := CatalogByName("agentfield-use")
	if err != nil {
		t.Fatalf("CatalogByName: %v", err)
	}
	if skill.Version == "" || skill.Trigger == "" {
		t.Fatalf("agentfield-use catalog entry incomplete: %+v", skill)
	}

	files, err := skill.EnumerateFiles()
	if err != nil {
		t.Fatalf("EnumerateFiles: %v", err)
	}
	if _, ok := files[skill.EntryFile]; !ok {
		t.Fatalf("entry file %q not embedded (got %d files)", skill.EntryFile, len(files))
	}

	content, err := skill.EntryContent()
	if err != nil {
		t.Fatalf("EntryContent: %v", err)
	}
	// The consumer skill must teach the durable discovery + execute surface.
	for _, needle := range []string{
		"/api/v1/discovery/capabilities",
		"/api/v1/execute/async/",
		"/api/v1/executions/",
	} {
		if !strings.Contains(string(content), needle) {
			t.Fatalf("agentfield-use SKILL.md is missing %q", needle)
		}
	}
}

func TestAgentfieldUseSourceFallbackContract(t *testing.T) {
	content := string(skillSource(t, "agentfield-use"))
	frontmatter, err := parseSkillFrontmatter([]byte(content))
	if err != nil {
		t.Fatalf("parse source frontmatter: %v", err)
	}
	if frontmatter.Name != "agentfield-use" || frontmatter.Version != "0.4.0" {
		t.Fatalf("source frontmatter = %+v, want name=agentfield-use version=0.4.0", frontmatter)
	}

	// These assertions make the handoff contract explicit: fallback is available
	// only after coverage is conclusively checked, and it cannot authorize itself.
	for _, needle := range []string{
		"health check,\ncapability discovery",
		"ranked search for the requested job",
		"No capable installed agent was found for this job.",
		"canonical **`agentfield`** skill",
		"List, inspect, and diagnose-only\nrequests never authorize building an agent.",
		"original request already authorized creating an agent, or when the user\nexplicitly accepts this offer",
		"agentfield: coverage_precheck_complete = true",
		"must enter its deliverable-selection path rather than\nrepeat discovery",
		"at most once: after this handoff, never re-enter discovery or bounce\nrecursively",
	} {
		if !strings.Contains(content, needle) {
			t.Fatalf("agentfield-use source SKILL.md is missing fallback contract text %q", needle)
		}
	}

	// The receiving skill must consume the same-request signal and proceed to
	// delivery selection. Otherwise the fallback can loop through discovery.
	builder := string(skillSource(t, "agentfield"))
	for _, needle := range []string{
		"coverage_precheck_complete",
		"Do not repeat this pre-check when `agentfield-use` returns here after finding no coverage.",
		"Only after the one-time pre-check establishes that no installed agent covers the request, choose the delivery:",
	} {
		if !strings.Contains(builder, needle) {
			t.Fatalf("agentfield source SKILL.md is missing same-request fallback handoff text %q", needle)
		}
	}
}

func TestParseSkillFrontmatterEdgeCases(t *testing.T) {
	tests := []struct {
		name    string
		content string
		wantErr string
	}{
		{name: "empty", content: "", wantErr: "missing opening"},
		{name: "missing closing delimiter", content: "---\nname: agentfield-use\n", wantErr: "missing closing"},
		{name: "malformed yaml", content: "---\nname: [\n---\n", wantErr: "parse frontmatter"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseSkillFrontmatter([]byte(tc.content))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("parseSkillFrontmatter(%q) error = %v, want containing %q", tc.content, err, tc.wantErr)
			}
		})
	}
}

// Contract: each skill's marker block carries its own trigger sentence, so a
// rules file holding both skills routes build requests and use requests to
// the right SKILL.md.
func TestPointerBlocksAreSkillSpecific(t *testing.T) {
	build, err := CatalogByName("agentfield")
	if err != nil {
		t.Fatalf("CatalogByName: %v", err)
	}
	use, err := CatalogByName("agentfield-use")
	if err != nil {
		t.Fatalf("CatalogByName: %v", err)
	}
	buildBlock := renderPointerBlock(build, "/canonical/agentfield/current")
	useBlock := renderPointerBlock(use, "/canonical/agentfield-use/current")
	if !strings.Contains(buildBlock, "architect or build") {
		t.Fatalf("build skill block lost its trigger: %q", buildBlock)
	}
	if !strings.Contains(useBlock, "delegate work") {
		t.Fatalf("use skill block lost its trigger: %q", useBlock)
	}
	if strings.Contains(useBlock, "architect or build") {
		t.Fatal("use skill block reuses the build trigger")
	}
}
