package skillkit

import (
	"fmt"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

type builderSkillFrontmatter struct {
	Name    string   `yaml:"name"`
	Version string   `yaml:"version"`
	Aliases []string `yaml:"aliases"`
}

func TestAgentfieldBuilderSourceFrontmatterContract(t *testing.T) {
	content := skillSource(t, "agentfield")
	var frontmatter builderSkillFrontmatter
	if err := parseBuilderSkillFrontmatter(content, &frontmatter); err != nil {
		t.Fatalf("parse source frontmatter: %v", err)
	}
	if frontmatter.Name != "agentfield" || frontmatter.Version != "0.5.1" {
		t.Fatalf("source frontmatter = %+v, want name=agentfield version=0.5.1", frontmatter)
	}
	if !containsString(frontmatter.Aliases, "agentfield-multi-reasoner-builder") {
		t.Fatalf("source aliases = %v, want agentfield-multi-reasoner-builder", frontmatter.Aliases)
	}
}

func TestParseBuilderSkillFrontmatterEdgeCases(t *testing.T) {
	tests := []struct {
		name    string
		content string
		wantErr string
		want    builderSkillFrontmatter
	}{
		{name: "empty", content: "", wantErr: "missing opening"},
		{name: "missing closing delimiter", content: "---\nname: agentfield\n", wantErr: "missing closing"},
		{name: "malformed yaml", content: "---\nname: [\n---\n", wantErr: "parse frontmatter"},
		{name: "missing version", content: "---\nname: agentfield\naliases: null\n---\n", want: builderSkillFrontmatter{Name: "agentfield"}},
		{name: "no aliases", content: "---\nname: agentfield\nversion: 0.6.0\naliases: []\n---\n", want: builderSkillFrontmatter{Name: "agentfield", Version: "0.6.0"}},
		{name: "null aliases", content: "---\nname: agentfield\nversion: 0.6.0\naliases: null\n---\n", want: builderSkillFrontmatter{Name: "agentfield", Version: "0.6.0"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got builderSkillFrontmatter
			err := parseBuilderSkillFrontmatter([]byte(tc.content), &got)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("parseBuilderSkillFrontmatter(%q) error = %v, want containing %q", tc.content, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseBuilderSkillFrontmatter(%q): %v", tc.content, err)
			}
			if got.Name != tc.want.Name || got.Version != tc.want.Version || len(got.Aliases) != 0 {
				t.Fatalf("parseBuilderSkillFrontmatter(%q) = %+v, want %+v", tc.content, got, tc.want)
			}
		})
	}
}

// The builder skill must keep the pre-split repository workflow untouched:
// no coverage pre-check, no deliverable routing, no personal-agent branch.
// Personal agents are the agentfield-personal skill's job; the only pointer
// the builder carries is the one sentence in its description frontmatter.
func TestAgentfieldBuilderHasNoRoutingGate(t *testing.T) {
	content := string(skillSource(t, "agentfield"))
	for _, banned := range []string{
		"coverage_precheck_complete",
		"Which deliverable do you want:",
		"Coverage pre-check",
		"Personal agent workflow",
		"~/agentfield-agents",
	} {
		if strings.Contains(content, banned) {
			t.Fatalf("builder source SKILL.md must not contain routing/personal-agent text %q", banned)
		}
	}
	for _, needle := range []string{"## Workflow", "## Output contract"} {
		if !strings.Contains(content, needle) {
			t.Fatalf("builder source SKILL.md is missing original heading %q", needle)
		}
	}
	if !strings.Contains(content, "`agentfield-personal` skill") {
		t.Fatal("builder description must point machine-installed agent requests at agentfield-personal")
	}
}

func TestAgentfieldBuilderProjectRepositoryRegressionContract(t *testing.T) {
	content := string(skillSource(t, "agentfield"))
	for _, needle := range []string{
		"`af init <slug> --language python --docker --defaults --non-interactive --default-model <model>`",
		"`docker compose config`",
		"`docker compose up --build`",
		"POST http://localhost:8080/api/v1/execute/async/<slug>.<entry>",
		"succeeded) echo \"$R\" | jq '.result'; break ;;",
	} {
		if !strings.Contains(content, needle) {
			t.Fatalf("builder source SKILL.md is missing preserved repository workflow text %q", needle)
		}
	}
}

func parseBuilderSkillFrontmatter(content []byte, into *builderSkillFrontmatter) error {
	text := string(content)
	if !strings.HasPrefix(text, "---\n") {
		return fmt.Errorf("missing opening frontmatter delimiter")
	}
	text = strings.TrimPrefix(text, "---\n")
	end := strings.Index(text, "\n---\n")
	if end < 0 {
		return fmt.Errorf("missing closing frontmatter delimiter")
	}
	if err := yaml.Unmarshal([]byte(text[:end]), into); err != nil {
		return fmt.Errorf("parse frontmatter: %w", err)
	}
	return nil
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
