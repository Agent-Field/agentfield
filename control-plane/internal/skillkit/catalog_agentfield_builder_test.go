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
	if frontmatter.Name != "agentfield" || frontmatter.Version != "0.6.0" {
		t.Fatalf("source frontmatter = %+v, want name=agentfield version=0.6.0", frontmatter)
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

func TestAgentfieldBuilderCoverageAndRoutingContract(t *testing.T) {
	content := string(skillSource(t, "agentfield"))
	for _, needle := range []string{
		"Before selecting a deliverable, entering the hard gate, designing, or scaffolding",
		"`agentfield-use` skill's health, discovery, and reasoner-capability search flow exactly once",
		"healthy, active",
		"documented description and input schema support",
		"Never infer coverage from an agent or reasoner name alone.",
		"do not build a duplicate: switch to `agentfield-use`",
		"control plane is unavailable",
		"af list",
		"installed-but-stopped agent from an absent capability",
		"Infer **Project repository**",
		"Infer **Personal agent**",
		"ask exactly this one question, then wait for the answer",
		"Which deliverable do you want: Project repository",
		"Personal agent",
	} {
		if !strings.Contains(content, needle) {
			t.Fatalf("builder source SKILL.md is missing coverage/routing contract text %q", needle)
		}
	}
	if got := strings.Count(content, "Which deliverable do you want:"); got != 1 {
		t.Fatalf("ambiguous-deliverable question count = %d, want exactly 1", got)
	}
	precheck := "use the `agentfield-use` skill's health, discovery, and reasoner-capability search flow exactly once"
	if got := strings.Count(content, precheck); got != 1 {
		t.Fatalf("installed-agent coverage pre-check directive count = %d, want exactly 1", got)
	}
	precheckAt := strings.Index(content, precheck)
	hardGateAt := strings.Index(content, "## Hard gate")
	if precheckAt < 0 || hardGateAt < 0 || precheckAt >= hardGateAt {
		t.Fatalf("coverage pre-check must precede hard gate: precheck=%d hard-gate=%d", precheckAt, hardGateAt)
	}
}

func TestAgentfieldBuilderPersonalLifecycleAndSecretSafetyContract(t *testing.T) {
	content := string(skillSource(t, "agentfield"))
	for _, needle := range []string{
		"filesystem-safe kebab-case package/name/node ID",
		"~/agentfield-agents/<name>",
		"Do not author in a temporary directory, a disposable checkout, or the generated `~/.agentfield` installation copy.",
		"`config_version: v1`",
		"distinct from the agent release `version`",
		"`entrypoint.start`",
		"`entrypoint.healthcheck: /health`",
		"`agent_node.node_id` equal to `<name>`",
		"`agent_node.default_port`",
		"only install dependencies the source needs",
		"`user_environment`",
		"actionable `description`",
		"`type: secret`",
		"`scope: global`",
		"`scope: node`",
		"Do not declare invented keys.",
		"`af install ~/agentfield-agents/<name>`",
		"`af secrets set KEY`",
		"`af secrets set --node <name> KEY`",
		"`af run <name>`",
		"GET ${AGENTFIELD_SERVER:-http://localhost:8080}/api/v1/nodes",
		"registered in an active/healthy state",
		"Invoke the public entry reasoner through the control plane",
		"terminal successful result",
		"Diagnose and safely retry correctable failures",
		"installation, secret setup, startup, registration, or invocation",
		"blocking handoff",
		"Never invent, echo, commit, put into `agentfield-package.yaml`, or include secret values in a handoff.",
		"Do not claim completion until healthy registration and a live reasoner result both succeed.",
		"now appears in the AgentField Desktop app",
		"declared keys are presented as a form",
		"auto-start toggle",
		"`af stop <name> && af run <name>`",
		"stop (`af stop <name>`)",
		"`af logs <name>`",
		"`af install ~/agentfield-agents/<name>` followed by `af run <name>`",
	} {
		if !strings.Contains(content, needle) {
			t.Fatalf("builder source SKILL.md is missing personal lifecycle/safety text %q", needle)
		}
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
		"Project repository output contract",
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
