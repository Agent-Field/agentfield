package skillkit

import (
	"strings"
	"testing"
)

// Contract: the agentfield-personal skill is registered and its content is
// really embedded — a missing go:embed directive would only surface at
// install time as "embedded skill is empty" without this guard.
func TestAgentfieldPersonalSkillEmbedded(t *testing.T) {
	skill, err := CatalogByName("agentfield-personal")
	if err != nil {
		t.Fatalf("CatalogByName: %v", err)
	}
	if skill.Version == "" || skill.Trigger == "" {
		t.Fatalf("agentfield-personal catalog entry incomplete: %+v", skill)
	}

	files, err := skill.EnumerateFiles()
	if err != nil {
		t.Fatalf("EnumerateFiles: %v", err)
	}
	if _, ok := files[skill.EntryFile]; !ok {
		t.Fatalf("entry file %q not embedded (got %d files)", skill.EntryFile, len(files))
	}
}

func TestAgentfieldPersonalSourceFrontmatterContract(t *testing.T) {
	content := skillSource(t, "agentfield-personal")
	frontmatter, err := parseSkillFrontmatter(content)
	if err != nil {
		t.Fatalf("parse source frontmatter: %v", err)
	}
	if frontmatter.Name != "agentfield-personal" || frontmatter.Version != "0.1.0" {
		t.Fatalf("source frontmatter = %+v, want name=agentfield-personal version=0.1.0", frontmatter)
	}
}

// The personal-agent deliverable contract: stable authoring source, a v1
// manifest the installer parses, scoped secrets handled only through the CLI,
// and completion claimed only on healthy registration plus a live call.
func TestAgentfieldPersonalLifecycleAndSecretSafetyContract(t *testing.T) {
	content := string(skillSource(t, "agentfield-personal"))
	for _, needle := range []string{
		"filesystem-safe kebab-case",
		"~/agentfield-agents/<name>",
		"Do not author in a temporary directory",
		"disposable checkout, or the generated `~/.agentfield` installation copy",
		"`config_version: v1`",
		"from the agent release `version`",
		"`entrypoint.start`",
		"`entrypoint.healthcheck: /health`",
		"`agent_node.node_id` equal to `<name>`",
		"`agent_node.default_port`",
		"only install dependencies the source needs",
		"`user_environment`",
		"`type: secret`",
		"`scope: global`",
		"`scope: node`",
		"Do not declare invented keys.",
		"`af install ~/agentfield-agents/<name>`",
		"`af secrets set KEY`",
		"`af secrets set --node <name> KEY`",
		"Never invent, echo, commit, put into",
		"`agentfield-package.yaml`, or include secret values in a handoff.",
		"`af run <name>`",
		"GET ${AGENTFIELD_SERVER:-http://localhost:8080}/api/v1/nodes",
		"registered in an active/healthy state",
		"Invoke the public entry reasoner through the control plane",
		"terminal successful result",
		"Diagnose and safely retry correctable",
		"blocking handoff",
		"healthy registration and a live reasoner result both succeed.",
		"appears in the AgentField Desktop app",
		"presented as a form",
		"auto-start toggle",
		"`af stop <name> && af run <name>`",
		"stop (`af stop <name>`)",
		"`af logs <name>`",
		"`af install ~/agentfield-agents/<name>` followed by `af run <name>`",
	} {
		if !strings.Contains(content, needle) {
			t.Fatalf("agentfield-personal SKILL.md is missing lifecycle/safety text %q", needle)
		}
	}
}

// Duplicate avoidance is a pre-build courtesy, not a routing protocol: check
// installed coverage once, offer the existing agent, but an explicit build
// request always wins.
func TestAgentfieldPersonalCoverageCourtesyContract(t *testing.T) {
	content := string(skillSource(t, "agentfield-personal"))
	for _, needle := range []string{
		"Check once whether an installed agent already covers the request",
		"unless the user explicitly asked to build a new or",
		"offer to start it with `af run <name>`",
	} {
		if !strings.Contains(content, needle) {
			t.Fatalf("agentfield-personal SKILL.md is missing coverage-courtesy text %q", needle)
		}
	}
	if strings.Contains(content, "coverage_precheck_complete") {
		t.Fatal("agentfield-personal must not reintroduce the cross-skill routing marker")
	}
}
