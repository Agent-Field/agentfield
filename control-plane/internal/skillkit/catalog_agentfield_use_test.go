package skillkit

import (
	"strings"
	"testing"
)

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
