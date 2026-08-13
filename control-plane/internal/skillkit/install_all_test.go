package skillkit

import (
	"errors"
	"strings"
	"testing"
)

// TestInstallAllInstallsEveryCatalogSkill is the contract for `af skill install`
// with no skill name: it must install every catalog skill — both the build
// skill (agentfield) and the drive skill (agentfield-use) — not just the first
// catalog entry. Uses the codex marker-block target so no symlink-only path is
// required and everything lands under an isolated temp HOME.
func TestInstallAllInstallsEveryCatalogSkill(t *testing.T) {
	t.Setenv("AGENTFIELD_SKIP_FURROW", "1")
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("AGENTFIELD_HOME", home)

	reports, err := InstallAll(InstallOptions{
		Targets: []string{"codex"},
		Force:   true,
	})
	if err != nil {
		t.Fatalf("InstallAll returned error: %v", err)
	}
	if len(reports) != len(Catalog) {
		t.Fatalf("expected %d reports (one per catalog skill), got %d", len(Catalog), len(reports))
	}

	state, err := LoadState()
	if err != nil {
		t.Fatalf("LoadState returned error: %v", err)
	}

	for _, want := range []string{"agentfield", "agentfield-use"} {
		installed, ok := state.Skills[want]
		if !ok {
			t.Fatalf("skill %q was not installed; state has %v", want, keysOf(state.Skills))
		}
		if _, ok := installed.Targets["codex"]; !ok {
			t.Errorf("skill %q was not installed into the codex target", want)
		}
	}
}

// Contract: one unusable skill must not cost the user every skill after it in
// the catalog. InstallAll keeps going and returns the failures together.
func TestInstallAllContinuesPastAFailingSkill(t *testing.T) {
	t.Setenv("AGENTFIELD_SKIP_FURROW", "1")
	withTempHome(t)

	origCatalog := Catalog
	t.Cleanup(func() { Catalog = origCatalog })
	// A skill whose embedded files are missing fails at the canonical write —
	// the first thing install does — so nothing downstream of it can run.
	broken := Skill{
		Name:      "broken-skill",
		Version:   "1.0.0",
		EmbedRoot: "skill_data/missing",
		EntryFile: "SKILL.md",
	}
	Catalog = append([]Skill{broken}, origCatalog...)

	origTargets := allTargets
	t.Cleanup(func() { allTargets = origTargets })
	success := &fakeTarget{name: "success", displayName: "Success", method: "marker-block", detected: true, path: fakeTargetPath(t, "success")}
	allTargets = []Target{success}

	reports, err := InstallAll(InstallOptions{Targets: []string{"success"}, Force: true})
	if err == nil {
		t.Fatal("InstallAll must report the failing skill")
	}
	if !strings.Contains(err.Error(), "broken-skill") {
		t.Fatalf("error does not name the failing skill: %v", err)
	}
	if len(reports) != len(Catalog)-1 {
		t.Fatalf("got %d reports, want one for every skill except the broken one (%d)", len(reports), len(Catalog)-1)
	}
	for i, report := range reports {
		if want := Catalog[i+1].Name; report.Skill.Name != want {
			t.Fatalf("report %d is for %q, want %q — the remaining skills must still install in catalog order", i, report.Skill.Name, want)
		}
	}
}

// Contract: `af skill install` exits non-zero when a target fails. The report
// is the user-facing detail; the error is what makes a launch-time install in
// the desktop app (or CI) notice that nothing landed.
func TestInstallReportsSurfaceTargetFailuresAsErrors(t *testing.T) {
	t.Setenv("AGENTFIELD_SKIP_FURROW", "1")
	withTempHome(t)

	origTargets := allTargets
	t.Cleanup(func() { allTargets = origTargets })
	failing := &fakeTarget{
		name: "failing", displayName: "Failing", method: "marker-block", detected: true,
		path: fakeTargetPath(t, "failing"), installErr: errors.New("disk on fire"),
	}
	allTargets = []Target{failing}

	report, err := Install(InstallOptions{SkillName: "agentfield", Targets: []string{"failing"}, Force: true})
	if err != nil {
		t.Fatalf("Install returned a hard error: %v", err)
	}
	reportErr := report.ReportError()
	if reportErr == nil {
		t.Fatal("a report with a failed target must produce an error")
	}
	for _, needle := range []string{"agentfield", "failing", "disk on fire"} {
		if !strings.Contains(reportErr.Error(), needle) {
			t.Fatalf("report error %q does not mention %q", reportErr, needle)
		}
	}

	reports, err := InstallAll(InstallOptions{Targets: []string{"failing"}, Force: true})
	if err != nil {
		t.Fatalf("InstallAll returned a hard error: %v", err)
	}
	batchErr := ReportsError(reports)
	if batchErr == nil {
		t.Fatal("ReportsError must surface per-target failures across the catalog")
	}
	for _, skill := range Catalog {
		if !strings.Contains(batchErr.Error(), skill.Name) {
			t.Fatalf("batch error %q does not mention failed skill %q", batchErr, skill.Name)
		}
	}

	// A clean report is not an error.
	if err := (&InstallReport{}).ReportError(); err != nil {
		t.Fatalf("clean report produced an error: %v", err)
	}
	if err := ReportsError(nil); err != nil {
		t.Fatalf("empty batch produced an error: %v", err)
	}
}

func keysOf(m map[string]InstalledSkill) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
