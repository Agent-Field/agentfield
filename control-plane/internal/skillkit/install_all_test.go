package skillkit

import "testing"

// TestInstallAllInstallsEveryCatalogSkill is the contract for `af skill install`
// with no skill name: it must install every catalog skill — both the build
// skill (agentfield) and the drive skill (agentfield-use) — not just the first
// catalog entry. Uses the codex marker-block target so no symlink-only path is
// required and everything lands under an isolated temp HOME.
func TestInstallAllInstallsEveryCatalogSkill(t *testing.T) {
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

func keysOf(m map[string]InstalledSkill) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
