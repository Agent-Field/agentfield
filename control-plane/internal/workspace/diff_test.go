package workspace

import (
	"os"
	"path/filepath"
	"testing"
)

func TestComputeDiffChangedAndDeleted(t *testing.T) {
	src := t.TempDir()
	cas := NewCAS(t.TempDir())
	writeFile(t, src, "keep.txt", "same", 0o644)
	writeFile(t, src, "edit.txt", "before", 0o644)
	writeFile(t, src, "gone.txt", "remove me", 0o644)

	input, err := Seal(src, cas)
	if err != nil {
		t.Fatalf("seal: %v", err)
	}

	// Simulate a reasoner mutating the workspace.
	writeFile(t, src, "edit.txt", "after", 0o644)
	writeFile(t, src, "new.txt", "brand new", 0o644)
	if err := os.Remove(filepath.Join(src, "gone.txt")); err != nil {
		t.Fatalf("remove: %v", err)
	}

	diff, err := ComputeDiff(input, src, cas)
	if err != nil {
		t.Fatalf("ComputeDiff: %v", err)
	}

	changed := map[string]bool{}
	for _, c := range diff.Changed {
		changed[c.Path] = true
	}
	if !changed["edit.txt"] || !changed["new.txt"] {
		t.Errorf("expected edit.txt and new.txt changed, got %+v", diff.Changed)
	}
	if changed["keep.txt"] {
		t.Errorf("keep.txt is unchanged and must not be in changed")
	}
	if len(diff.Deleted) != 1 || diff.Deleted[0] != "gone.txt" {
		t.Errorf("expected gone.txt deleted, got %+v", diff.Deleted)
	}
}

func TestApplyWritesAndDeletes(t *testing.T) {
	// Original caller dir (what was sealed).
	orig := t.TempDir()
	cas := NewCAS(t.TempDir())
	writeFile(t, orig, "a.txt", "original-a", 0o644)
	writeFile(t, orig, "del.txt", "to-delete", 0o644)
	inputManifest, err := Seal(orig, cas)
	if err != nil {
		t.Fatalf("seal: %v", err)
	}

	// Node produced new content for a.txt and a brand-new file; deleted del.txt.
	newA := "changed-a"
	shaA, _ := cas.PutBytes([]byte(newA))
	newB := "created-b"
	shaB, _ := cas.PutBytes([]byte(newB))
	diff := &Diff{
		Changed: []ChangedFile{
			{Path: "a.txt", SHA256: shaA, Size: int64(len(newA)), Mode: 0o644},
			{Path: "b.txt", SHA256: shaB, Size: int64(len(newB)), Mode: 0o644},
		},
		Deleted: []string{"del.txt"},
	}

	res, err := Apply(orig, inputManifest, diff, cas, false)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if len(res.Conflicts) != 0 {
		t.Fatalf("unexpected conflicts: %+v", res.Conflicts)
	}
	assertFile(t, orig, "a.txt", newA)
	assertFile(t, orig, "b.txt", newB)
	if _, err := os.Stat(filepath.Join(orig, "del.txt")); !os.IsNotExist(err) {
		t.Errorf("del.txt should have been removed")
	}
}

func TestApplyConflictSkipAndForce(t *testing.T) {
	orig := t.TempDir()
	cas := NewCAS(t.TempDir())
	writeFile(t, orig, "a.txt", "original-a", 0o644)
	writeFile(t, orig, "del.txt", "sealed-del", 0o644)
	inputManifest, err := Seal(orig, cas)
	if err != nil {
		t.Fatalf("seal: %v", err)
	}

	// The user edits both files locally AFTER sealing — now they differ from
	// what was sealed, so apply must treat them as conflicts.
	writeFile(t, orig, "a.txt", "user-local-edit", 0o644)
	writeFile(t, orig, "del.txt", "user-local-edit", 0o644)

	newA := "node-changed-a"
	shaA, _ := cas.PutBytes([]byte(newA))
	diff := &Diff{
		Changed: []ChangedFile{{Path: "a.txt", SHA256: shaA, Size: int64(len(newA)), Mode: 0o644}},
		Deleted: []string{"del.txt"},
	}

	res, err := Apply(orig, inputManifest, diff, cas, false)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	conflicts := map[string]bool{}
	for _, c := range res.Conflicts {
		conflicts[c] = true
	}
	if !conflicts["a.txt"] || !conflicts["del.txt"] {
		t.Fatalf("expected a.txt and del.txt conflicts, got %+v", res.Conflicts)
	}
	// Files must be untouched on conflict-skip.
	assertFile(t, orig, "a.txt", "user-local-edit")
	assertFile(t, orig, "del.txt", "user-local-edit")

	// --force overrides conflicts.
	res, err = Apply(orig, inputManifest, diff, cas, true)
	if err != nil {
		t.Fatalf("Apply force: %v", err)
	}
	if len(res.Conflicts) != 0 {
		t.Fatalf("force must clear conflicts, got %+v", res.Conflicts)
	}
	assertFile(t, orig, "a.txt", newA)
	if _, err := os.Stat(filepath.Join(orig, "del.txt")); !os.IsNotExist(err) {
		t.Errorf("del.txt should have been force-removed")
	}
}

func assertFile(t *testing.T, dir, rel, want string) {
	t.Helper()
	got, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatalf("read %s: %v", rel, err)
	}
	if string(got) != want {
		t.Errorf("%s content = %q, want %q", rel, got, want)
	}
}

func TestStagedRecordMerge(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	runID := "run-123"

	// CLI-side contribution.
	if err := MergeStaged(runID, func(r *StagedRecord) {
		r.OriginalDir = "/tmp/project"
		r.ManifestID = "abc"
		r.InputManifest = &Manifest{Version: 1}
	}); err != nil {
		t.Fatalf("merge cli: %v", err)
	}
	// Control-plane-side contribution.
	if err := MergeStaged(runID, func(r *StagedRecord) {
		r.Diff = &Diff{Changed: []ChangedFile{{Path: "x.txt", SHA256: "h"}}}
		r.NodeBaseURL = "http://node:9000"
		r.NodeID = "node-a"
	}); err != nil {
		t.Fatalf("merge cp: %v", err)
	}

	rec, err := LoadStaged(runID)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if rec.OriginalDir != "/tmp/project" || rec.ManifestID != "abc" || rec.InputManifest == nil {
		t.Errorf("CLI contribution lost: %+v", rec)
	}
	if rec.Diff == nil || rec.NodeBaseURL != "http://node:9000" || rec.NodeID != "node-a" {
		t.Errorf("control-plane contribution lost: %+v", rec)
	}
}
