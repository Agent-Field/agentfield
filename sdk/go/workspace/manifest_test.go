package workspace

import (
	"encoding/json"
	"testing"
)

func TestCanonicalJSONSortsKeysAndFiles(t *testing.T) {
	m := &Manifest{
		Version: 1,
		Files: []FileEntry{
			{Path: "src/b.py", Size: 2, Mode: 420, MtimeNS: 20, SHA256: "bbb"},
			{Path: "src/a.py", Size: 1, Mode: 420, MtimeNS: 10, SHA256: "aaa"},
		},
	}
	got, err := CanonicalJSON(m)
	if err != nil {
		t.Fatalf("CanonicalJSON: %v", err)
	}
	want := `{"files":[{"mode":420,"mtime_ns":10,"path":"src/a.py","sha256":"aaa","size":1},{"mode":420,"mtime_ns":20,"path":"src/b.py","sha256":"bbb","size":2}],"version":1}`
	if string(got) != want {
		t.Fatalf("canonical JSON mismatch:\n got: %s\nwant: %s", got, want)
	}
}

func TestManifestIDStableAcrossFileOrder(t *testing.T) {
	a := &Manifest{
		Version: 1,
		Files: []FileEntry{
			{Path: "a.txt", Size: 1, Mode: 420, MtimeNS: 5, SHA256: "h1"},
			{Path: "b.txt", Size: 2, Mode: 420, MtimeNS: 6, SHA256: "h2"},
			{Path: "c.txt", Size: 3, Mode: 420, MtimeNS: 7, SHA256: "h3"},
		},
	}
	b := &Manifest{
		Version: 1,
		Files: []FileEntry{
			{Path: "c.txt", Size: 3, Mode: 420, MtimeNS: 7, SHA256: "h3"},
			{Path: "a.txt", Size: 1, Mode: 420, MtimeNS: 5, SHA256: "h1"},
			{Path: "b.txt", Size: 2, Mode: 420, MtimeNS: 6, SHA256: "h2"},
		},
	}
	idA, err := ManifestID(a)
	if err != nil {
		t.Fatalf("ManifestID a: %v", err)
	}
	idB, err := ManifestID(b)
	if err != nil {
		t.Fatalf("ManifestID b: %v", err)
	}
	if idA != idB {
		t.Fatalf("manifest_id must be order-independent: %s != %s", idA, idB)
	}
	if len(idA) != 64 {
		t.Fatalf("manifest_id must be a 64-char hex sha256, got %d chars", len(idA))
	}
}

func TestCanonicalJSONDoesNotMutateInput(t *testing.T) {
	m := &Manifest{
		Version: 1,
		Files: []FileEntry{
			{Path: "z.txt", SHA256: "z"},
			{Path: "a.txt", SHA256: "a"},
		},
	}
	if _, err := CanonicalJSON(m); err != nil {
		t.Fatalf("CanonicalJSON: %v", err)
	}
	if m.Files[0].Path != "z.txt" {
		t.Fatalf("CanonicalJSON must not reorder the caller's slice, got %s first", m.Files[0].Path)
	}
}

func TestManifestJSONRoundTrip(t *testing.T) {
	m := &Manifest{Version: 1, Files: []FileEntry{{Path: "a.py", Size: 123, Mode: 420, MtimeNS: 170, SHA256: "deadbeef"}}}
	encoded, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded Manifest
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(decoded.Files) != 1 || decoded.Files[0].Size != 123 || decoded.Files[0].MtimeNS != 170 {
		t.Fatalf("round-trip lost fields: %+v", decoded)
	}
}
