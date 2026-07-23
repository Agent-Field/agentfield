package workspace

import "testing"

// TestCanonicalBytesVector pins the exact canonical JSON bytes and manifest_id
// for a known manifest. The same vector is asserted byte-for-byte by the Python
// SDK and the control plane, so this test is the cross-language contract that
// proves every implementation hashes an identical manifest to an identical id.
//
// The expected values were produced independently by the Python canonicalizer
// (json.dumps(..., sort_keys=True, separators=(",",":"))) — see
// docs/design/workspace-artifacts.md.
func TestCanonicalBytesVector(t *testing.T) {
	// Files are supplied out of order on purpose; canonicalization must sort
	// them by path before hashing.
	m := &Manifest{
		Version: 1,
		Files: []FileEntry{
			{Path: "src/b.py", Size: 2, Mode: 420, MtimeNS: 20, SHA256: "bbb"},
			{Path: "src/a.py", Size: 1, Mode: 420, MtimeNS: 10, SHA256: "aaa"},
		},
	}

	const wantBytes = `{"files":[{"mode":420,"mtime_ns":10,"path":"src/a.py","sha256":"aaa","size":1},{"mode":420,"mtime_ns":20,"path":"src/b.py","sha256":"bbb","size":2}],"version":1}`
	const wantID = "22f593f17ad084c3a7a76fb28f15936715f80e5091a0f9970bcd0c2b7736e204"

	got, err := CanonicalJSON(m)
	if err != nil {
		t.Fatalf("CanonicalJSON: %v", err)
	}
	if string(got) != wantBytes {
		t.Fatalf("canonical bytes mismatch:\n got: %s\nwant: %s", got, wantBytes)
	}

	id, err := ManifestID(m)
	if err != nil {
		t.Fatalf("ManifestID: %v", err)
	}
	if id != wantID {
		t.Fatalf("manifest_id mismatch:\n got: %s\nwant: %s", id, wantID)
	}
}
