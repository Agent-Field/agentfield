package packages

import (
	"strings"
	"testing"
)

// Validation contract for the manifest schema version (config_version):
//   - absent  -> v0 (legacy) and parses fine
//   - "v1"    -> the current version, parses fine
//   - a version newer than this binary understands is refused with a clear error
//   - a malformed version fails loudly (not silently treated as v0)
//   - the "v" prefix is optional/case-insensitive
//   - a manifest at the current version tolerates fields the reader doesn't know
//     (additive changes must not require a version bump)

func TestParseConfigVersion(t *testing.T) {
	cases := []struct {
		raw     string
		want    int
		wantErr bool
	}{
		{"", 0, false},    // absent -> v0
		{"  ", 0, false},  // whitespace -> v0
		{"v0", 0, false},  // explicit v0
		{"v1", 1, false},  // current
		{"V1", 1, false},  // case-insensitive prefix
		{"1", 1, false},   // bare integer accepted
		{"v2", 2, false},  // parses (rejection happens against CurrentConfigVersion)
		{"v1.0", 0, true}, // not an int form
		{"latest", 0, true},
		{"v-1", 0, true},
	}
	for _, c := range cases {
		got, err := parseConfigVersion(c.raw)
		if c.wantErr {
			if err == nil {
				t.Errorf("parseConfigVersion(%q): expected error, got %d", c.raw, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseConfigVersion(%q): unexpected error: %v", c.raw, err)
			continue
		}
		if got != c.want {
			t.Errorf("parseConfigVersion(%q) = %d, want %d", c.raw, got, c.want)
		}
	}
}

func TestParsePackageMetadataConfigVersion(t *testing.T) {
	t.Run("absent config_version reads as v0", func(t *testing.T) {
		dir := t.TempDir()
		writeTestPackage(t, dir, "name: legacy-node\nversion: 0.1.0\n")
		md, err := ParsePackageMetadata(dir)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := md.ConfigVersionNumber(); got != 0 {
			t.Fatalf("ConfigVersionNumber() = %d, want 0", got)
		}
	})

	t.Run("explicit v1 accepted", func(t *testing.T) {
		dir := t.TempDir()
		writeTestPackage(t, dir, "config_version: v1\nname: n\nversion: 0.1.0\n")
		md, err := ParsePackageMetadata(dir)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := md.ConfigVersionNumber(); got != 1 {
			t.Fatalf("ConfigVersionNumber() = %d, want 1", got)
		}
	})

	t.Run("future version rejected with a helpful error", func(t *testing.T) {
		dir := t.TempDir()
		writeTestPackage(t, dir, "config_version: v99\nname: n\nversion: 0.1.0\n")
		_, err := ParsePackageMetadata(dir)
		if err == nil {
			t.Fatal("expected error for future config_version, got nil")
		}
		if !strings.Contains(err.Error(), "upgrade AgentField") {
			t.Fatalf("error should tell the user to upgrade, got: %v", err)
		}
	})

	t.Run("malformed version fails loudly", func(t *testing.T) {
		dir := t.TempDir()
		writeTestPackage(t, dir, "config_version: latest\nname: n\nversion: 0.1.0\n")
		if _, err := ParsePackageMetadata(dir); err == nil {
			t.Fatal("expected error for malformed config_version, got nil")
		}
	})

	t.Run("additive unknown fields do not break the current-version reader", func(t *testing.T) {
		dir := t.TempDir()
		// A field this reader has never heard of must be ignored, not fatal —
		// additive changes are allowed without a config_version bump.
		writeTestPackage(t, dir, "config_version: v1\nname: n\nversion: 0.1.0\nsome_future_key: hello\n")
		md, err := ParsePackageMetadata(dir)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if md.Name != "n" {
			t.Fatalf("Name = %q, want n", md.Name)
		}
	})
}
