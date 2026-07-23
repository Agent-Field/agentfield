package skillkit

import (
	"bytes"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// TestSkillCatalogAndEmbeddedMirrorsStayAligned protects the three copies of
// shipped-skill metadata: catalog, canonical source, and Go-embedded mirror.
func TestSkillCatalogAndEmbeddedMirrorsStayAligned(t *testing.T) {
	tests := []struct {
		name    string
		version string
	}{
		{name: "agentfield", version: "0.5.1"},
		{name: "agentfield-personal", version: "0.1.0"},
		{name: "agentfield-use", version: "0.4.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			skill, err := CatalogByName(tt.name)
			if err != nil {
				t.Fatalf("CatalogByName(%q): %v", tt.name, err)
			}
			if skill.Name != tt.name || skill.Version != tt.version {
				t.Fatalf("catalog skill = %+v, want name=%q version=%q", skill, tt.name, tt.version)
			}

			canonical := skillSource(t, tt.name)
			sourceFrontmatter, err := parseSkillFrontmatter(canonical)
			if err != nil {
				t.Fatalf("parse canonical frontmatter: %v", err)
			}
			if sourceFrontmatter.Name != skill.Name || sourceFrontmatter.Version != skill.Version {
				t.Fatalf("canonical frontmatter = %+v, want name=%q version=%q", sourceFrontmatter, skill.Name, skill.Version)
			}

			embedded, err := skill.EntryContent()
			if err != nil {
				t.Fatalf("read embedded entry: %v", err)
			}
			embeddedFrontmatter, err := parseSkillFrontmatter(embedded)
			if err != nil {
				t.Fatalf("parse embedded frontmatter: %v", err)
			}
			if embeddedFrontmatter.Name != skill.Name || embeddedFrontmatter.Version != skill.Version {
				t.Fatalf("embedded frontmatter = %+v, want name=%q version=%q", embeddedFrontmatter, skill.Name, skill.Version)
			}

			compareSkillTrees(t, skillSourceDirectory(t, tt.name), skillEmbeddedDirectory(t, tt.name))
		})
	}
}

func TestDeprecatedBuilderAliasRemainsOnlyAnAlias(t *testing.T) {
	const alias = "agentfield-multi-reasoner-builder"
	skill, err := CatalogByName(alias)
	if err != nil {
		t.Fatalf("CatalogByName(%q): %v", alias, err)
	}
	if skill.Name != "agentfield" {
		t.Fatalf("CatalogByName(%q) = %q, want canonical agentfield", alias, skill.Name)
	}
	for _, catalogSkill := range Catalog {
		if catalogSkill.Name == alias {
			t.Fatalf("deprecated alias %q must not be a standalone catalog skill", alias)
		}
	}
	if _, err := os.Stat(skillEmbeddedDirectory(t, alias)); !os.IsNotExist(err) {
		t.Fatalf("deprecated embedded alias directory exists or cannot be checked: %v", err)
	}
}

// Keep the repository's generation check executable from the Go suite as well
// as comparing the resulting trees above. This catches a future mismatch
// between the catalog's shipped skills and the sync script's mirror list.
func TestEmbeddedSkillSyncCheck(t *testing.T) {
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skipf("bash unavailable; sync checker is exercised in Bash-capable CI: %v", err)
	}
	repoRoot := filepath.Dir(filepath.Dir(skillSourceDirectory(t, "agentfield")))
	command := exec.Command(bash, "scripts/sync-embedded-skills.sh", "--check")
	command.Dir = repoRoot
	output, err := command.CombinedOutput()
	if err != nil {
		if strings.Contains(string(output), "execvpe(/bin/bash) failed") {
			t.Skipf("Bash launcher is present but its WSL runtime is unavailable; sync checker is exercised in Bash-capable CI: %s", output)
		}
		t.Fatalf("scripts/sync-embedded-skills.sh --check failed: %v\n%s", err, output)
	}
	if !strings.Contains(string(output), "All embedded skills are in sync with sources.") {
		t.Fatalf("scripts/sync-embedded-skills.sh --check output = %q, want success confirmation", output)
	}
}

func skillSourceDirectory(t *testing.T, name string) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate skillkit test directory")
	}
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "skills", name)
}

func skillEmbeddedDirectory(t *testing.T, name string) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate skillkit test directory")
	}
	return filepath.Join(filepath.Dir(file), "skill_data", name)
}

func compareSkillTrees(t *testing.T, canonical, embedded string) {
	t.Helper()
	canonicalFiles := skillTreeFiles(t, canonical)
	embeddedFiles := skillTreeFiles(t, embedded)

	if got, want := sortedSkillPaths(canonicalFiles), sortedSkillPaths(embeddedFiles); !equalStrings(got, want) {
		t.Fatalf("skill tree paths differ:\ncanonical: %v\nembedded:  %v", got, want)
	}
	for relativePath, canonicalContent := range canonicalFiles {
		if !bytes.Equal(canonicalContent, embeddedFiles[relativePath]) {
			t.Fatalf("skill tree file %q differs between canonical and embedded mirrors", relativePath)
		}
	}
}

func skillTreeFiles(t *testing.T, root string) map[string][]byte {
	t.Helper()
	files := make(map[string][]byte)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		relativePath, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		files[filepath.ToSlash(relativePath)] = content
		return nil
	})
	if err != nil {
		t.Fatalf("walk skill tree %q: %v", root, err)
	}
	return files
}

func sortedSkillPaths(files map[string][]byte) []string {
	paths := make([]string, 0, len(files))
	for path := range files {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths
}

func equalStrings(left, right []string) bool {
	return strings.Join(left, "\x00") == strings.Join(right, "\x00")
}
