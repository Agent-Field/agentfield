package skillkit

import (
	"fmt"
	"io/fs"
	"path"
	"strings"
)

// Skill describes a skill that ships with the af binary. The catalog below
// is the only place new skills get registered. Bump Version on every change
// so `af skill update` knows there's a new build.
type Skill struct {
	Name        string // canonical skill name (kebab-case, used as directory name)
	Version     string // semver-ish version string baked into the binary
	Description string // one-line description for `af skill list`
	EmbedRoot   string // root path inside SkillData where this skill's files live
	EntryFile   string // relative path to the skill's main file (usually SKILL.md)
}

// Catalog is the registry of every skill the binary ships. Add a new entry
// here when adding a new skill, and drop the source files into
// skill_data/<name>/ so the embed picks them up.
var Catalog = []Skill{
	{
		Name:        "agentfield-multi-reasoner-builder",
		Version:     "0.3.0",
		Description: "Architect and ship complete multi-agent backends on AgentField — composite intelligence from five foundational principles, deep dynamic call graphs, async-first smoke tests, and hard runtime-contract rules.",
		EmbedRoot:   "skill_data/agentfield-multi-reasoner-builder",
		EntryFile:   "SKILL.md",
	},
}

// CatalogByName returns the skill with the given name, or an error if it
// is not in the registry.
func CatalogByName(name string) (Skill, error) {
	for _, s := range Catalog {
		if s.Name == name {
			return s, nil
		}
	}
	available := make([]string, len(Catalog))
	for i, s := range Catalog {
		available[i] = s.Name
	}
	return Skill{}, fmt.Errorf("skill %q not found in the af binary catalog (available: %s)", name, strings.Join(available, ", "))
}

// EnumerateFiles walks the embedded skill data and returns every file path
// relative to the skill's EmbedRoot, paired with its raw bytes. Used by the
// installer to write the canonical on-disk copy.
func (s Skill) EnumerateFiles() (map[string][]byte, error) {
	files := make(map[string][]byte)
	err := fs.WalkDir(SkillData, s.EmbedRoot, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		rel, err := relativeUnderEmbed(s.EmbedRoot, p)
		if err != nil {
			return err
		}
		data, err := fs.ReadFile(SkillData, p)
		if err != nil {
			return fmt.Errorf("read embedded %s: %w", p, err)
		}
		files[rel] = data
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("enumerate embedded skill %q: %w", s.Name, err)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("embedded skill %q is empty — did the embed directive in embed.go include this skill's files?", s.Name)
	}
	return files, nil
}

// EntryContent returns the raw bytes of the skill's entry file (SKILL.md).
// Used by `af skill install --print` and by Cursor's clipboard fallback.
func (s Skill) EntryContent() ([]byte, error) {
	return fs.ReadFile(SkillData, path.Join(s.EmbedRoot, s.EntryFile))
}

func relativeUnderEmbed(root, p string) (string, error) {
	rootSlash := strings.TrimSuffix(root, "/") + "/"
	if !strings.HasPrefix(p, rootSlash) {
		return "", fmt.Errorf("path %q is not under embed root %q", p, root)
	}
	return strings.TrimPrefix(p, rootSlash), nil
}
