// Package skillkit owns the skill catalog: it embeds skill content into the
// af binary, installs / uninstalls / lists skills against multiple coding-agent
// targets (Claude Code, Codex, Gemini, OpenCode, Aider, Windsurf, Cursor),
// and tracks state in ~/.agentfield/skills/.state.json.
//
// The canonical on-disk layout (after `af skill install`) is:
//
//	~/.agentfield/skills/
//	├── .state.json                                # tracking
//	└── <skill-name>/
//	    ├── current → ./<active-version>/         # symlink
//	    └── <version>/                            # versioned store
//	        ├── SKILL.md
//	        └── references/
//	            └── ...
//
// Each target then either:
//   - symlinks into ~/.<agent>/skills/<name> (Claude Code style), OR
//   - appends a marker block to the agent's global rules file pointing at the
//     canonical SKILL.md path so updates flow through automatically.
//
// New skills are added by dropping a directory into skill_data/ and registering
// it in catalog.go. The skill content is embedded at build time via go:embed
// (the source-of-truth lives in repo-root skills/<name>/ — keep them in sync
// via scripts/sync-embedded-skills.sh).
package skillkit

import "embed"

// SkillData is the embedded filesystem containing the source-of-truth content
// for every shipped skill. Files live under skill_data/<skill-name>/ and are
// copied from the repo-root skills/ directory at build time.
//
//go:embed skill_data/agentfield-multi-reasoner-builder/SKILL.md
//go:embed skill_data/agentfield-multi-reasoner-builder/references/*.md
//go:embed skill_data/agentfield-multi-reasoner-builder/commands/*.md
var SkillData embed.FS
