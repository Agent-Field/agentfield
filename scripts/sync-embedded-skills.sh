#!/usr/bin/env bash
# sync-embedded-skills.sh — keep the Go embed copy of every shipped skill in
# sync with the canonical source-of-truth files in skills/.
#
# The af binary embeds skill content at build time via go:embed in
# control-plane/internal/skillkit/embed.go. The embed directive can only
# reach files inside the skillkit package, so we maintain a mirror at
# control-plane/internal/skillkit/skill_data/<skill-name>/ that is bytewise
# identical to skills/<skill-name>/.
#
# Run this script whenever you edit a skill in skills/ before committing,
# or before running `go build` if you've made local edits. The Makefile's
# build target should also call this.
#
# Usage:
#   ./scripts/sync-embedded-skills.sh           # sync all shipped skills
#   ./scripts/sync-embedded-skills.sh --check   # exit non-zero if out of sync (CI)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/skills"
EMBED_DIR="${REPO_ROOT}/control-plane/internal/skillkit/skill_data"

# Skills to mirror. Add new skills here when they're added to the catalog.
SKILLS=(
  "agentfield-multi-reasoner-builder"
)

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "ERROR: source directory $SOURCE_DIR does not exist" >&2
  exit 1
fi

mkdir -p "$EMBED_DIR"

drift_found=0

for skill in "${SKILLS[@]}"; do
  src="${SOURCE_DIR}/${skill}"
  dst="${EMBED_DIR}/${skill}"

  if [[ ! -d "$src" ]]; then
    echo "ERROR: skill source $src does not exist" >&2
    exit 1
  fi

  if [[ "$CHECK_ONLY" == "1" ]]; then
    if [[ ! -d "$dst" ]] || ! diff -rq "$src" "$dst" >/dev/null 2>&1; then
      echo "DRIFT: $skill — embed copy out of sync with source" >&2
      drift_found=1
    fi
    continue
  fi

  # Sync: rsync if available, otherwise rm + cp -R
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${src}/" "${dst}/"
  else
    rm -rf "$dst"
    mkdir -p "$dst"
    cp -R "${src}/." "${dst}/"
  fi

  echo "  ✓ synced $skill"
done

if [[ "$CHECK_ONLY" == "1" ]]; then
  if [[ "$drift_found" == "1" ]]; then
    echo "" >&2
    echo "Run ./scripts/sync-embedded-skills.sh to fix the drift, then commit." >&2
    exit 1
  fi
  echo "All embedded skills are in sync with sources."
fi
