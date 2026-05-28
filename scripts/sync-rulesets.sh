#!/usr/bin/env bash
# Sync checked-in rulesets under .github/rulesets/ to the GitHub repo.
#
# Usage:
#   ./scripts/sync-rulesets.sh                    # auto-detect repo from git remote
#   ./scripts/sync-rulesets.sh owner/repo         # explicit target
#   DRY_RUN=1 ./scripts/sync-rulesets.sh          # print what would happen, make no changes
#
# Requires `gh` with a token that has `repo` scope on the target repository.
# Designed to be called both by humans (for the initial rollout) and by
# .github/workflows/sync-rulesets.yml on pushes to main.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RULESETS_DIR="$ROOT_DIR/.github/rulesets"

if [[ ! -d "$RULESETS_DIR" ]]; then
  echo "no rulesets directory at $RULESETS_DIR" >&2
  exit 1
fi

REPO="${1:-}"
if [[ -z "$REPO" ]]; then
  REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)"
fi
if [[ -z "$REPO" ]]; then
  echo "could not determine repo. Pass owner/repo as arg or run inside a gh-authenticated checkout." >&2
  exit 1
fi

echo "==> syncing rulesets to $REPO"

# List existing rulesets so we can update by name instead of blindly creating
# duplicates on every run.
EXISTING_JSON="$(gh api "repos/$REPO/rulesets" 2>/dev/null || echo '[]')"

shopt -s nullglob
for file in "$RULESETS_DIR"/*.json; do
  name="$(jq -r '.name' "$file")"
  if [[ -z "$name" || "$name" == "null" ]]; then
    echo "   skipping $file (no .name field)" >&2
    continue
  fi

  existing_id="$(jq -r --arg n "$name" '.[] | select(.name == $n) | .id' <<<"$EXISTING_JSON" | head -1)"

  if [[ -n "${DRY_RUN:-}" ]]; then
    if [[ -n "$existing_id" ]]; then
      echo "   [dry-run] would PUT repos/$REPO/rulesets/$existing_id  from $(basename "$file")"
    else
      echo "   [dry-run] would POST repos/$REPO/rulesets              from $(basename "$file")"
    fi
    continue
  fi

  if [[ -n "$existing_id" ]]; then
    echo "   updating ruleset '$name' (id=$existing_id)"
    gh api --method PUT "repos/$REPO/rulesets/$existing_id" --input "$file" >/dev/null
  else
    echo "   creating ruleset '$name'"
    gh api --method POST "repos/$REPO/rulesets" --input "$file" >/dev/null
  fi
done

echo "==> done"
