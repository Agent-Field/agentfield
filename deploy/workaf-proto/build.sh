#!/bin/zsh
# Build everything on spark into ~/af-workaf-proto/stage:
#   linux-arm64/agentfield-server     the control plane (this branch)
#   darwin-arm64/workaf-support       the example node (examples/workaf-support)
#   darwin-arm64/workaf-research      the second example node, when present
#   darwin-arm64/codeaf               codeaf at $CODEAF_REF, for the harness and the engine
# Idempotent: a target whose inputs did not change is rebuilt from Go's cache in
# seconds. Usage: build.sh [all|cp|nodes|codeaf|clean]
set -euo pipefail
HERE=${0:A:h}; source $HERE/env.sh
export GOCACHE=$LOCAL/gocache GOFLAGS=-mod=mod CGO_ENABLED=0
STAGE=$LOCAL/stage
mkdir -p $STAGE/linux-arm64 $STAGE/darwin-arm64

cp_build() {
  say "control plane ← $AF_ROOT/control-plane"
  # The control plane needs cgo (go-sqlite3), so it is built natively on spark.
  (cd $AF_ROOT/control-plane && CGO_ENABLED=1 go build -trimpath -o $STAGE/linux-arm64/agentfield-server ./cmd/agentfield-server)
}

nodes_build() {
  for ex in workaf-support workaf-research; do
    [[ -d $AF_ROOT/examples/$ex ]] || continue
    say "node $ex ← examples/$ex"
    (cd $AF_ROOT/examples/$ex && GOOS=darwin GOARCH=arm64 go build -trimpath -o $STAGE/darwin-arm64/$ex .)
  done
}

codeaf_build() {
  local src=$LOCAL/src/codeaf rev
  rev=$(git -C $CODEAF_REPO rev-parse "$CODEAF_REF")
  if [[ -f $src/.rev && $(<$src/.rev) == $rev && -x $STAGE/darwin-arm64/codeaf ]]; then
    say "codeaf ${rev:0:9} already built"; return
  fi
  say "codeaf ← $CODEAF_REF (${rev:0:9})"
  rm -rf $src && mkdir -p $src
  git -C $CODEAF_REPO archive $rev | tar -x -C $src
  (cd $src && GOOS=darwin GOARCH=arm64 go build -trimpath -o $STAGE/darwin-arm64/codeaf ./cmd/codeaf)
  print -r -- $rev > $src/.rev
}

case ${1:-all} in
  all) cp_build; nodes_build; codeaf_build;;
  cp) cp_build;; nodes) nodes_build;; codeaf) codeaf_build;;
  clean) go clean -cache 2>/dev/null || true; rm -rf $LOCAL/stage $LOCAL/src $LOCAL/gocache; say "removed stage, sources and GOCACHE";;
  *) die "usage: build.sh [all|cp|nodes|codeaf|clean]";;
esac
ls -la $STAGE/*/ 2>/dev/null | grep -v '^total' || true
