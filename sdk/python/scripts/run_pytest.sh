#!/usr/bin/env bash

set -euo pipefail

# pytest <9 uses a predictable /tmp root on UNIX. Use a private temp root unless
# the caller has already chosen one explicitly.
if [[ -z "${PYTEST_DEBUG_TEMPROOT:-}" ]]; then
  _agentfield_pytest_root="$(mktemp -d)"
  cleanup() {
    rm -rf "${_agentfield_pytest_root}"
  }
  trap cleanup EXIT
  export PYTEST_DEBUG_TEMPROOT="${_agentfield_pytest_root}"
fi

exec pytest "$@"
