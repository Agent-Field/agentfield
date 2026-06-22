#!/usr/bin/env bash
# Cross-language interop proof for DID payload encryption.
# Exercises the REAL TypeScript and Python SDK crypto primitives in both
# directions: TS-encrypt -> Python-decrypt, and Python-encrypt -> TS-decrypt.
#
# Run from anywhere:  bash sdk/interop/run_interop.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_SDK="$HERE/../typescript"
PY_SDK="$HERE/../python"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

TSX="$TS_SDK/node_modules/.bin/tsx"
PAYLOAD='{"scope":{"tier":"project","workspaceId":"ws_1","projectId":"proj_42"},"k":"secret-token"}'

run_ts() { ( cd "$TS_SDK" && "$TSX" "$HERE/ts_tool.mts" "$@" ); }
# Force the workspace's agentfield package onto the path (the env may have a
# different checkout installed); a bare script puts its own dir on sys.path, not cwd.
run_py() { ( cd "$PY_SDK" && PYTHONPATH="$PY_SDK${PYTHONPATH:+:$PYTHONPATH}" python3 "$HERE/py_tool.py" "$@" ); }

pass=0; fail=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "  PASS: $1"; pass=$((pass+1));
  else echo "  FAIL: $1"; echo "    expected: $2"; echo "    actual:   $3"; fail=$((fail+1)); fi
}

echo "== Direction A: TS encrypt -> Python decrypt =="
run_ts genkey > "$TMP/ts_keys.json"
python3 -c "import json,sys;d=json.load(open('$TMP/ts_keys.json'));open('$TMP/a_pub.json','w').write(json.dumps(d['publicJwk']));open('$TMP/a_priv.json','w').write(json.dumps(d['privateJwk']))"
run_ts encrypt "$TMP/a_pub.json" "$PAYLOAD" > "$TMP/a.jwe"
A_OUT="$(run_py decrypt "$TMP/a_priv.json" "$TMP/a.jwe")"
check "TS->Py round-trip" "$PAYLOAD" "$A_OUT"

echo "== Direction B: Python encrypt -> TS decrypt =="
run_py genkey > "$TMP/py_keys.json"
python3 -c "import json;d=json.load(open('$TMP/py_keys.json'));open('$TMP/b_pub.json','w').write(json.dumps(d['publicJwk']));open('$TMP/b_priv.json','w').write(json.dumps(d['privateJwk']))"
run_py encrypt "$TMP/b_pub.json" "$PAYLOAD" > "$TMP/b.jwe"
B_OUT="$(run_ts decrypt "$TMP/b_priv.json" "$TMP/b.jwe")"
check "Py->TS round-trip" "$PAYLOAD" "$B_OUT"

echo
echo "interop: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
