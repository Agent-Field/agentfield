#!/usr/bin/env bash
#
# Fires signed test events at the local AgentField control plane so the
# triggers-demo can be exercised end-to-end without standing up a real Stripe
# or GitHub provider.
#
# Discovers the agent's code-managed triggers via the CP's API, signs the
# bundled fixture payloads with the demo secrets baked into docker-compose,
# and POSTs them at the public ingest endpoint. The CP verifies the
# signature, persists the event, and dispatches it to the agent — every
# step shows up live in http://localhost:8080/ui/triggers.
#
# Usage (after `docker compose up -d`):
#
#   ./scripts/fire-events.sh
#
# To target a non-default CP host:
#
#   AGENTFIELD_URL=http://my-host:8080 ./scripts/fire-events.sh

set -euo pipefail

AGENTFIELD_URL="${AGENTFIELD_URL:-http://localhost:8080}"
STRIPE_DEMO_SECRET="${STRIPE_DEMO_SECRET:-whsec_demo_stripe_secret_change_me_in_prod}"
GITHUB_DEMO_SECRET="${GITHUB_DEMO_SECRET:-ghsecret_demo_github_secret_change_me}"
SLACK_DEMO_SIGNING_SECRET="${SLACK_DEMO_SIGNING_SECRET:-slack_demo_signing_secret_change_me}"
GENERIC_HMAC_DEMO_SECRET="${GENERIC_HMAC_DEMO_SECRET:-generic_hmac_demo_secret_change_me}"
GENERIC_BEARER_DEMO_TOKEN="${GENERIC_BEARER_DEMO_TOKEN:-generic_bearer_demo_token_change_me}"
TARGET_NODE_ID="${TARGET_NODE_ID:-triggers-demo-agent}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: this script needs '$1' on PATH (install or use a different shell)" >&2
    exit 2
  }
}

require_cmd curl
require_cmd python3

cp_alive() {
  curl -fsS "${AGENTFIELD_URL}/api/v1/sources" >/dev/null 2>&1
}

discover_trigger_id() {
  # Find a code-managed trigger by source name and (optionally) accepted
  # event type. The agent declares two GitHub triggers — one for
  # `pull_request`, one for `issues` — so we filter by event_type to ensure
  # we hit the reasoner that actually wants the payload we're about to send.
  local source="$1" event_type="${2:-}"
  python3 -c "
import json, sys, urllib.request
source = '${source}'
event_type = '${event_type}'
with urllib.request.urlopen('${AGENTFIELD_URL}/api/v1/triggers') as r:
    body = json.loads(r.read())
for t in body.get('triggers', []):
    if t.get('source_name') != source:
        continue
    if event_type and event_type not in (t.get('event_types') or []):
        continue
    print(t['id'])
    sys.exit(0)
sys.exit(1)
" 2>/dev/null
}

stripe_sign() {
  # Stripe-Signature format: t=<unix_ts>,v1=<hex_hmac_sha256(<ts>.<body>)>
  local secret="$1" body="$2"
  local ts; ts=$(date +%s)
  local sig
  sig=$(python3 -c "
import hashlib, hmac, sys
secret = sys.argv[1].encode()
ts     = sys.argv[2].encode()
body   = sys.argv[3].encode()
print(hmac.new(secret, ts + b'.' + body, hashlib.sha256).hexdigest())
" "$secret" "$ts" "$body")
  printf 't=%s,v1=%s' "$ts" "$sig"
}

github_sign() {
  # X-Hub-Signature-256: sha256=<hex_hmac_sha256(body)>
  local secret="$1" body="$2"
  python3 -c "
import hashlib, hmac, sys
print('sha256=' + hmac.new(sys.argv[1].encode(), sys.argv[2].encode(), hashlib.sha256).hexdigest())
" "$secret" "$body"
}

slack_sign() {
  # Slack signing-secret format:
  #   X-Slack-Request-Timestamp: <unix_ts>
  #   X-Slack-Signature: v0=<hex_hmac_sha256("v0:<ts>:<body>")>
  local secret="$1" ts="$2" body="$3"
  python3 -c "
import hashlib, hmac, sys
secret, ts, body = sys.argv[1].encode(), sys.argv[2].encode(), sys.argv[3].encode()
basestring = b'v0:' + ts + b':' + body
print('v0=' + hmac.new(secret, basestring, hashlib.sha256).hexdigest())
" "$secret" "$ts" "$body"
}

hmac_sign_hex() {
  # Plain HMAC-SHA256 hex digest of body — used by the generic_hmac plugin
  # with the default X-Signature header and no scheme prefix.
  local secret="$1" body="$2"
  python3 -c "
import hashlib, hmac, sys
print(hmac.new(sys.argv[1].encode(), sys.argv[2].encode(), hashlib.sha256).hexdigest())
" "$secret" "$body"
}

ensure_ui_trigger() {
  # Idempotently create a UI-managed trigger for a given source if one
  # routing at handle_inbound doesn't already exist. Echoes the trigger ID
  # on stdout. Quiet on success — chatty on failure so the operator sees it.
  local source="$1" secret_env="$2" config_json="$3" event_types_json="$4"
  local existing
  existing=$(python3 -c "
import json, sys, urllib.request
src = '${source}'
target_reasoner = 'handle_inbound'
target_node = '${TARGET_NODE_ID}'
with urllib.request.urlopen('${AGENTFIELD_URL}/api/v1/triggers') as r:
    body = json.loads(r.read())
for t in body.get('triggers', []):
    if t.get('source_name') == src and t.get('target_reasoner') == target_reasoner and t.get('target_node_id') == target_node:
        print(t['id']); sys.exit(0)
sys.exit(1)
" 2>/dev/null) || existing=""
  if [[ -n "${existing:-}" ]]; then
    printf '%s' "$existing"
    return 0
  fi
  # Pass user-supplied JSON values via env vars rather than substituting them
  # into the python literal — avoids any quote-escaping foot-guns when the
  # config or event_types contain double quotes. Note: defaults are applied
  # via plain conditional rather than ${var:-{}}, since the shell's brace
  # parsing inside :- defaults will append a stray `}` when var ends in `}`.
  local cfg_in="${config_json:-}"
  local types_in="${event_types_json:-}"
  [[ -z "$cfg_in" ]] && cfg_in='{}'
  [[ -z "$types_in" ]] && types_in='[]'
  local payload
  payload=$(SOURCE="$source" \
    SECRET_ENV="$secret_env" \
    TARGET_NODE_ID="$TARGET_NODE_ID" \
    CONFIG_JSON="$cfg_in" \
    EVENT_TYPES_JSON="$types_in" \
    python3 -c '
import json, os
print(json.dumps({
    "source_name": os.environ["SOURCE"],
    "config": json.loads(os.environ["CONFIG_JSON"]),
    "secret_env_var": os.environ["SECRET_ENV"],
    "target_node_id": os.environ["TARGET_NODE_ID"],
    "target_reasoner": "handle_inbound",
    "event_types": json.loads(os.environ["EVENT_TYPES_JSON"]),
    "enabled": True,
}))')
  local resp
  resp=$(curl -sS -X POST "${AGENTFIELD_URL}/api/v1/triggers" \
    -H 'Content-Type: application/json' \
    --data-binary "$payload")
  local id
  id=$(printf '%s' "$resp" | python3 -c "import json,sys;print(json.load(sys.stdin).get('id') or '')" 2>/dev/null || true)
  if [[ -z "$id" ]]; then
    echo "error: failed to create UI-managed ${source} trigger: ${resp}" >&2
    return 1
  fi
  printf '%s' "$id"
}

post_event() {
  local label="$1" url="$2" body="$3"
  shift 3
  local headers=("$@")
  echo "→ POST $label  →  $url"
  local args=(-sS -X POST "$url" -H 'Content-Type: application/json' --data-binary "$body")
  for h in "${headers[@]}"; do
    args+=(-H "$h")
  done
  local resp
  resp=$(curl "${args[@]}")
  echo "  response: $resp"
}

# ---------------------------------------------------------------------------
# Wait for CP + agent registration
# ---------------------------------------------------------------------------

echo "checking control plane at ${AGENTFIELD_URL}..."
for _ in $(seq 1 60); do
  if cp_alive; then break; fi
  sleep 1
done
if ! cp_alive; then
  echo "error: control plane did not become reachable at ${AGENTFIELD_URL}" >&2
  echo "  (did you run 'docker compose up -d' from the triggers-demo directory?)" >&2
  exit 1
fi

# Wait until the demo agent has registered with the CP and its code-managed
# triggers exist. The agent declares 3 triggers; we wait until at least the
# Stripe and GitHub ones appear.
echo "waiting for the demo agent to register triggers..."
for _ in $(seq 1 60); do
  stripe_id=$(discover_trigger_id stripe payment_intent.succeeded || true)
  github_pr_id=$(discover_trigger_id github pull_request || true)
  if [[ -n "${stripe_id:-}" && -n "${github_pr_id:-}" ]]; then
    break
  fi
  sleep 1
done

if [[ -z "${stripe_id:-}" || -z "${github_pr_id:-}" ]]; then
  echo "error: demo agent's triggers didn't register within 60s" >&2
  echo "  current triggers:" >&2
  curl -fsS "${AGENTFIELD_URL}/api/v1/triggers" >&2 || true
  exit 1
fi
echo "  stripe trigger:        ${stripe_id}"
echo "  github pull_request:   ${github_pr_id}"
echo

# ---------------------------------------------------------------------------
# Fire one Stripe payment_intent.succeeded
# ---------------------------------------------------------------------------

# Randomize Stripe event + payment_intent ids per run so re-running the
# script always produces fresh events on the trigger detail page. (Real
# providers depend on idempotency to safely retry; we want every demo run
# to show new events.)
stripe_evt_suffix="$(python3 -c 'import secrets;print(secrets.token_hex(6))')"
stripe_body=$(STRIPE_SUFFIX="$stripe_evt_suffix" python3 -c '
import json, os
suf = os.environ["STRIPE_SUFFIX"]
print(json.dumps({
    "id": f"evt_demo_{suf}",
    "object": "event",
    "type": "payment_intent.succeeded",
    "created": 1735395600,
    "data": {"object": {
        "id": f"pi_demo_{suf}",
        "object": "payment_intent",
        "amount": 4200,
        "currency": "usd",
        "customer": "cus_demo_42",
        "status": "succeeded",
        "metadata": {"order_id": f"ord_demo_{suf}"},
    }},
}))')

stripe_sig=$(stripe_sign "$STRIPE_DEMO_SECRET" "$stripe_body")
post_event "stripe payment_intent.succeeded" \
  "${AGENTFIELD_URL}/sources/${stripe_id}" \
  "$stripe_body" \
  "Stripe-Signature: ${stripe_sig}"

# ---------------------------------------------------------------------------
# Fire one GitHub pull_request.opened
# ---------------------------------------------------------------------------

github_body=$(cat <<'JSON'
{"action":"opened","number":42,"pull_request":{"id":1234567890,"number":42,"state":"open","title":"AgentField triggers demo","html_url":"https://github.com/demo-org/demo-repo/pull/42","user":{"login":"demo-user","id":1000,"type":"User"},"draft":false,"merged":false},"repository":{"id":654321,"name":"demo-repo","full_name":"demo-org/demo-repo","private":false},"sender":{"login":"demo-user","type":"User"}}
JSON
)

github_sig=$(github_sign "$GITHUB_DEMO_SECRET" "$github_body")
delivery_id="$(uuidgen 2>/dev/null || python3 -c 'import uuid;print(uuid.uuid4())')"
post_event "github pull_request.opened" \
  "${AGENTFIELD_URL}/sources/${github_pr_id}" \
  "$github_body" \
  "X-GitHub-Event: pull_request" \
  "X-GitHub-Delivery: ${delivery_id}" \
  "X-Hub-Signature-256: ${github_sig}"

# ---------------------------------------------------------------------------
# Fire one GitHub issues.opened (only if the issues trigger is registered)
# ---------------------------------------------------------------------------

github_issues_id=$(discover_trigger_id github issues || true)
if [[ -n "${github_issues_id:-}" ]]; then
  github_issue_body=$(cat <<'JSON'
{"action":"opened","issue":{"id":2222222,"number":7,"title":"Login button does nothing on Safari 17","body":"When I click the login button on Safari 17 macOS Sonoma, nothing happens. Console shows no errors. Steps to reproduce: open the site in a fresh incognito window, click login. Expected: the OAuth modal opens. Actual: silent no-op. Reproduces 100% of the time on my end.","state":"open","html_url":"https://github.com/demo-org/demo-repo/issues/7","user":{"login":"demo-reporter","id":2000,"type":"User"}},"repository":{"id":654321,"name":"demo-repo","full_name":"demo-org/demo-repo","private":false},"sender":{"login":"demo-reporter","type":"User"}}
JSON
)
  github_issue_sig=$(github_sign "$GITHUB_DEMO_SECRET" "$github_issue_body")
  issue_delivery_id="$(uuidgen 2>/dev/null || python3 -c 'import uuid;print(uuid.uuid4())')"
  post_event "github issues.opened" \
    "${AGENTFIELD_URL}/sources/${github_issues_id}" \
    "$github_issue_body" \
    "X-GitHub-Event: issues" \
    "X-GitHub-Delivery: ${issue_delivery_id}" \
    "X-Hub-Signature-256: ${github_issue_sig}"
fi

# ---------------------------------------------------------------------------
# Slack — Events API (signing-secret HMAC)
#
# Lazily creates a UI-managed Slack trigger routed at handle_inbound and
# fires a synthetic event_callback. The Slack source plugin verifies
# X-Slack-Signature against the body using SLACK_DEMO_SIGNING_SECRET and
# unwraps event.type out of the event_callback envelope before dispatch.
# ---------------------------------------------------------------------------

slack_id=$(ensure_ui_trigger slack SLACK_DEMO_SIGNING_SECRET '{}' '["app_mention"]')
if [[ -n "${slack_id:-}" ]]; then
  slack_event_id="Ev$(python3 -c 'import secrets;print(secrets.token_hex(8))')"
  slack_body=$(SLACK_EVT_ID="$slack_event_id" python3 -c '
import json, os
eid = os.environ["SLACK_EVT_ID"]
print(json.dumps({
    "type": "event_callback",
    "team_id": "T1234",
    "api_app_id": "A1234",
    "event": {"type": "app_mention", "user": "U1234", "text": "<@U_BOT> hello from the demo",
              "ts": "1735395600.000100", "channel": "C1234", "event_ts": "1735395600.000100"},
    "event_id": eid,
    "event_time": 1735395600,
}))')
  slack_ts=$(date +%s)
  slack_sig=$(slack_sign "$SLACK_DEMO_SIGNING_SECRET" "$slack_ts" "$slack_body")
  post_event "slack app_mention" \
    "${AGENTFIELD_URL}/sources/${slack_id}" \
    "$slack_body" \
    "X-Slack-Request-Timestamp: ${slack_ts}" \
    "X-Slack-Signature: ${slack_sig}"
fi

# ---------------------------------------------------------------------------
# Generic HMAC — bring-your-own-signed inbound webhook
#
# Default config: signature in `X-Signature` (hex digest, no scheme prefix),
# event type in `X-Event-Type`. Useful for in-house webhooks where the
# provider isn't one of the named integrations.
# ---------------------------------------------------------------------------

hmac_id=$(ensure_ui_trigger generic_hmac GENERIC_HMAC_DEMO_SECRET \
  '{"signature_header":"X-Signature","event_type_header":"X-Event-Type"}' \
  '["order.created"]')
if [[ -n "${hmac_id:-}" ]]; then
  hmac_body='{"order_id":"ord_demo_77","total_cents":12500,"currency":"usd","customer":"cus_demo_77"}'
  hmac_sig=$(hmac_sign_hex "$GENERIC_HMAC_DEMO_SECRET" "$hmac_body")
  post_event "generic_hmac order.created" \
    "${AGENTFIELD_URL}/sources/${hmac_id}" \
    "$hmac_body" \
    "X-Signature: ${hmac_sig}" \
    "X-Event-Type: order.created"
fi

# ---------------------------------------------------------------------------
# Generic Bearer — token-authenticated inbound webhook
#
# Default config: `Authorization: Bearer <token>`. The CP compares the
# presented token against the value in GENERIC_BEARER_DEMO_TOKEN.
# ---------------------------------------------------------------------------

bearer_id=$(ensure_ui_trigger generic_bearer GENERIC_BEARER_DEMO_TOKEN \
  '{"event_type_header":"X-Event-Type"}' \
  '["alert.fired"]')
if [[ -n "${bearer_id:-}" ]]; then
  bearer_body='{"alert":"db_high_latency","p95_ms":1450,"region":"us-east-1"}'
  post_event "generic_bearer alert.fired" \
    "${AGENTFIELD_URL}/sources/${bearer_id}" \
    "$bearer_body" \
    "Authorization: Bearer ${GENERIC_BEARER_DEMO_TOKEN}" \
    "X-Event-Type: alert.fired"
fi

# ---------------------------------------------------------------------------
# Cron fires itself every minute — nothing to send.
# ---------------------------------------------------------------------------

echo
echo "done. Open ${AGENTFIELD_URL}/ui/triggers to see the events flow through."
echo "the cron trigger will also fire automatically every minute on the minute."
