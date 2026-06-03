# Verification — Prove the Build Is Real

A scaffold that "looks right" but isn't wired up is worse than no scaffold. Static checks prove syntax and shape; only the **live smoke test** proves the call graph works.

---

## The verification ladder (run in order)

```bash
# 1. Control plane health
curl -fsS http://localhost:8080/api/v1/health | jq

# 2. Agent registered (PRIMARY check — durable across CP versions)
#    Response shape: .capabilities[].reasoners[].id  (NOT .reasoners[].name)
curl -fsS http://localhost:8080/api/v1/discovery/capabilities \
  | jq --arg slug "<slug>" '.capabilities[] | select(.agent_id==$slug) | {
      agent_id,
      n_reasoners: (.reasoners | length),
      entry: [.reasoners[] | select(.tags[]? == "entry") | .id],
      all_reasoner_ids: [.reasoners[].id]
    }'

# 3. Run the entry reasoner async (avoids the 90s sync timeout)
#    Body shape: {"input": {...kwargs...}} — kwargs NEVER raw at top level
EXEC_ID=$(curl -sS -X POST http://localhost:8080/api/v1/execute/async/<slug>.<entry> \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "<kwarg1>": "<realistic value>",
      "model": "<model_from_af_doctor>"
    }
  }' | jq -r '.execution_id')
echo "Execution: $EXEC_ID"

# 4. Poll until done
while :; do
  R=$(curl -sS http://localhost:8080/api/v1/executions/$EXEC_ID)
  S=$(echo "$R" | jq -r '.status')
  case "$S" in
    succeeded) echo "$R" | jq '.result'; break ;;
    failed)    echo "$R" | jq '.'; break ;;
    *)         sleep 2 ;;
  esac
done

# 5. (Showpiece) verifiable workflow chain — no other framework gives you this
LAST_EXEC=$(curl -s http://localhost:8080/api/v1/executions | jq -r '.[0].workflow_id')
curl -s http://localhost:8080/api/v1/did/workflow/$LAST_EXEC/vc-chain | jq
```

---

## Use `af agent` for the same checks (cleaner during dev)

| Want | af command |
|---|---|
| Find an endpoint | `af agent discover -q "discovery"` |
| Recent failed runs | `af agent query --resource executions --status failed --limit 5` |
| Inspect one execution | `af agent run --id <exec_id>` |
| Agent summary | `af agent agent-summary --id <slug>` |
| Goal-oriented dev help | `af agent kb guide --goal "<intent>"` |

---

## Common failures & fast diagnosis

| Symptom | Likely cause | Fix |
|---|---|---|
| `/api/v1/health` hangs/refuses | CP container still booting | Wait 5–10s, retry. Then `docker compose logs control-plane`. |
| `/api/v1/discovery/capabilities` shows agent but no reasoners | Reasoners are in a router that wasn't `app.include_router(...)`'d in `main.py` | Add the include. |
| Reasoner present but execute hangs | LLM call failing silently | `docker compose logs <slug> --follow` while curling — look for litellm errors. |
| Execute returns 500 "model not found" | `AI_MODEL` doesn't match the provider key | OpenRouter keys need `openrouter/...` model names, etc. |
| Execute 200 but result is empty/garbage | Architecture is wrong (e.g. `.ai()` got truncated input, or upstream `confident=False` cascaded as safe-default) | Read logs to see what input each reasoner actually got. |
| `AttributeError: 'dict' object has no attribute 'X'` | Cross-boundary reconstitution bug (Pydantic instance passed across `app.call`, receiver tried to use it as one) | `Model(**payload)` on receiver, OR render prose on sender. See `primitives-snapshot.md`. |
| `AttributeError: 'AgentRouter' has no attribute 'X'` | You used a router attribute that isn't proxied | Read from env or switch to direct agent access. See `primitives-snapshot.md` → router proxy surface. |
| `TypeError: argument after ** must be a mapping` | Same cross-boundary bug | Same fix. |
| Pipeline times out at 90s | Slow model, or fan-out too wide for sync, or stuck sub-reasoner | Use the async endpoint (already canonical). Pick a faster default model. |

---

## Sync vs async — when to use each

- **Sync (`POST /api/v1/execute/<target>`)** — 90-second hard timeout. Only use when you can guarantee the entire pipeline finishes in <60s. Single-call gates, simple chains.
- **Async (`POST /api/v1/execute/async/<target>`)** — returns `execution_id` immediately. Poll `GET /api/v1/executions/<id>`. **Canonical choice** for multi-reasoner pipelines.

When in doubt, use async. The cost is one extra polling loop in the smoke test — negligible compared to a timeout halfway through.

---

## Introspection endpoints (cheat sheet)

| Endpoint | Use for |
|---|---|
| `GET /api/v1/health` | Is the CP up? |
| `GET /api/v1/discovery/capabilities` | **Primary** registration check — durable across CP versions. Use this. |
| `GET /api/v1/nodes` | Secondary. Filter parameters vary across CP builds; can return empty even when registration succeeded. Cosmetic. |
| `POST /api/v1/execute/<target>` | Sync. 90s timeout. |
| `POST /api/v1/execute/async/<target>` | Async. Returns `execution_id`. **Canonical for multi-reasoner.** |
| `GET /api/v1/executions/<id>` | Status + result of an async execution. `running` → `succeeded` / `failed`. |
| `GET /api/v1/did/workflow/<workflow_id>/vc-chain` | Verifiable credential chain — the AgentField showpiece. |

**Rule of thumb:** prefer the endpoint whose semantics are stable across versions. "Does my reasoner exist?" is a durable question. "Is my node healthy according to filter parameters X/Y/Z?" is version-dependent.

---

## Mandatory live smoke test (before telling the user it's ready)

A build is not done until the canonical async curl has been fired against the live stack and returned `status: "succeeded"` with a real reasoned `result`.

```bash
# 1. Bring it up
docker compose up --build -d

# 2. Wait for registration
for i in $(seq 1 15); do
  READY=$(curl -fsS http://localhost:8080/api/v1/discovery/capabilities 2>/dev/null \
    | jq -r '.capabilities[] | select(.agent_id=="<slug>") | .agent_id')
  [ -n "$READY" ] && break
  sleep 2
done
[ -z "$READY" ] && { echo "Agent never registered"; docker compose logs <slug> --tail=50; exit 1; }

# 3. Fire the async curl with realistic input
EXEC_ID=$(curl -sS -X POST http://localhost:8080/api/v1/execute/async/<slug>.<entry> \
  -H 'Content-Type: application/json' \
  -d @./sample_payload.json | jq -r '.execution_id')

# 4. Poll
while :; do
  R=$(curl -sS http://localhost:8080/api/v1/executions/$EXEC_ID)
  S=$(echo "$R" | jq -r '.status')
  case "$S" in
    succeeded)
      echo "$R" | jq '.result'
      echo "✅ LIVE SMOKE TEST PASSED"
      break
      ;;
    failed)
      echo "❌ LIVE SMOKE TEST FAILED"
      echo "$R" | jq '.'
      docker compose logs <slug> --tail=100
      exit 1
      ;;
    *) sleep 2 ;;
  esac
done

# 5. Tear down
docker compose down
```

If the live smoke test fails, **do not hand off.** Read the error + agent container logs, find the stack trace, fix the bug, restart from step 1.

---

## When you cannot run Docker

You **must** still validate the Python and the compose file syntactically:

```bash
python3 -m py_compile main.py
python3 -m py_compile reasoners/*.py 2>/dev/null || true
OPENROUTER_API_KEY=sk-or-v1-FAKE docker compose config > /dev/null
```

Then provide the verification commands in the README as a checklist for the user to run themselves. "I generated it but didn't check syntax" is a failure mode.
