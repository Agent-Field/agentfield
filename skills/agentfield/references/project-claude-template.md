# Project `CLAUDE.md` Template

Every generated AgentField project ships with a `CLAUDE.md` at its root. This file is the contract that any *future* coding agent (including a fresh Claude Code session next week) must follow when extending the project.

Without this file, the next agent will refactor the system back into a flat chain. With it, the architecture survives.

---

## Required structure

Generate a `CLAUDE.md` with these exact sections, customized to the specific build. Fill in every `<placeholder>`.

```markdown
# CLAUDE.md — <Use Case Name>

## Mission

<One sentence: what this system does and for whom.>

External callers should hit `<slug>.<entry_reasoner_name>` first.

## Architecture at a glance

- **Pattern(s) (post-hoc name, not template):** <e.g., parallel hunters + dynamic router>, derived via the `agentfield` skill's derivation procedure (cognitive jobs → autonomy → verification rungs → dynamism rung + budgets)
- **Topology:** one AgentField node (`<slug>`) with <N> reasoners
- **Depth from entry to leaf:** <N> layers — every "specialist" is itself an orchestrator calling 2–4 sub-reasoners
- **Entry reasoner:** `<entry_reasoner_name>` — orchestrates the full pipeline
- **Internal reasoners:**
  - `<reasoner_1>` (`.ai()` / `.harness()` / `.skill()`) — <one-line role>
  - `<reasoner_2>` (...) — <one-line role>
  - …
- **Inter-reasoner traffic:** all internal calls go through `app.call("<slug>.X", ...)`. Never direct HTTP.

## Why this architecture (not a chain)

<2–3 sentences explaining what makes this composite intelligence rather than a linear chain. Cite the dynamic-routing decisions, the parallelism, the per-reasoner judgment split. This is the "do not undo this" justification for the next agent.>

## Primitive selection rules (binding)

- `.ai()` is used at: <list of gates and routers>. Every `.ai()` here has a `confident` field and a defined fallback path.
- `@app.reasoner()` orchestrators are at: <list of the composers>. Each calls 2–4 sub-reasoners.
- `@app.skill()` is used for: <list of deterministic transforms>.
- `.harness()` is used at: <list, or "not used in this build">. If used, each has hard caps on iterations and cost AND the Dockerfile installs the CLI AND `main.py` has a `shutil.which()` guard.

## Data-flow rules

- Structured JSON between code and reasoners (when code branches on the result).
- Natural-language strings between reasoners that feed each other context.
- Hybrid only when both consumers exist.
- **Cross-`app.call` boundary:** every structured payload arrives as a dict on the receiver — reconstruct via `Model(**payload)`, or render to prose on the sender.

## Model selection

- Default model: `<model_from_af_doctor>` via `AI_MODEL` env.
- The entry reasoner accepts an OPTIONAL `model` parameter in the request body. When present, it propagates to all child reasoners via `app.call(..., model=model)`. This lets users A/B models per request without redeploying.
- Provider keys: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` — any LiteLLM-compatible model works.

## Runtime contract

- Local runtime is `docker-compose.yml` in this directory.
- One container: `agentfield/control-plane:latest` (local mode, SQLite/BoltDB).
- One container: this Python agent node, built from `Dockerfile`.
- The agent node depends on the control plane being healthy before it boots.
- Default ports: control plane `8080`, agent node `8001`. Override via env if needed.

## Delivery contract — every change must preserve

- ✅ A runnable `docker compose up --build` (validated with `docker compose config`)
- ✅ A valid `.env.example` listing all required keys
- ✅ A `README.md` with the build checks (health → discovery/capabilities → async execute + poll → vc-chain)
- ✅ The canonical async curl smoke test in the README — body shape `{"input": {...kwargs...}}`, returns a real reasoned answer
- ✅ This `CLAUDE.md`

## Validation commands (run after every change)

```bash
python3 -m py_compile main.py
python3 -m py_compile reasoners/*.py 2>/dev/null || true
docker compose config > /dev/null
docker compose up --build -d
sleep 8
curl -fsS http://localhost:8080/api/v1/health
curl -fsS http://localhost:8080/api/v1/discovery/capabilities | jq '.capabilities[] | select(.agent_id=="<slug>") | .reasoners | map(.id)'
# the canonical async curl from README.md
docker compose down
```

If any of those fail, the change is not done.

## Anti-patterns (reject these)

- ❌ Direct HTTP between reasoners. All internal traffic uses `app.call`.
- ❌ Replacing a `.harness()` with `.ai()` "for speed" without proving the input fits.
- ❌ Adding a new reasoner without registering it through the entry reasoner OR through a router included in `main.py`.
- ❌ Removing the smoke test from README "because it's obvious."
- ❌ Hardcoding `node_id` in `app.call`. Always use `f"{app.node_id}.X"`.
- ❌ Hardcoding the model. Always read from env (`AI_MODEL`) and accept a per-request override.
- ❌ Replacing dynamic routing in `<entry_reasoner_name>` with a static `for` loop.
- ❌ Unbounded loops or recursive harness spawns without explicit caps.
- ❌ Removing the `confident` field from a `.ai()` schema without replacing the validation check.
- ❌ Flattening depth to 2 (entry → specialists → done) — every specialist must itself orchestrate sub-reasoners.

## Extension points (where to safely add work)

<3–5 bullets specific to the architecture. Examples — customize:>
- Add a new analysis dimension: create a new `@app.reasoner()` with the existing dimension reviewer's signature, and add it to the dispatch list in `<entry_reasoner_name>`.
- Swap an `.ai()` intake for a chunked-loop reasoner when inputs grow past ~2k tokens.
- Add provenance by having each dimension reviewer return citation keys, then add a `provenance_collector` that aggregates them into the final response.

## Owner

This system was scaffolded by the `agentfield` skill. To rebuild, run that skill again with the same use case description. To extend, follow this CLAUDE.md.
```

---

## Generation rules

1. **Fill in every `<placeholder>`.** Do not ship a CLAUDE.md with `<entry_reasoner_name>` still in it.
2. **List every reasoner you actually generated** with its primitive and one-line role.
3. **Justify the architecture** in 2–3 sentences. The "Why this architecture" section is the most important — it tells the next agent what NOT to undo.
4. **Customize the extension points** to the specific build.
5. **Match the validation commands to the actual reasoners and node ID.** No `<slug>` placeholders in the final file.
