# Project `CLAUDE.md` Template

Every generated AgentField project ships with a `CLAUDE.md` at its root. This file is the contract that any *future* coding agent (including a fresh Claude Code session next week) must follow when extending the project.

Without this file, the next agent will refactor the system back into a CrewAI-style chain. With it, the architecture survives.

## Required structure

Generate a `CLAUDE.md` with these exact sections, customized to the specific build.

```markdown
# CLAUDE.md — <Use Case Name>

## Mission

<One sentence: what this system does and for whom.>

External callers should hit `<slug>.<entry_reasoner_name>` first.

## Architecture at a glance

- **Pattern(s):** <e.g., Parallel Hunters + HUNT→PROVE + Streaming>
- **Topology:** one AgentField node (`<slug>`) with N reasoners
- **Entry reasoner:** `<entry_reasoner_name>` — orchestrates the full pipeline
- **Internal reasoners:**
  - `<reasoner_1>` (`.ai()` / `.harness()`) — <one-line role>
  - `<reasoner_2>` (`.ai()` / `.harness()`) — <one-line role>
  - …
- **Inter-reasoner traffic:** all internal calls go through `app.call("<slug>.X", ...)`. Never direct HTTP.

## Why this architecture (not a chain)

<2–3 sentences explaining what makes this composite intelligence rather than a linear chain. Cite the dynamic-routing decisions, the parallelism, the harness/ai split. This is the "do not undo this" justification for the next agent.>

## Primitive selection rules (binding)

- `.ai()` is used ONLY at gates and routers (currently: `<list>`). Every `.ai()` here has a `confident` field and a `.harness()` fallback.
- `.harness()` is used for `<list>`. Each has hard caps on iterations and cost.
- `@app.skill()` is used for deterministic transforms (`<list>`).
- New reasoners default to `.harness()`. To use `.ai()`, prove the input fits in <2k tokens AND output fits in 4 flat fields AND there's a fallback.

## Data-flow rules

- Structured JSON between code and reasoners (when code branches on the result).
- Natural-language strings between reasoners that feed each other context.
- Hybrid only when both consumers exist. Do not use hybrid by default.

## Model selection

- Default model: `<openrouter/google/gemini-2.5-flash>` via `AI_MODEL` env.
- The entry reasoner accepts an OPTIONAL `model` parameter in the request body. When present, it propagates to all child reasoners via `app.call(..., model=model)`. This lets users A/B models per request without redeploying.
- Provider keys: `OPENROUTER_API_KEY` (default), `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` — any LiteLLM-compatible model works.

## Runtime contract

- Local runtime is `docker-compose.yml` in this directory.
- One container: `agentfield/control-plane:latest` (local mode, SQLite/BoltDB).
- One container: this Python agent node, built from `Dockerfile`.
- The agent node depends on the control plane being healthy before it boots.
- Default ports: control plane `8080`, agent node `8001`. Override via env if needed.

## Delivery contract — every change must preserve

- ✅ A runnable `docker compose up --build` (validated with `docker compose config`)
- ✅ A valid `.env.example` listing all required keys
- ✅ A `README.md` with the exact verification ladder (health → nodes → capabilities → execute)
- ✅ The canonical curl smoke test in the README — body shape `{"input": {...kwargs...}}`, returns a real reasoned answer not a stub
- ✅ This `CLAUDE.md`

## Validation commands (run after every change)

```bash
python3 -m py_compile main.py
docker compose config > /dev/null
docker compose up --build -d
sleep 8
curl -fsS http://localhost:8080/api/v1/health
curl -fsS http://localhost:8080/api/v1/nodes | jq '.[].node_id'
curl -fsS http://localhost:8080/api/v1/discovery/capabilities | jq '.reasoners | map(select(.node_id=="<slug>")) | map(.name)'
# the canonical curl from README.md
docker compose down
```

If any of those fail, the change is not done.

## Anti-patterns (reject these)

- ❌ Direct HTTP between reasoners. All internal traffic uses `app.call`.
- ❌ Replacing a `.harness()` with `.ai()` "for speed" without proving the input fits.
- ❌ Adding a new reasoner without registering it through the entry reasoner OR through a router that's included in `main.py`.
- ❌ Removing the smoke test from README "because it's obvious."
- ❌ Hardcoding `node_id` in `app.call`. Always use `f"{app.node_id}.X"` so renaming the node doesn't break the system.
- ❌ Hardcoding the model. Always read from env (`AI_MODEL`) and accept a per-request override.
- ❌ Replacing the dynamic routing in `<entry_reasoner_name>` with a static `for` loop.
- ❌ Unbounded loops or recursive harness spawns without explicit caps.
- ❌ Removing the `confident` field from a `.ai()` schema without replacing the validation check.

## Extension points (where to safely add work)

<3–5 bullets specific to the architecture. Examples:>
- Add a new analysis dimension: create a new `@app.reasoner()` that takes the same inputs as the existing dimension reviewers, and add it to the dispatch list in `<entry_reasoner_name>`.
- Switch from `.ai()` intake to `.harness()` intake when inputs grow past 2 pages: replace `intake_router` with `intake_navigator` per `references/primitives.md` in the skill.
- Add provenance: have each dimension reviewer return citation keys, then add a `provenance_collector` that aggregates them into the final response.

## Owner

This system was scaffolded by the `agentfield-multi-reasoner-builder` skill. To rebuild, run that skill again with the same use case description. To extend, follow this CLAUDE.md.
```

## Generation rules

When you write the actual `CLAUDE.md` for a build:

1. **Fill in every `<placeholder>`.** Do not ship a CLAUDE.md with `<entry_reasoner_name>` still in it.
2. **List every reasoner you actually generated** with its primitive (`.ai()` or `.harness()`) and one-line role.
3. **Justify the architecture** in 2–3 sentences. The "Why this architecture" section is the most important part — it tells the next agent what NOT to undo.
4. **Customize the extension points** to the specific build. Don't copy the generic examples.
5. **Match the validation commands to the actual reasoners and node ID.** No `<slug>` placeholders in the final file.
