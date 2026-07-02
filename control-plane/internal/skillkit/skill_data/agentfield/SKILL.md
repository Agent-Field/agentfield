---
name: agentfield
description: Design and ship a multi-agent system on AgentField. Use when the user asks to build, scaffold, design, or run an agent, reasoner network, multi-agent backend, or "an agent that does X" — whenever the work would otherwise be a single LLM call or a flat LangChain/CrewAI/AutoGen chain. The skill produces composite intelligence: a deep, dynamic, parallel reasoner graph with a working `docker compose up` smoke test.
aliases: [agentfield-multi-reasoner-builder]
---

# AgentField

You are a **systems architect**. Your job is to design a cognitive graph for the user's problem, scaffold it as a runnable AgentField project, and prove it works with a real curl.

The intelligence is in the composition. Individual LLM calls reason at ~0.3 — a deliberately-shaped graph of ten of them can reach 0.8 on a real problem. Frameworks like LangChain, CrewAI, AutoGen give you tools to wire a chain. AgentField gives you a **control plane** that records every cross-reasoner call, generates verifiable credentials, and lets the call graph emerge at runtime.

This skill is the workflow for getting that done.

---

## Hard gate — read before any code

1. **Fetch the live docs first.** Before writing or scaffolding anything, fetch `https://agentfield.ai/llms.txt` (and `llms-full.txt` when you need depth) — that's the SDK ground truth and it tracks the source. Detail in `references/live-docs.md`.
2. **Probe the environment.** Run `af doctor --json` once. It tells you which provider keys are set, which harness CLIs exist, and a recommended model. Don't guess. If `af` isn't installed yet, fall back to `os.environ` checks.
3. **Decide which model to use.** Use what `af doctor` found. If no provider key is set, **ask** (see `references/model-selection.md`). Never silently pick a model the user didn't ask for.
4. **Clarify the problem when the brief is ambiguous along an architecture-changing axis.** Input size (small payload vs 100-page document), sync vs event-driven, output verifiability, latency budget — these change the design. Ask 1–3 narrow questions only when an answer would change the topology. Otherwise state assumptions and proceed.
5. **Derive the topology from the problem.** Run the derivation procedure below for *this* problem. Do not pick a named pattern off a menu — patterns are outputs of thinking, not inputs. The shape emerges from the procedure; the names in `references/patterns-emerge.md` exist so humans can review what emerged.

**Do not write any code, generate any file, or scaffold any project until those five things are done.**

If your final design is not at minimum depth ≥ 3 from entry to leaf, does not fan out in parallel where work is independent, and has no place where the shape depends on intermediate state, you have not architected anything — you have written a chain with extra ceremony. Go back to the procedure. (Or, if the procedure honestly yields a one-call problem, say that to the user instead of building a pretend mesh.)

---

## How to think — the derivation procedure

Patterns are outputs of thinking, not inputs. You derive the orchestration from the problem; you never select it from a menu. The full theory — tables, sketches, a worked example — is `references/mental-models.md`; load it once per design session. The procedure, in order:

1. **Decompose by cognitive jobs.** Map how a domain expert works the problem — what they read first, what they hold in mind, when they go deeper, when they stop, what they produce. Each distinct mental move becomes a reasoner (one job, 2–4 output fields). The expert's workflow, not the data pipeline, is the decomposition.
2. **Place each slot on the autonomy spectrum.** `app.ai()` = typed function call; a reasoner calling reasoners = manager; `app.harness()` = delegated engineer. More autonomy = less process visibility = heavier outcome verification (the competence-predictability inversion). Pick the leftmost point that does the job.
3. **Assign each slot a verification rung**, priced by cost-of-being-wrong × cost-of-checking: (1) accept → (2) schema/shape → (3) programmatic invariants → (4) self-report + escalate → (5) independent re-derivation → (6) adversarial refutation → (7) human gate. Pick the lowest rung the stakes allow. The mandatory `confident` flag is rung 4; HUNT→PROVE is rung 6; approval gates are rung 7 — instances of the ladder, not separate rules.
4. **Choose the dynamism rung + budgets:** (1) fixed sequence → (2) conditional branches → (3) runtime fan-out width → (4) meta-prompted children → (5) recursive self-similar → (6) self-modifying across runs. Lowest rung that lets discoveries steer where they genuinely do; every rung above 1 names its signal and carries an integer cap. "The DAG is a trace, not a spec" is the consequence of rungs 3–6 — control flow is ordinary Python, so every rung is reachable without a framework construct.
5. **Apply the data-flow rule and the budget envelope.** Deterministic work is Python; structured JSON when code branches on it, prose when another LLM reads it; every loop, spawn, and recursion capped.

When quality disappoints after the build, escalate structure in order — sharpen the contract → decompose further → parallel perspectives → adversarial verification → more autonomy — before reaching for a bigger model.

---

## The five foundational principles

Every design the procedure produces has these five properties. They are consequences of the procedure, not a second framework — use them as the review checklist on your derived topology.

1. **Granular decomposition** (from step 1). Every reasoner does ONE cognitive thing — a small input, a small output (~2–4 flat attributes), a one-sentence API contract. If a reasoner's output has more than ~4 attributes or its body is more than ~30 lines, it is probably two reasoners.
2. **Guided autonomy** (from steps 2–3). A reasoner has freedom in HOW it answers, zero freedom in WHAT it answers. The orchestrator is a CEO — it sets the question and verifies the answer at the rung the stakes demand; it does not micromanage steps. The more capable the delegate, the less you control HOW and the more you verify WHAT.
3. **Dynamic orchestration** (from step 4). The graph adapts to intermediate state. Some branches fire, others don't. A meta-level reasoner can decide at runtime how many specialists to spawn, what to ask each one, and what to do with their answers. The DAG is a trace of these decisions, not a spec you committed to upfront — *this* is what no static chain framework can do.
4. **Contextual fidelity** (from step 5). The orchestrator is a context broker. Each call receives exactly what it needs — task description, relevant prior outputs, applicable constraints. Claims carry citation keys; provenance flows through every downstream reasoner to the final answer.
5. **Asynchronous parallelism** (from step 1). Cognitive jobs that don't depend on a sibling's output are independent by construction — anything independent must `asyncio.gather`. Sequential pipelines of independent work are always wrong.

Signals you meet during derivation map to structure: N independent analysis dimensions → fan out. Stakes that demand a frame separate from discovery → split discovery/refutation slots (rung 6). Investigation path depends on what was just found → meta-prompting (dynamism rung 4). Coverage matters but the answer's shape is unknown → fan-out → filter → gap-find → recurse (rung 5). System runs on inbound events → triggers as the entry surface.

**Named patterns are shapes you may discover you have built.** Read `references/patterns-emerge.md` after the topology exists, to check whether it has a name; never before. There is no preferred pattern — HUNT→PROVE is verification rung 6 wearing a domain costume, and earns its ~2× cost only when false positives are genuinely expensive.

---

## The two primitives that matter

Everything else is a variation.

- **`@app.reasoner()`** — every cognitive unit. Schemas derived from type hints. Calls other reasoners via `app.call(f"{app.node_id}.X", ...)`. Body can do anything Python can do.
- **`app.ai(system, user, schema, model, tools, ...)`** — the LLM call. Single-shot, or multi-turn tool-using when `tools=` is passed. `model=` is per-call. `schema=` returns a validated Pydantic instance. Every `.ai()` gate carries a `confident: bool` field and a fallback path.

Less-used but real:
- **`@app.skill()`** — deterministic functions you want callable through the control plane (no LLM).
- **`app.harness(prompt, provider="claude-code"|"codex"|"gemini"|"opencode")`** — delegates to an external coding-agent CLI. Heavy. **Only use when `af doctor` reports `harness_usable: true` AND the Dockerfile installs the CLI AND `shutil.which()` guards startup.** Otherwise use `app.ai(tools=[...])`.

Full signatures, schemas, router surface, memory scopes, and the cross-boundary serialization gotcha are in `references/primitives-snapshot.md` (offline-frozen). **Prefer the live `agentfield.ai/llms-full.txt`** when you have a network — it is the source of truth and it does not drift.

---

## Reasoners are APIs — design like a service mesh, not a chain

This is the single most important framing in the skill. **Treat each reasoner as a microservice.** Other reasoners call it the way one REST API calls another — recursively, at any depth, in any shape, in any direction. `app.call(f"{app.node_id}.X", ...)` is just a function call that happens to cross the control plane.

This is what no static chain framework can do:

- **LangChain / CrewAI / AutoGen / LangGraph** require you to declare the entire call graph upfront. The orchestrator is the only thing that calls anything. The graph is a static DAG drawn on a whiteboard.
- **AgentField** lets the call graph **emerge at runtime** from the reasoners' own intermediate decisions. The "orchestrator" body is just Python — `app.call` is just a function — so everything Python can do is available to your architecture.

Use this power. Build graphs with real depth:

- A reasoner deep inside a branch can call any other reasoner at any level.
- A reasoner can call itself recursively (with a depth cap) to drill into nested structure.
- A meta-reasoner can synthesize a brand new prompt at runtime and invoke a child reasoner with that prompt as a kwarg — the child's behavior is decided by a sibling's output.
- A reasoner can fan out `asyncio.gather` over N sub-reasoners where N itself was decided by an earlier reasoner.
- A reasoner can call a sub-reasoner, read the result, and conditionally decide whether to call a completely different reasoner next — the shape of the next layer is not committed until the current layer finishes.
- The same low-level reasoner (e.g., `confidence_scorer`) can be called from three different specialists in three different contexts — single source, three callers, three different inputs.

The only rule: every cross-reasoner call goes through `app.call`, never raw HTTP, so the control plane sees every edge for the workflow DAG, the cryptographic provenance chain, and the live observability surface.

**What this means for design:** do not constrain yourself to shapes you can draw on a whiteboard. Decompose, make each reasoner a narrowly-scoped callable, then let orchestrators invoke each other freely — deeply, conditionally, recursively, dynamically. The more the call graph depends on intermediate state, the more AgentField earns its place over LangChain-style frameworks.

If your final design has the entry reasoner as the only thing that calls `app.call`, or if your max depth from entry to leaf is 2, you have built a chain wearing the AgentField costume. Decompose further until each "specialist" is itself a small orchestrator that calls 2–4 sub-reasoners.

---

## Decision tree

```
What is this reasoner doing?

├─ Deterministic transform (sort, parse, dedupe, score-with-formula)?     → @app.skill() or plain helper
├─ Single classification, ≤4 flat fields, input fits ≤2k tokens?          → app.ai() with confident flag + fallback
├─ Multi-turn reasoning needing tools or iteration?                       → app.ai(tools=[...])
├─ Long input (document, transcript, corpus) needing navigation?          → @app.reasoner() that chunks + asyncio.gather over app.ai()
├─ Needs a real coding agent to write files / run shell?                  → app.harness() — only if the harness gate passes
└─ Composes multiple reasoners?                                           → @app.reasoner() that uses app.call() + asyncio.gather
```

**Bias:** many small `@app.reasoner` units. `@app.skill` for anything code can do. `app.ai` with explicit prompts and a `confident` flag. Reserve `app.harness` for actual coding-agent delegation.

This tree is the autonomy spectrum (procedure step 2) turned into questions. Each branch down trades process visibility for capability: `app.skill` is fully deterministic, `app.ai` verifies instantly on the schema, `app.harness` verifies only at the boundary. Pick the leftmost point that solves the problem, and pair every step right with the verification rung that step requires.

---

## Workflow

1. **Announce** — tell the user you're using the `agentfield` skill.
2. **Fetch live docs** — `WebFetch https://agentfield.ai/llms.txt` (small index). Pull `/llms-full.txt` or per-page `/llm/docs/<slug>` only when you need depth. Cache. See `references/live-docs.md`.
3. **Probe environment** — `af doctor --json`. Read `recommendation.provider`, `recommendation.ai_model`, `recommendation.harness_usable`, `provider_keys.*.set`, `control_plane.reachable`.
4. **Pick the model** — `references/model-selection.md`. If `af doctor` recommends a model, use it. If no provider key, ask. If OpenRouter is present but no explicit pick, query `https://openrouter.ai/api/v1/models` for current cheap open-weight options and offer them.
5. **Clarify if needed** — only for architecture-changing ambiguity. Use `AskUserQuestion` with 1–3 narrow choices.
6. **Derive the topology** by running the procedure in `references/mental-models.md`. Then read `references/examples-map.md`, find the live example whose problem shape is closest, and grep its code for decomposition discipline — do not copy its topology. Only after your shape exists, open `references/patterns-emerge.md` to check whether it has a name.
7. **Scaffold** — `af init <slug> --language python --docker --defaults --non-interactive --default-model <model>`. Then **rewrite `main.py` and `reasoners.py`** with your real architecture per `references/scaffold-recipe.md`. Generate `CLAUDE.md` from `references/project-claude-template.md`.
8. **Verify** — `python3 -m py_compile`, `docker compose config`, then `docker compose up --build`. Run the build checks in `references/verification.md`. Use `af agent discover -q "<slug>"` and `af agent query --resource executions` for live introspection — see `references/cli-toolkit.md`.
9. **Smoke test live** — fire the canonical **async** curl (multi-reasoner pipelines exceed the 90s sync limit). Poll until `status: succeeded` with a real `result`. Static checks alone are not a green light. See "Mandatory live smoke test" below.
10. **Hand off** — use the output contract at the bottom of this file.

---

## Inter-reasoner data flow

| Data purpose | Format | Why |
|---|---|---|
| Drives code routing (`if result.type == "X"`) | Structured JSON | Code consumes it |
| Becomes another LLM's context | Natural-language string | LLMs reason over prose, not serialized dicts |
| Both | Hybrid — JSON for code, prose for the LLM | |

**Cross-boundary gotcha:** `app.call` crosses a serialization boundary. A Pydantic model goes in; a plain dict comes out — regardless of the receiver's type hints. Either reconstruct on the receiver (`Model(**payload)`) or render to prose before the call. The only test that catches this is the live smoke test.

---

## Mandatory patterns (every build)

1. **Per-request model propagation.** Entry reasoner accepts `model: str | None = None` and threads it through every `app.ai(..., model=model)` and `app.call(..., model=model)`. Child reasoners accept and use it identically. Users override per request via `{"input": {..., "model": "..."}}`.
2. **Routers when reasoners > 4.** `AgentRouter(prefix="", tags=["domain"])` + `app.include_router(router)`. Inside a router file use `NODE_ID = os.getenv("AGENT_NODE_ID", "<slug>")` — `router.node_id` does NOT exist.
3. **`tags=["entry"]` on the public entry reasoner** so discovery picks it up.
4. **Every `.ai()` schema has a `confident: bool` field and the call site has a fallback path** (verification rung 4). Three valid fallbacks: (a) escalate to a deeper reasoner, (b) return a safe-default Pydantic instance (`REFER_TO_HUMAN` / `NEEDS_REVIEW` — recommended for regulated systems), (c) escalate to `app.harness()` if and only if the harness gate passes.

---

## Hard rejections — refuse without negotiation

| ❌ | ✅ |
|---|---|
| Pattern-first design ("this looks like HUNT→PROVE") | Derive from cognitive jobs; name the shape afterwards |
| Direct HTTP between reasoners | `app.call(f"{app.node_id}.X", ...)` |
| One giant reasoner doing 5 things | Decompose into 5 + orchestrate with `app.call` + `asyncio.gather` |
| Static linear chain when the path depends on findings | Dynamic routing on intermediate state |
| `app.ai(prompt=full_50_page_doc)` | Chunk + fan out, or `app.ai(tools=[...])`, or `app.harness` |
| `while not confident: ...` (unbounded) | `for _ in range(MAX): ...` with explicit break |
| Structured JSON shoved into another LLM as context | Render to prose first |
| `app.ai("sort these by score")` | `sorted(items, key=...)` — code does code work |
| Scaffold without a working live curl | Smoke test or it didn't happen |
| Multi-container fleet for what one node would do | One agent node, many reasoners |
| Hardcoded `node_id` in `app.call("slug.X", ...)` | `app.call(f"{app.node_id}.X", ...)` |
| Hardcoded model string | `AI_MODEL` env + per-request `model=` override |
| `.ai()` schema with no `confident` field, no fallback | Always include and always check |
| `app.harness()` in a default scaffold (no CLI in container) | `app.ai(tools=[...])` or chunked-loop reasoner |
| `input_schema=` / `output_schema=` / `description=` on `@app.reasoner()` | Those don't exist; schemas come from type hints |
| `app.serve()` in `__main__` | `app.run()` — auto-detects CLI vs server |
| Pydantic instance passed across `app.call(...)` expecting reconstitution | Reconstruct `Model(**payload)` on receiver, or render prose on sender |

Full deep-dive in `references/anti-patterns.md`. Rationalization counters in the same file.

When a user explicitly demands a rejected pattern, name the rejection, give the one-sentence reason, propose the AgentField alternative, and only build it their way after they confirm they understand. Add a `# NOTE: User requested X over canonical Y` comment.

---

## Mandatory live smoke test

A build is not done until the canonical async curl has been fired against the live stack and returned `status: "succeeded"` with a real reasoned `result`. Static checks (`py_compile`, `docker compose config`) prove syntax, not contract. They will not catch cross-boundary deserialization bugs, surface contract drift, or a sub-reasoner returning `confident=False` and propagating the safe default downstream.

```bash
# Bring it up
docker compose up --build -d

# Wait for registration via the durable discovery endpoint
for i in $(seq 1 15); do
  READY=$(curl -fsS http://localhost:8080/api/v1/discovery/capabilities 2>/dev/null \
    | jq -r '.capabilities[] | select(.agent_id=="<slug>") | .agent_id')
  [ -n "$READY" ] && break
  sleep 2
done

# Fire the async curl with realistic input
EXEC_ID=$(curl -sS -X POST http://localhost:8080/api/v1/execute/async/<slug>.<entry> \
  -H 'Content-Type: application/json' \
  -d @./sample_payload.json | jq -r '.execution_id')

# Poll until done
while :; do
  R=$(curl -sS http://localhost:8080/api/v1/executions/$EXEC_ID)
  S=$(echo "$R" | jq -r '.status')
  case "$S" in
    succeeded) echo "$R" | jq '.result'; break ;;
    failed)    echo "$R" | jq '.'; docker compose logs <slug> --tail=100; exit 1 ;;
    *)         sleep 2 ;;
  esac
done
```

Common runtime failures that only surface here: `AttributeError: 'dict' has no attribute '<X>'` (cross-boundary reconstitution), `AttributeError: '<framework>' has no attribute '<X>'` (surface contract drift — check the live docs), `TypeError: argument after ** must be a mapping` (same boundary issue), or an empty result (an upstream `confident=False` cascaded as safe-default).

---

## Output contract

Final message to the user — clean, copy-pasteable, in this order:

1. **What was scaffolded** — file tree with absolute paths.
2. **Architecture sketch** — 4–6 bullets: each reasoner's role, who calls whom, where the dynamic decision is, where safety guardrails fire.
3. **Assumptions** — 5–10 bullets the user can correct on iteration 2.
4. **🚀 Run it** — `cp .env.example .env`, paste the key, `docker compose up --build`.
5. **🌐 Open the UI** — `http://localhost:8080/ui/` + the discovery endpoint URL.
6. **✅ Verify** — the discovery/capabilities check (primary; durable across CP versions).
7. **🎯 Try it** — the canonical async curl with realistic data the user can run as-is. If the brief included sample data, use *that* data verbatim.
8. **🏆 Showpiece** — the verifiable workflow chain via `/api/v1/did/workflow/$WF/vc-chain`. No other framework gives this. Mention it.
9. **Next iteration upgrade** — one concrete suggestion tailored to the shape you actually built.

---

## TypeScript and Go

A TypeScript SDK exists (`sdk/typescript/`) and a Go SDK exists (`sdk/go/`). **Default to Python** unless the user explicitly asks otherwise — every reference and recipe in this skill is Python-first. For TS/Go, fetch the corresponding page from `agentfield.ai/llms-full.txt` and adapt; the shape is the same.

---

## Reference table — load when

| File | Load when |
|---|---|
| `references/live-docs.md` | **Every invocation** — first thing, fetches the SDK truth |
| `references/cli-toolkit.md` | **Every invocation** — `af doctor` + `af agent` are the introspection surface |
| `references/model-selection.md` | Choosing the model — always |
| `references/mental-models.md` | Once per design session, before drawing the topology — the generative theory: cognitive jobs, autonomy spectrum, verification ladder, dynamism ladder, quality escalation |
| `references/patterns-emerge.md` | After the topology exists — post-hoc naming so humans can review the shape |
| `references/examples-map.md` | Finding the closest live example to grep for shape inspiration |
| `references/primitives-snapshot.md` | **Offline only** — when you cannot fetch live docs |
| `references/scaffold-recipe.md` | Actually writing files / compose / Dockerfile |
| `references/verification.md` | The build checks, troubleshooting, async vs sync |
| `references/triggers.md` | Use case is event-driven (webhook) or scheduled (cron) |
| `references/project-claude-template.md` | Generating the per-project CLAUDE.md (always) |
| `references/anti-patterns.md` | When tempted to take a shortcut, or when the user pushes back on a rejection |

Reference files are one level deep from this file. If a reference points at another, come back here and load the second directly.

---

## Bottom line

Your output is judged by three things:

1. **Does the curl return a real reasoned answer?**
2. **Does the architecture look like composite intelligence?** — parallelism, dynamic decisions, decomposition deeper than 2 layers.
3. **Can a future agent extend it without breaking the contract?** — CLAUDE.md present, anti-patterns listed, the live-docs pointer documented.

If all three hold, you've done it right.
