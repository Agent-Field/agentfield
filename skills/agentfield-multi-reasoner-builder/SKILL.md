---
name: agentfield-multi-reasoner-builder
description: Architect and ship a complete multi-agent backend system on AgentField from a one-line user request. Use when the user asks to build, scaffold, design, or ship an agent system, multi-agent pipeline, reasoner network, AgentField project, financial reviewer, research agent, compliance agent, or any LLM composition that should outperform LangChain/CrewAI/AutoGen — especially when they want a runnable Docker-compose stack and a working curl smoke test.
---

# AgentField Multi-Reasoner Builder

You are not a prompt engineer. You are a **systems architect** building composite reasoning machines on AgentField. The intelligence is in the composition, not the components.

## HARD GATE — READ BEFORE ANYTHING ELSE

> **Do NOT write any code, generate any file, or scaffold any project until you have:**
> 1. Either (a) asked the ONE grooming question and received an answer, OR (b) confirmed that the user's first message ALREADY contains a clear use case — in which case **skip the question and proceed straight to design**. The "build now, key later" rule (below in the grooming protocol) ALWAYS overrides this gate when the brief is complete; you do NOT need a key in chat to start building because the user will paste it into `.env` themselves
> 2. Read `references/choosing-primitives.md` (mandatory — sets the philosophy and the real SDK signatures)
> 3. Designed the reasoner topology **from the problem up, not from a template down**. The shape depends on what the problem actually needs (see "Reasoners are software APIs" below and `references/architecture-patterns.md`). Do not copy a previous build's shape unless the problem is the same shape.
>
> **Do NOT default to a single big reasoner with one `app.ai` call.** That's a CrewAI clone. Decompose.
>
> **Do NOT default to a single fat orchestrator that calls every specialist directly in one fan-out.** That's a star pattern, also a CrewAI clone wearing a different costume. Build deep call chains.
>
> **Do NOT default to HUNT→PROVE or any adversarial pattern.** HUNT→PROVE is ONE architectural option out of many. It only earns its cost when false positives are genuinely expensive (medical, legal, financial, security, regulated verdicts). Routing, extraction, generation, research, content pipelines, data enrichment, orchestration — none of these need an adversary. Pick the pattern that matches the problem, not the pattern you just saw in an example.
>
> If you cannot draw your system as a non-trivial graph **with depth ≥ 3** AND explain in one sentence why the shape matches the problem, you have not architected anything.
>
> Violating the letter of this gate is violating the spirit of the gate. There are no exceptions for "simple" use cases.

## The unit of intelligence is the reasoner — treat them as software APIs

This is the most important framing in the entire skill. **Each reasoner is a microservice. Reasoners call other reasoners the way one REST API calls another.** The orchestrator at the top is not the only thing that calls reasoners — every reasoner can (and often should) call sub-reasoners that are themselves further decomposed.

The shape of the DAG is never picked from a menu. It is **derived from the problem by walking the five foundational principles below**. A great architect reaches the right topology by asking the right questions in order, and the shape falls out of the principles. There is no catalog to copy from. There is no "this kind of problem gets this kind of shape." Every use case is different and every topology is the consequence of applying the principles honestly.

### What we are actually doing

A single LLM call reasons at roughly 0.3 on a 0.0–1.0 scale. It pattern-matches well in narrow sprints, but it is shallow, brittle, and cannot plan across steps. You cannot prompt-engineer your way to 0.7 or 0.8. You can only **compose your way there**.

A composite system of ten 0.3-grade reasoners connected deliberately can outperform a single 0.4-grade call by 5–10×, because **the architecture itself encodes intelligence** about how to decompose the problem, how to allocate cognitive work across specialized frames, how to combine partial results, and how to stay coherent across steps. The whole becomes greater than the sum of its parts.

You are not a prompt engineer. You are a systems architect. Your job is to engineer the cognitive graph. The LLMs are interchangeable components.

The value we deliver is **autonomous thinking at multiple levels**. Anything a deterministic function could do belongs in code, not an LLM. LLMs are reserved for judgment, discovery, synthesis, pattern-spotting, and decisions that cannot be encoded as rules. Every LLM call should be earning its place by doing something a `for` loop genuinely cannot.

### The five principles — apply each one in order

**1. Granular decomposition is mandatory.**
No single reasoner is trusted to solve a complex task. Decompose the problem into the smallest logical, independent sub-tasks — the atomic reasoning units. Each unit does ONE cognitive thing, takes a small well-shaped input, and returns a small well-shaped output. The schema constraint is a forcing function: if a reasoner's output has more than ~4 flat attributes, it is probably two reasoners glued together. Complex outputs are assembled from multiple simple calls; they are never generated in a single call.

> *"What is the simplest meaningful cognitive question I can ask at each step?"*

**2. Guided autonomy, not free autonomy.**
Every reasoner has freedom to USE its intelligence inside its assigned role, but zero freedom to redefine the role. The orchestrator is a CEO: it sets objectives, allocates context, defines success, and verifies outcomes. It does not micromanage steps. A reasoner chooses HOW to answer its question; it does not choose WHICH question to answer. This is what separates "guided" systems (which ship) from "autonomous" ones (which hallucinate their way off mission).

> *"What is this reasoner's one-sentence scope, and what is the one-sentence verification test for its output?"*

**3. Dynamic orchestration — the graph adapts to intermediate state.**
A static pipeline A → B → C is a useful starting point but it is not where the intelligence lives. Real power comes from graphs whose shape changes based on what the system just discovered: different branches fire, different parameters flow, different sub-reasoners are invoked depending on what a prior reasoner returned. A meta-level reasoner can decide at runtime how many specialists to spawn, what exactly to ask each one, and how to combine them. The graph is responsive to its own intermediate state — this is the "meta-level" where the output of A literally determines the structure of B's subsystem.

> *"At which points does my graph's structure need to change based on something the system learned mid-run?"*

**4. Contextual fidelity — the orchestrator is a context broker.**
A reasoner's performance is a direct function of the context it receives. Too little and it guesses. Too much and it drowns. The orchestrator's most important engineering task is to assemble precisely the right context for each call: task description, relevant prior outputs, applicable constraints — nothing else. When a reasoner emits a claim, it also emits a citation key back to the source, and the orchestrator carries that key through every downstream reasoner. The final output is not just correct; it is verifiable.

> *"What is the minimum context each reasoner needs, and how is provenance carried through the whole chain?"*

**5. Asynchronous parallelism — decompose to parallelize.**
The moment a problem is decomposed into independent sub-tasks, those sub-tasks should run concurrently. A hundred focused reasoners running in parallel for two seconds can process, analyze, and synthesize at a scale and speed impossible for any sequential process. Parallelism is not a nice-to-have; it is how we overcome the "small, dumb LLM" constraint. If your pipeline runs sequentially when the pieces don't depend on each other, either your decomposition is wrong or your orchestration is wrong.

> *"Which reasoners genuinely depend on which others, and can everything that doesn't `asyncio.gather` together?"*

### What the principles produce

When you apply all five to your specific problem, **the topology emerges on its own**. You never pick from a menu of named shapes.

- **Decomposition produces depth.** Each reasoner has sub-reasoners, which have sub-reasoners. The DAG grows downward until every leaf is an atomic cognitive unit with a one-sentence API contract.
- **Dynamic orchestration produces branching.** Wherever the path depends on intermediate state, you get routing decisions instead of static edges. Some branches fire, others don't, and a meta-layer may decide at runtime which specialists to invoke and how.
- **Contextual fidelity produces clean data flow.** Claims carry provenance. Partial results carry exactly what the next step needs and nothing more.
- **Asynchronous parallelism produces fan-out at every layer.** Independent sub-tasks run concurrently wherever they appear — not just at the top.
- **Guided autonomy produces specialization.** Every reasoner has a narrow frame, a clear API contract, and a verification test the orchestrator can apply to its output.

**If your final topology does not have depth ≥ 3, does not parallelize wherever work is independent, and has no place where the shape depends on intermediate state, you did not apply the principles deeply enough.** Go back and ask the five questions again.

### Bad shape — flat star (the default a coding agent will reach for)

```
entry_orchestrator
├── specialist_1   ──┐
├── specialist_2   ──┤
├── specialist_3   ──┼── all called once, in parallel, by the orchestrator
├── specialist_4   ──┤
└── specialist_5   ──┘
        │
        v
   synthesizer
```

Depth = 2. Every sub-task is a sibling of every other sub-task. There is no sub-decomposition, no branching on intermediate state, no meta-level decision about what to invoke or how to invoke it. This is the shape the principles reject by default. If your design lands here, you stopped applying principle 1 (decomposition) and principle 3 (dynamic orchestration) too early. It is `asyncio.gather([llm_call_1, llm_call_2, ...])` with extra ceremony. Go back and ask the five questions again.

### Non-negotiable invariants (apply regardless of what shape the principles produce)

- **Every reasoner has a one-sentence API contract** you could write on a sticky note. If you can't, it is doing too much.
- **Every reasoner produces a flat output** of 2–4 attributes. Complex outputs are assembled from multiple simple calls; never generated in a single call.
- **Every reasoner receives only the context it needs** — never the kitchen sink.
- **Claims carry citation keys.** Provenance flows through the whole graph.
- **Independent work runs in parallel.** Sequential pipelines of independent steps are always wrong.
- **Deterministic work lives in `@app.skill()` or plain helpers** — never use an LLM for anything a `for` loop could do. The value we deliver is intelligence; anything programmatic belongs in code.
- **Depth ≥ 3 layers** from entry to leaf. Two layers means you stopped decomposing too early.
- **At least one place where the graph's shape depends on intermediate state.** If every input produces the exact same DAG, you are not using dynamic orchestration — a script would have worked and you did not need AgentField.
- **Reasoners do not redefine their own roles.** The orchestrator sets the frame; each reasoner has freedom inside it, not over it.

### Reference patterns live in `architecture-patterns.md`

When you have walked the five principles and want to sanity-check your topology against known-good compositions (parallel hunters, HUNT→PROVE, streaming, meta-prompting, control loops, fan-out→filter→gap-find, reactive enrichment, etc.), read `references/architecture-patterns.md`. Treat those patterns as **names for emergent consequences of the principles**, not as a menu to pick from. They are useful vocabulary for describing what you built, not templates to copy into a new problem.

**The unit of intelligence is the reasoner. Apply the five principles to your problem. The shape will emerge.**

## The non-negotiable promise

Every invocation of this skill must end with the user able to run a small set of commands and see a real reasoned answer come back.

```bash
# 1. Bring the stack up
docker compose up --build

# 2. Kick off the entry reasoner (async — returns an execution_id immediately)
EXEC_ID=$(curl -sS -X POST http://localhost:8080/api/v1/execute/async/<node>.<entry_reasoner> \
  -H 'Content-Type: application/json' \
  -d '{"input": {"...": "..."}}' | jq -r '.execution_id')

# 3. Poll until done and print the result
while :; do
  R=$(curl -sS http://localhost:8080/api/v1/executions/$EXEC_ID)
  S=$(echo "$R" | jq -r '.status')
  case "$S" in
    succeeded) echo "$R" | jq '.result'; break ;;
    failed)    echo "$R" | jq '.'; break ;;
    *)         sleep 2 ;;
  esac
done
```

If you cannot deliver that, you have failed. No theoretical architectures. No "here's how you would do it." A running stack and a real reasoned answer.

**Why async by default:** the control plane enforces a hard **90-second timeout** on the sync endpoint `POST /api/v1/execute/<target>`. Any deep composition (parallel specialists, meta-level spawning, multi-layer fan-out) can easily exceed that. The async endpoint `POST /api/v1/execute/async/<target>` returns an `execution_id` immediately and the caller polls `GET /api/v1/executions/<id>` — no time budget, no ceiling, no hanging curl. **Always use async in the canonical smoke test.** Use the sync endpoint only when you can genuinely guarantee the entire pipeline finishes in under 90 seconds.

**Note the request body shape: `{"input": {...kwargs...}}`** — the control plane wraps reasoner kwargs in an `input` field. Verified against `control-plane/internal/handlers/execute.go`. Many coding agents get this wrong.

## Workflow (universal — works for any coding agent)

1. **Announce** you're using the `agentfield-multi-reasoner-builder` skill.
2. **Probe the environment** with `af doctor --json` (one command, see "Environment introspection" below). This tells you which provider keys are set, which harness CLIs are present, and the recommended `AI_MODEL`. Use this output instead of guessing.
3. **Ask the one grooming question** (below) ONLY if the user hasn't already provided everything.
4. **Read `choosing-primitives.md` ALWAYS.** Read other references when their trigger fires (table below).
5. **Design the topology** before writing files.
6. **Lay down infrastructure** with `af init <slug> --language python --docker --defaults --non-interactive --default-model <model_from_doctor>` (one command, see "Infrastructure scaffold" below).
7. **Customize `main.py` and `reasoners.py`** with the real reasoner architecture per `scaffold-recipe.md`. Generate `CLAUDE.md` (from `project-claude-template.md`) and `README.md` AFTER you know the entry reasoner name and the curl payload.
8. **Validate**: `python3 -m py_compile main.py`, `docker compose config`, ideally `docker compose up --build` + verification ladder.
9. **Hand off** with the output contract below.

## Environment introspection: `af doctor`

Run this **once** at the start of every build. It returns ground truth about the local environment in a single JSON document instead of having you probe `which`, `env`, `docker image inspect`, etc. yourself:

```bash
af doctor --json
```

Key fields you'll consume:
- `recommendation.provider` — `openrouter` / `openai` / `anthropic` / `google` / `none`
- `recommendation.ai_model` — the LiteLLM-style model string to bake into the scaffold's `AI_MODEL` default
- `recommendation.harness_usable` — `true` only if at least one of `claude-code` / `codex` / `gemini` / `opencode` is on PATH. **If `false`, do not use `app.harness()` in the scaffold under any circumstance.**
- `recommendation.harness_providers` — list of available CLI names (use these as the `provider=` value if and only if `harness_usable` is true)
- `provider_keys.{name}.set` — boolean per provider (no values leaked)
- `control_plane.docker_image_local` — whether `agentfield/control-plane:latest` is already cached (informs whether the first `docker compose up` will need to pull)
- `control_plane.reachable` — whether a control plane is already running locally (so you can curl test reasoners against it before building your own)

**Use the doctor's output to set the `--default-model` flag on `af init` and to decide whether `app.harness()` is even an option in the architecture.** Do not hardcode your assumptions about the environment.

## Infrastructure scaffold: `af init --docker`

Run this **once** after `af doctor` and your architecture design. It produces the four infrastructure files that you should not customize plus the language scaffold (Python `main.py`, `reasoners.py`, `requirements.txt`):

```bash
af init <slug> --language python --docker --defaults --non-interactive \
  --default-model <model_string_from_doctor>
```

What it generates:
- `Dockerfile` — universal Python 3.11-slim, builds from project dir, no repo coupling
- `docker-compose.yml` — control-plane + agent service with healthcheck and service-healthy gating
- `.env.example` — all four provider keys (OpenRouter, OpenAI, Anthropic, Google) and `AI_MODEL` with the doctor-recommended default
- `.dockerignore`
- `main.py`, `reasoners.py`, `requirements.txt`, `README.md`, `.gitignore` — the standard language scaffold (you'll **rewrite `main.py` and `reasoners.py`** with your real architecture)

What it does NOT generate (intentionally):
- `CLAUDE.md` — you generate this from `references/project-claude-template.md` AFTER writing the real reasoners, so it can name them and justify the architecture
- A README with the real curl — the default `README.md` is generic; you replace it AFTER picking the entry reasoner so the curl uses real kwargs

The four infrastructure files are zero-change for the agent: Dockerfile installs `agentfield` from `requirements.txt` and copies the project dir; compose wires control-plane + agent with healthcheck; `.env.example` exposes all providers; `.dockerignore` covers the standard cases. **Do not modify them unless you have a real reason.**

## Reference table — load when

| File | Load when |
|---|---|
| `choosing-primitives.md` | **Every invocation** — before any code |
| `architecture-patterns.md` | Designing inter-reasoner flow / picking the right shape — sequential cascade, parallel fan-out, dynamic routing, streaming, meta-prompting, HUNT→PROVE (only when false positives are expensive), etc. |
| `scaffold-recipe.md` | Actually writing files / docker-compose / Dockerfile |
| `verification.md` | Writing the smoke test ladder or declaring done |
| `project-claude-template.md` | Generating the per-project CLAUDE.md (always) |
| `anti-patterns.md` | When tempted to take a shortcut OR when the user pushes back on a rejection |

Reference files are one level deep from this file. Do not nest reads — if a reference points at another reference, come back here and load the second one directly.

## The grooming protocol (1 question, then build)

Ask **exactly one** question and **one** key request. Nothing else upfront:

> "Tell me in 1–2 sentences what you want this agent system to do, and paste your provider key. We support OpenRouter (default), OpenAI, or Anthropic — any LiteLLM-compatible model. Example: `OPENROUTER_API_KEY=sk-or-v1-...`"

**Skip-the-question rule:** if the user's first message ALREADY contains a clear use case, do NOT ask the grooming question — even if they didn't paste a provider key. This is the **"build now, key later"** policy:

- If the user gives a clear use case AND a provider key → proceed straight to design + build
- If the user gives a clear use case AND says they'll paste the key into `.env` later → ALSO proceed straight to design + build. The scaffold will work with `OPENROUTER_API_KEY=sk-or-v1-FAKE` for `docker compose config` validation. The user runs the real key from `.env` when they're ready
- If the user gives a clear use case AND says nothing about a key → proceed straight to design + build. The `.env.example` you generate makes it obvious where to put the key
- If the user's request is genuinely vague or ambiguous along an architecture-changing axis → THEN ask one question

The point is to **never block the build on a key the user is going to drop into `.env` themselves**. Asking a redundant question after the user has already given you the use case wastes their time and signals you're following a script instead of understanding.

Then proceed. Infer everything else from the use case. State your assumptions in the final handoff so the user can correct them in iteration 2.

**Only ask follow-up questions if the use case is genuinely ambiguous along an axis that changes the architecture** (not the wording). Examples that warrant a follow-up:

- Input is a 200-page document vs. a small JSON payload (changes whether you need a navigator harness)
- Output must include verifiable citations (changes whether you need a provenance reasoner)
- Synchronous request/response vs. event-driven (pattern 8 vs. patterns 1–7)

Examples that do **NOT** warrant a follow-up: model preference, file naming, port number, code style, what to call the entry reasoner. Decide and state.

## The five primitives (cheat sheet — full detail in `choosing-primitives.md`)

- **`@app.reasoner()`** — every cognitive unit. Schemas come from **type hints** (no `input_schema=` param exists).
- **`@app.skill()`** — deterministic functions. No LLM. Use whenever an LLM call is overkill.
- **`app.ai(system, user, schema, model, tools, ...)`** — single OR multi-turn LLM call. `tools=[...]` makes it stateful. `model="..."` per call overrides AIConfig default.
- **`app.harness(prompt, provider="claude-code"|"codex"|"gemini"|"opencode")`** — delegates to an external coding-agent CLI. **Not** a generic tool-using LLM (that's `app.ai(tools=[...])`). **REQUIRES** the chosen provider's CLI to be installed inside the agent container — see "Harness availability gate" below.
- **`app.call(target, **kwargs)`** — inter-reasoner traffic THROUGH the control plane. Returns `dict`. **No model override param** — thread `model` as a regular reasoner kwarg.

**The bias:** many small `@app.reasoner()` units. `@app.skill()` for anything code can do. `app.ai()` with explicit prompts. Reserve `app.harness()` for real coding-agent delegation.

## Harness availability gate (READ BEFORE USING `app.harness()`)

`app.harness()` runs an external coding-agent CLI inside the agent container — `claude-code`, `codex`, `gemini`, or `opencode`. **The default `python:3.11-slim` Docker image has none of these installed.** A scaffold that uses `app.harness()` without installing the CLI in the Dockerfile will crash at runtime.

**The check is automated.** `af doctor --json` reports `recommendation.harness_usable` (true/false) and `recommendation.harness_providers` (the list of CLIs on PATH). Use the doctor output as the source of truth — do not assume.

**Default rule:** scaffolds **MUST NOT** use `app.harness()` at all when `recommendation.harness_usable == false`. Use `app.ai(tools=[...])` for stateful reasoning, or a `@app.reasoner()` that loops `app.ai()` for chunked work. These work in the default container with zero extra setup.

**You may use `app.harness()` ONLY when ALL of the following are true:**

1. The use case **genuinely requires a real coding agent** in the loop — i.e. the reasoner needs to write/edit files on disk, run shell commands, or perform complex non-LLM coding work that `app.ai(tools=[...])` cannot do.
2. You modify the Dockerfile to install the chosen provider's CLI. Example for Claude Code:
   ```dockerfile
   RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
       && npm install -g @anthropic-ai/claude-code \
       && rm -rf /var/lib/apt/lists/*
   ```
3. You add a **startup availability check** in `main.py` that fails fast with a clear error if the CLI is not on PATH:
   ```python
   import shutil, sys
   if not shutil.which("claude"):  # or "codex" / "gemini" / "opencode"
       print("ERROR: app.harness(provider='claude-code') requires the `claude` CLI in PATH.", file=sys.stderr)
       sys.exit(1)
   ```
4. The README explicitly tells the user that the agent container ships with `claude-code` (or whatever) and explains the consequence on image size.

**If any of the four are not satisfied, do not use `app.harness()`.** Refactor the reasoner to use `app.ai(tools=[...])` or a chunked `@app.reasoner()` loop. There is no scenario where it's OK to write `app.harness(provider="claude-code")` in code that ships in a container without the `claude` binary.

When in doubt: **don't use harness.** The user can ask for it in iteration 2. The first build's job is to work on `docker compose up` with zero external CLI dependencies.

## Mandatory patterns (every build must have all three)

### 1. Per-request model propagation

The entry reasoner accepts `model: str | None = None` and threads it through every `app.ai(..., model=model)` and `app.call(..., model=model)`. Child reasoners accept `model` the same way and use it. The user can A/B test models per request:

```bash
curl -X POST http://localhost:8080/api/v1/execute/<slug>.<entry> \
  -d '{"input": {"...": "...", "model": "openrouter/openai/gpt-4o"}}'
```

If `model` is omitted, the AIConfig default from the env var `AI_MODEL` is used. **`app.call()` has no native model override — you MUST thread model through reasoner kwargs.**

### 2. Routers when reasoners > 4

Use `AgentRouter(prefix="domain", tags=["domain"])` and `app.include_router(router)` to split reasoners into separate files. Tags merge between router and per-decorator. **Note:** `prefix="clauses"` auto-namespaces reasoner IDs as `clauses_<func_name>` — call them as `app.call(f"{app.node_id}.clauses_<func_name>", ...)`.

### 3. Tags on the entry reasoner

The public entry reasoner is decorated with `tags=["entry"]` so it surfaces in the discovery API. Tags are free-form (not reserved); use domain tags for internal reasoners.

## Hard rejections — refuse these without negotiation

| ❌ Rejected pattern | ✅ AgentField alternative |
|---|---|
| Direct HTTP between reasoners (`httpx.post(...)`) | `await app.call(f"{app.node_id}.X", ...)` — control plane needs to see every call to track DAG, generate VCs, replay |
| One giant reasoner doing 5 things | Decompose into 5 reasoners coordinated by an orchestrator using `app.call` + `asyncio.gather` |
| Static linear chain `A → B → C → D` (always, no routing) | Dynamic routing: intake reasoner picks downstream reasoners based on what it found |
| `app.ai(prompt=full_50_page_doc)` | `@app.reasoner` that loops `app.ai` per chunk, OR `app.ai(tools=[...])` with explicit tool calls |
| Unbounded `while not confident: app.ai(...)` | Hard cap: `for _ in range(MAX_ROUNDS): ...` with explicit break |
| Passing structured JSON between two LLM reasoners | Convert to prose. LLMs reason over natural language, not JSON serialization |
| Replicating sort/dedup/score work with `app.ai` | `@app.skill()` with plain Python |
| Scaffold without a working `curl` that returns real output | The promise is `docker compose up` + curl. Always include it |
| Multi-container agent fleet when one node would do | One agent node, many reasoners — unless there's a real boundary |
| Hardcoded `node_id` in `app.call("financial-reviewer.X", ...)` | `app.call(f"{app.node_id}.X", ...)` — survives `AGENT_NODE_ID` rename |
| Hardcoded model | `model=os.getenv("AI_MODEL", default)` AND per-request override via reasoner kwarg |
| `app.ai()` schema with no `confident` field and no fallback | Schema must include `confident: bool`, call site checks it and escalates |
| `app.harness(provider="claude-code")` in a default scaffold | Default container has no `claude` CLI. Use `app.ai(tools=[...])` or a chunked-loop reasoner. See "Harness availability gate" |
| `input_schema=` or `output_schema=` parameter on `@app.reasoner` | These don't exist. Schemas come from type hints |
| `app.serve()` in `__main__` | `app.run()` — auto-detects CLI vs server mode |
| Passing a Pydantic instance (or a list/dict of them) across `app.call` and expecting the receiver to get the instance | **Cross-boundary data never auto-reconstitutes.** The receiver gets plain `dict` / `list[dict]` regardless of type hints. Either reconstruct explicitly on the receiving side (`Model(**payload)`) OR render to prose in the caller and pass a string. See `choosing-primitives.md` → "Cross-boundary data does NOT auto-reconstitute" |
| Trusting prose descriptions of framework contracts with words like "every", "all", "transparently forwards" | **Surface contracts are always narrower than the words describing them.** Verify the exact attribute or method you plan to use against the enumerated list in `choosing-primitives.md`. If it's not on the list, treat it as unsupported |
| Treating static validation (`py_compile`, `docker compose config`, type checks) as sufficient proof the build works | Static checks catch syntax and shape, not contract drift. A build is not done until the canonical async curl has been fired against the live stack and returned `status: "succeeded"` with a real reasoned `result`. See "Mandatory live smoke test before handoff" |

When the user explicitly demands a rejected pattern, name the rejection, explain *why* in one sentence, propose the AgentField alternative, and only build it their way after they've confirmed they understand the tradeoff. Add a `# NOTE: User requested X over canonical Y` comment.

## Mandatory live smoke test before handoff

Static validation — `python3 -m py_compile`, `docker compose config`, visual invariants, type checks — is **necessary but not sufficient**. None of those checks exercise the live call graph. None of them catch contract drift between reasoners. None of them reveal runtime errors like a cross-boundary type mismatch, a missing proxied attribute, a wrong env var name, or an unreachable sub-reasoner.

**A build is not done until the canonical async curl from section 7 of the Output Contract has been fired against the live stack and returned `status: "succeeded"` with a real reasoned `result` payload.** No exceptions.

The required sequence before you tell the user "it's ready":

1. `docker compose up --build -d` — bring the stack up.
2. Poll `/api/v1/discovery/capabilities` (see section 6) until the agent is registered and every reasoner you defined is listed.
3. Fire the canonical async curl from the README with realistic input.
4. Poll `/api/v1/executions/<id>` until `status` transitions to either `succeeded` or `failed`.
5. If `status == "succeeded"`: read the `result` field and confirm it is a real structured response from your entry reasoner — not an empty object, not a fallback, not a half-finished dict.
6. If `status == "failed"`: read the `error` field AND tail the agent container logs (`docker compose logs <slug> --tail=100`) to find the actual stack trace. Fix the bug. Go back to step 1.

**Do not hand off a build that has only passed static checks.** Static validation means "the files are syntactically valid and the compose graph is shaped correctly". It does not mean "the reasoners can actually call each other, the type contracts line up at runtime, the prompts fit in the context window, and the final synthesis produces something coherent." Only the live execution proves those things.

**Common runtime failures that ONLY show up in the live smoke test:**

- `AttributeError: '<object>' has no attribute '<X>'` — typically a cross-boundary data reconstitution bug (a structured payload arrived as a dict, code tried to access it as a model). See `choosing-primitives.md` → "Cross-boundary data does NOT auto-reconstitute".
- `AttributeError: '<framework object>' has no attribute '<X>'` — a framework surface contract was narrower than assumed. The attribute does not proxy. See `choosing-primitives.md` → router surface table.
- `TypeError: argument after ** must be a mapping, not <T>` — a downstream reasoner tried to `**payload` a value that wasn't a dict. Almost always the same cross-boundary issue.
- A reasoner returned an empty or half-filled schema — usually means an upstream `.ai()` call had a `confident=False` result and the fallback path returned a safe-default instance, but the orchestrator didn't notice and passed the default downstream as if it were real data.
- The pipeline timed out — either the model is too slow, the fan-out is too wide for the sync endpoint, or a sub-reasoner is stuck waiting on something. Switch the smoke test to async if you haven't already.

**The general rule (applies to any multi-component system, not just this one):** the only test that proves a distributed system works is the test that exercises the distribution. Running the canonical path against the live system is not optional; it is the single most important validation step, and the one most likely to catch subtle cross-component contract bugs. Never hand off without it.

## Rationalization counters & red flags

These thoughts mean STOP. If you notice any of them, re-read the linked reference and reconsider.

| Thought / symptom | Reality / re-read |
|---|---|
| "Quick demo, I'll skip the architecture" | The skill exists to be stronger than a chain. Weak demo proves nothing |
| "I'll pass JSON between two reasoners" | LLMs reason over prose. Strings between LLMs, JSON only for code |
| "One big `analyze()` reasoner is fewer files" | Decompose. Granularity is the forcing function for parallelism. `choosing-primitives.md` |
| "I'll skip the CLAUDE.md / README" | They're how the next coding agent extends without breaking it. Always generate |
| "I'll ask 5 questions to be safe" | One question. State assumptions. Iterate |
| "Curl is enough, skip discovery API" | Discovery API tells you in 2s which step actually failed. `verification.md` |
| "I need stateful tool-using → `app.harness()`" | NO. `app.harness()` is external coding-agent CLI delegation AND requires the CLI in the container. Use `app.ai(tools=[...])` or a chunked-loop reasoner |
| "I'll add `app.harness(provider='claude-code')` for the deep reasoning step" | The default Python container has no `claude` CLI. The scaffold will crash on first run. Read "Harness availability gate" |
| "I'll add `input_schema=` to the decorator" | That param doesn't exist. Schemas come from type hints |
| ".ai() for a 50-page document" | `app.ai(tools=[...])` or a chunked-loop reasoner. `choosing-primitives.md` |
| "Static `for` loop of LLM calls, no routing" | Add dynamic routing or admit AgentField isn't justified. `architecture-patterns.md` |
| "Skipping `python3 -m py_compile` and `docker compose config`" | Always run. `scaffold-recipe.md` |
| "I'll write `import requests` to call the other reasoner" | Use `app.call(f"{app.node_id}.X", ...)`. `choosing-primitives.md` |
| "I'll use `app.serve()` in main" | Use `app.run()`. Auto-detects CLI vs server |

## Output contract (every build)

The final message to the user MUST contain these sections, in this order, in a clean copy-pasteable format. The whole point is the first-time user can read the message top to bottom and within 60 seconds have the system running and a working curl in another terminal.

### 1. What was scaffolded

Generated file tree with absolute paths.

### 2. Architecture sketch

4–6 bullets: what each reasoner does, who calls whom, where the dynamic routing happens, where the safety guardrails fire.

### 3. Assumptions made

5–10 bullets — the things you inferred without asking.

### 4. 🚀 Run it (3 commands)

```bash
cd <absolute_project_path>
cp .env.example .env       # then paste your OPENROUTER_API_KEY into .env
docker compose up --build
```

Wait until you see `agent registered` in the logs (~30–90 seconds first run).

### 5. 🌐 Open the UI

After the stack is up, open these URLs in your browser:

| URL | What it shows |
|---|---|
| **http://localhost:8080/ui/** | AgentField control plane web UI — live workflow DAG, reasoner discovery, execution history, verifiable credential chains |
| **http://localhost:8080/api/v1/discovery/capabilities** | JSON: every reasoner registered with the control plane (proves your build deployed) |
| **http://localhost:8080/api/v1/health** | Health check |

### 6. ✅ Verify the build (in another terminal)

Use `/api/v1/discovery/capabilities` as the **primary, durable registration check**. The `/api/v1/nodes` endpoint has filter behaviour that varies across control-plane versions and deployments — a node that successfully registered can still return empty under some filter combinations. Discovery is the only introspection path whose semantics are stable across versions.

```bash
# 1. Control plane up?
curl -fsS http://localhost:8080/api/v1/health | jq '.status'

# 2. Agent registered and reasoners discoverable? (PRIMARY CHECK — always use this)
#    Response shape: .capabilities[].reasoners[].id (NOT .reasoners[].name)
curl -fsS http://localhost:8080/api/v1/discovery/capabilities \
  | jq '.capabilities[] | select(.agent_id=="<slug>") | {
      agent_id,
      n_reasoners: (.reasoners | length),
      entry: [.reasoners[] | select(.tags[]? == "entry") | .id],
      all_reasoner_ids: [.reasoners[].id]
    }'
```

If the JSON above returns your `agent_id` with `n_reasoners > 0` and an `entry` array containing at least one id, **registration succeeded**. That is the only thing that matters. If you also want to look at `/api/v1/nodes`, do it as a secondary diagnostic — never as the primary gate — and do not worry if it shows `health: "unknown"` or returns an empty list; those are cosmetic/filter artifacts, not registration failures.

### 7. 🎯 Try it — sample async curl (canonical)

**Use async.** Multi-reasoner compositions routinely exceed the 90-second timeout on the sync endpoint. The canonical smoke test kicks off the async endpoint and polls until the execution succeeds.

```bash
# 1. Kick off — returns an execution_id immediately
EXEC_ID=$(curl -sS -X POST http://localhost:8080/api/v1/execute/async/<slug>.<entry_reasoner> \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "<param1>": "<realistic value>",
      "<param2>": <realistic value>,
      "model": "openrouter/google/gemini-2.5-flash"
    }
  }' | jq -r '.execution_id')
echo "Execution: $EXEC_ID"

# 2. Poll until done and print the result
while :; do
  R=$(curl -sS http://localhost:8080/api/v1/executions/$EXEC_ID)
  S=$(echo "$R" | jq -r '.status')
  case "$S" in
    succeeded) echo "$R" | jq '.result'; break ;;
    failed)    echo "$R" | jq '.'; break ;;
    *)         sleep 2 ;;
  esac
done
```

**The payload must use realistic data the user can run as-is and see a real reasoned answer.** Do not use placeholder values like `"foo"` or `"test"`. Use concrete values that actually exercise every reasoner in the system. The optional `"model"` field overrides the AIConfig default per-request — show it in the example so users discover the per-request override.

If the user provided test data in the brief (sample input, sample record, sample document), use THAT data verbatim in this curl. The first execution they run should be the most demonstrative one.

**When it is safe to use the sync endpoint instead:** only if you can genuinely guarantee the entire pipeline — every layer of fan-out, every child `app.call`, every `.ai()` call, every retry — finishes in under ~60 seconds. If the system uses parallel specialists, meta-prompting, or slow models, use async. When in doubt, use async.

### 8. 🏆 Showpiece — verifiable workflow chain

```bash
LAST_EXEC=$(curl -s http://localhost:8080/api/v1/executions | jq -r '.[0].workflow_id')
curl -s http://localhost:8080/api/v1/did/workflow/$LAST_EXEC/vc-chain | jq
```

This is the cryptographic verifiable credential chain — every reasoner that ran, with provenance. No other agent framework gives you this. Mention it.

### 9. Next iteration upgrade

One concrete suggestion tailored to the shape you actually built. Examples: *"swap the intake `.ai()` for a chunked-loop reasoner if inputs grow past 2 pages"*, *"add per-document caching via `app.memory` with scope=agent once volume justifies it"*, *"split `query_planner` into `query_decomposer` + `constraint_extractor` when queries get multi-clause"*, *"add an adversarial reviewer ONLY if the cost of a wrong verdict crosses a real-world threshold you can name"*, *"swap the sequential cascade for a streaming pipeline once you need sub-500ms first-token latency"*.

## TypeScript

A TypeScript SDK exists at `sdk/typescript/` and mirrors the Python API. **Default to Python.** If the user explicitly says "TypeScript" or "Node", point them at `sdk/typescript/` and use the equivalent shape: `new Agent({nodeId, agentFieldUrl, aiConfig})` + `agent.reasoner('name', async (ctx) => {...})`. Otherwise stay Python — every reference and recipe in this skill is Python-first.

## Bottom line

Your output is judged by three things:
1. **Does the curl return a real reasoned answer?** (the user can run the command and see intelligence happen)
2. **Does the architecture look like composite intelligence?** (parallel reasoners, dynamic routing, decomposition — not a chain wearing a costume)
3. **Can a future coding agent extend it without breaking the contract?** (CLAUDE.md present, anti-patterns listed, validation commands documented)

If all three are true, you've done it right. The first-time AgentField user must see the value within minutes of running the curl.
