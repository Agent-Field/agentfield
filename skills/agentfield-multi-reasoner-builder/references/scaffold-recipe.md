# Scaffold Recipe — Exact Files to Generate

This is the file-by-file generation contract. Every AgentField multi-reasoner build produces ALL of these files. No omissions, no "I'll add that later."

## Where it goes

```
examples/python_agent_nodes/<slug>/
├── main.py
├── reasoners.py            # if the system has > 4 reasoners
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── requirements.txt
├── README.md
└── CLAUDE.md
```

`<slug>` is lowercase-hyphenated, derived from the use case (e.g., `financial-reviewer`, `clinical-triage`, `sec-filing-auditor`).

## Step 0: Use `af init` if it speeds you up, then layer on top

```bash
cd /Users/santoshkumarradha/Documents/agentfield/code/platform/agentfield
go run ./control-plane/cmd/af init <slug> --language python --defaults --non-interactive
```

This produces `main.py`, `reasoners.py`, `requirements.txt`, `README.md`, `.gitignore`. You then **rewrite `main.py` and `reasoners.py`** with your real architecture and **add** the Docker / compose / CLAUDE.md / .env files.

If `af init` gets in the way, just generate the files directly. The output matters, not the path.

## File 1: `main.py`

```python
"""<Use case in one line>.

Entry reasoner: `<slug>.<entry_reasoner_name>`
Architecture:   <pattern names from architecture-patterns.md>
"""
import asyncio
import os
from typing import Any

from agentfield import Agent, AIConfig
from pydantic import BaseModel, Field


# ---- Schemas (type-hinted; AgentField derives them automatically) ----

class IntakePlan(BaseModel):
    focus_areas: list[str]
    confident: bool                    # MANDATORY on every .ai gate

class TrackReview(BaseModel):
    axis: str
    score: int = Field(ge=1, le=10)
    rationale: str

class FinalVerdict(BaseModel):
    overall: str
    strengths: list[str]
    risks: list[str]


# ---- Agent ----

app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "<slug>"),
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("AI_MODEL", "openrouter/google/gemini-2.5-flash"),
    ),
    dev_mode=True,
)


# ---- Internal reasoners ----

@app.reasoner()
async def intake_router(
    payload: dict,
    model: str | None = None,            # propagate model
) -> IntakePlan:
    plan = await app.ai(
        system="You classify the input and pick the smallest set of analysis tracks needed.",
        user=str(payload),
        schema=IntakePlan,
        model=model,
    )
    if not plan.confident or not plan.focus_areas:
        # FALLBACK: escalate (could be a chunked-loop reasoner or a deeper pass)
        plan.focus_areas = ["default_a", "default_b"]
    return plan


@app.reasoner()
async def dimension_reviewer(
    payload: dict,
    axis: str,
    model: str | None = None,
) -> TrackReview:
    return await app.ai(
        system=f"You are a {axis} reviewer. Score and rationalize.",
        user=f"Axis: {axis}\nPayload: {payload}",
        schema=TrackReview,
        model=model,
    )


# ---- Entry reasoner (the public surface) ----

@app.reasoner(tags=["entry"])
async def review(
    payload: dict,
    model: str | None = None,            # per-request model override
) -> dict:
    plan_dict = await app.call(
        f"{app.node_id}.intake_router",
        payload=payload,
        model=model,
    )
    plan = IntakePlan(**plan_dict)

    # Parallel fan-out across selected dimensions
    review_dicts = await asyncio.gather(*[
        app.call(
            f"{app.node_id}.dimension_reviewer",
            payload=payload,
            axis=axis,
            model=model,
        )
        for axis in plan.focus_areas
    ])

    # Synthesize via another LLM reasoner — pass prose, not JSON
    review_prose = "\n".join(
        f"- [{r['axis']}] score={r['score']} — {r['rationale']}"
        for r in review_dicts
    )
    verdict = await app.ai(
        system="You are the lead reviewer. Synthesize the dimension findings into a verdict.",
        user=review_prose,
        schema=FinalVerdict,
        model=model,
    )

    return {
        "plan": plan.model_dump(),
        "reviews": review_dicts,
        "verdict": verdict.model_dump(),
    }


if __name__ == "__main__":
    # app.run() auto-detects CLI vs server mode (verified at sdk/python/agentfield/agent.py:4194)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)
```

**Hard requirements:**
- `node_id`, `agentfield_server`, `model` all read from env with sensible defaults
- `auto_port=False` so the port is deterministic and the curl works
- Exactly ONE entry reasoner with `tags=["entry"]` for discovery
- Schemas are derived from **type hints** — do NOT pass `input_schema=` or `output_schema=` to `@app.reasoner` (those parameters do not exist)
- Every `.ai()` gate has a `confident: bool` field in its schema and a fallback path
- Every reasoner that calls `.ai()` accepts an optional `model: str | None = None` parameter and threads it through `app.ai(model=model)`
- The entry reasoner accepts `model` and propagates it via `app.call(..., model=model)` to all children
- All inter-reasoner calls use `app.call(f"{app.node_id}.X", ...)` — never hardcoded node IDs
- Never `requests.post()` to another reasoner. Use `app.call`
- Use `app.run()` in `__main__`, not `app.serve()`

## File 2: the `reasoners/` package (canonical layout for non-trivial systems)

When the system has more than 4 reasoners, **use this canonical 4-file router package layout**. It separates concerns cleanly and makes the build extensible without breaking the orchestrator:

```
<slug>/
├── main.py                      # Agent + entry reasoner + orchestration
└── reasoners/
    ├── __init__.py              # Re-exports the routers so main.py can include them
    ├── models.py                # Pydantic schemas — every BaseModel used by every reasoner
    ├── helpers.py               # Plain Python utilities: math, prose renderers, fact registry, fallbacks
    ├── specialists.py           # AgentRouter for the parallel "hunter" / specialist reasoners
    └── committee.py             # AgentRouter for the orchestration-layer reasoners (intake router, adversarial reviewer, synthesizer)
```

**`reasoners/__init__.py`:**
```python
from .committee import router as committee_router
from .specialists import router as specialists_router

__all__ = ["committee_router", "specialists_router"]
```

**`reasoners/models.py`** — every Pydantic schema in one place. Includes the input application schema, the per-specialist review schema (with `confident: bool` mandatory), the routing plan schema, the adversarial review schema, the final decision schema, and any deterministic-metric schemas. Keeping these in one file makes type-checking trivial and prevents circular imports between routers.

**`reasoners/helpers.py`** — plain Python (NOT decorated with `@app.skill`) for: deterministic math (DTI, payment amount, employment-gap calc), `render_specialist_review()` and similar **prose renderers** that convert Pydantic instances to natural-language strings before passing them to another LLM, the fact-registry builder for citation IDs, and **fallback constructors** like `fallback_specialist_review(axis, reason)` that produce safe-default Pydantic instances when an `.ai()` call returns `confident=False`.

> **Why plain helpers vs `@app.skill()`?** `@app.skill()` makes a function discoverable and callable through `app.call`. Use it when the deterministic function is something the system might call from a reasoner OR something an external caller might want to invoke directly through the control plane. For purely internal helpers used inside reasoner bodies (math, prose rendering, schema construction), plain Python is cleaner — no decorator overhead, no registration ceremony. Promote a helper to `@app.skill()` only when you actually want to call it via `app.call`.

**`reasoners/specialists.py`** — one `AgentRouter(prefix="", tags=["specialist"])`, one `@router.reasoner` per analysis dimension. Often these specialists share a `_run_specialist_review()` private helper that takes a system prompt + focus prompt as parameters, runs `router.ai(...)`, and applies the `confident=False` fallback. This keeps each specialist body to ~5 lines of configuration.

**`reasoners/committee.py`** — one `AgentRouter(prefix="", tags=["committee"])` with the orchestration-layer reasoners: `intake_router` (decides which specialists to run), `adversarial_challenger` (the HUNT→PROVE counterpart), `committee_reconciler` (synthesizes specialists + adversarials → final decision).

**`main.py`** does three things:
1. Construct `Agent(...)` with `node_id`, `agentfield_server`, `ai_config`
2. `app.include_router(committee_router)` and `app.include_router(specialists_router)`
3. Define the public **entry reasoner** with `tags=["entry"]` that orchestrates the full pipeline using `app.call(f"{app.node_id}.X", ...)` and `asyncio.gather` for parallel fan-out, plus deterministic governance overrides at the end

**This is the layout that emerges naturally** when you decompose a real composite-intelligence system. If your build has fewer than 4 reasoners, keep everything in `main.py` and skip the package. If it has more, use this layout. Do not invent a different layout.

### Smaller systems (≤4 reasoners): keep everything in `main.py`

For trivial builds, skip the package and inline everything. Use `@app.reasoner()` directly on `app`. Don't create a router with one reasoner in it.

### Referencing the node id from inside a router file

Router files need to make `app.call(f"{node_id}.target", ...)` calls, but **`router.node_id` does not exist** — the router only proxies a fixed set of callables (`ai`, `call`, `memory`, `harness`), not arbitrary data attributes. The canonical pattern inside every router file:

```python
import os
from agentfield import AgentRouter

NODE_ID = os.getenv("AGENT_NODE_ID", "<slug>")   # same default as main.py
router = AgentRouter(prefix="", tags=["..."])

@router.reasoner()
async def some_reasoner(payload: dict, model: str | None = None) -> dict:
    return await router.call(f"{NODE_ID}.child_reasoner", payload=payload, model=model)
```

Never write `f"{router.node_id}.child"`. It will raise `AttributeError` at runtime. See `choosing-primitives.md` → router surface table.

### Reasoners compose like normal function calls — this is the superpower

Every reasoner is callable from every other reasoner, at any depth, in any shape. `app.call(f"{NODE_ID}.target", ...)` works **exactly like calling a regular Python function or hitting a REST endpoint** — you can invoke it from anywhere in your code: inside another reasoner's body, inside a loop, inside a branch, inside a recursive descent, inside a conditional that depends on a prior call's output, inside a meta-reasoner that decides at runtime which sub-reasoner to call and with what arguments.

This is the thing you cannot do with a hand-authored static DAG (LangGraph, hand-written asyncio pipelines, CrewAI's fixed topology). A static DAG forces the entire call graph to be declared upfront. AgentField lets the call graph **emerge at runtime** from the reasoners' own intermediate decisions — the orchestrator body is just Python, and `app.call` is just a function, so everything Python can do (branching, recursion, loops, dynamic dispatch, meta-programming, runtime prompt construction) is available to your architecture.

Use this power. Build graphs with real depth:

- A reasoner deep inside a branch can call another reasoner at the top of the tree.
- A reasoner can call itself recursively (with a depth cap) to drill into nested structure.
- A meta-reasoner can synthesize a brand new prompt and then invoke a child reasoner with that prompt as a kwarg — the child's behavior is determined at runtime by a sibling's output.
- A reasoner can fan out a `asyncio.gather` over N sub-reasoners where N itself was decided by an earlier reasoner.
- A reasoner can call a sub-reasoner, read its result, and conditionally decide whether to call a completely different sub-reasoner next — the shape of the next layer is not committed until the current layer finishes.

The only rule is that every `app.call` goes through the control plane (never direct HTTP), so the workflow DAG, the cryptographic provenance chain, and the observability layer see every edge. The resulting call graph is often impossible to draw upfront — that is the point. A static DAG would require you to enumerate every possible path; a composable reasoner system only walks the paths that actually apply to the current input.

**What this means for your design:** do not constrain yourself to the shapes you can draw on a whiteboard. Decompose, make each reasoner a narrowly-scoped callable, then let orchestrator bodies invoke each other freely — deeply, conditionally, recursively, dynamically. The more the call graph depends on intermediate state, the more AgentField earns its place over simpler frameworks.

**Important gotcha — cross-boundary serialization.** Because every `app.call` crosses a serialization boundary, structured payloads lose their runtime type identity in transit. Pydantic models become plain dicts on the receiving side regardless of type hints. Either reconstruct explicitly on the receiver (`Model(**payload)`), or render to a string on the caller before the call. Both are fine; pick whichever fits the boundary. See `choosing-primitives.md` → "Cross-boundary data does NOT auto-reconstitute" for the full treatment.

## File 3: `Dockerfile`

**Use `af init --docker` to generate this. The command produces the universal shape below — do not customize.**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

EXPOSE 8001

CMD ["python", "main.py"]
```

**Key properties of this Dockerfile (verified against `af init --docker`):**
- **Universal — no repo coupling.** The build context is the project directory itself (`docker-compose.yml` uses `context: .`), so the same scaffold works whether the project lives inside the agentfield repo at `examples/python_agent_nodes/<slug>/` or completely standalone at `/tmp/my-build/`.
- The SDK is installed via `pip install -r requirements.txt`, where `requirements.txt` lists `agentfield`. **Do not** add `COPY sdk/python /tmp/python-sdk` — that's the old repo-coupled pattern, and it breaks for out-of-repo builds.
- `requirements.txt` must contain at least `agentfield` (one line). Add `pydantic>=2,<3` and any libraries the reasoners actually need.

## File 4: `docker-compose.yml`

**Use `af init --docker` to generate this. The command produces the universal shape below — do not customize unless you have a specific reason.**

```yaml
services:
  control-plane:
    image: agentfield/control-plane:latest
    environment:
      AGENTFIELD_STORAGE_MODE: local
      AGENTFIELD_HTTP_ADDR: 0.0.0.0:8080
    ports:
      - "${AGENTFIELD_HTTP_PORT:-8080}:8080"
    volumes:
      - agentfield-data:/data
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:8080/api/v1/health"]
      interval: 3s
      timeout: 2s
      retries: 20

  <slug>:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      AGENTFIELD_SERVER: http://control-plane:8080
      AGENT_CALLBACK_URL: http://<slug>:8001
      AGENT_NODE_ID: ${AGENT_NODE_ID:-<slug>}
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY:-}
      AI_MODEL: ${AI_MODEL:-openrouter/google/gemini-2.5-flash}
      PORT: ${PORT:-8001}
    ports:
      - "${AGENT_NODE_PORT:-8001}:8001"
    depends_on:
      control-plane:
        condition: service_healthy
    restart: on-failure

volumes:
  agentfield-data:
```

**Build context is `.` (the project directory itself), not `../../..`.** This makes the scaffold portable to any location on disk. All four provider env vars are exposed with `:-` defaults so missing keys don't crash compose validation.

**Hard requirements:**
- Control plane has a healthcheck so the agent only starts after the control plane is ready
- Agent uses `depends_on: condition: service_healthy` (not just `depends_on: [control-plane]`)
- All three common provider env vars (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are exposed so the user can swap providers without editing compose
- Default model is OpenRouter Claude 3.5 Sonnet (most reliable for reasoning) but trivially overridable via `AI_MODEL`
- Port 8080 = control plane, port 8001 = agent node, never co-located

## File 5: `.env.example`

```bash
# Required: pick ONE provider
OPENROUTER_API_KEY=sk-or-v1-...
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Model — must match the provider above
AI_MODEL=openrouter/google/gemini-2.5-flash
# AI_MODEL=gpt-4o
# AI_MODEL=anthropic/claude-3-5-sonnet-20241022

# Optional overrides
AGENT_NODE_ID=<slug>
AGENT_NODE_PORT=8001
AGENTFIELD_HTTP_PORT=8080
```

## File 6: `requirements.txt`

```
# The agentfield SDK is installed from the local repo by the Dockerfile.
# Keep this file to additional runtime deps the reasoners need.
pydantic>=2.0
```

Add libraries the reasoners actually use (httpx, beautifulsoup4, pdfplumber, etc.). Don't list `agentfield` here — it comes from the local SDK copy.

## File 7: `.dockerignore`

```
__pycache__
*.pyc
.pytest_cache
.env
.venv
*.log
```

## File 8: `README.md`

```markdown
# <Use Case Title>

<2-sentence description.>

## Architecture

- **Entry reasoner:** `<slug>.<entry_reasoner_name>`
- **Pattern(s):** <e.g., Parallel Hunters + HUNT→PROVE>
- **Reasoners:**
  - `intake_router` — `.ai()` gate that classifies inputs and selects active dimensions
  - `<dimension_a>_reviewer` — analyzer for dimension A (parallel)
  - `<dimension_b>_reviewer` — analyzer for dimension B (parallel)
  - `synthesizer` — combines dimension findings into a final verdict

## Run

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY (or your provider of choice)
docker compose up --build
```

Wait until you see `agent registered` in the logs.

## Verify (run in another terminal)

```bash
# 1. Control plane is up
curl -fsS http://localhost:8080/api/v1/health | jq

# 2. Agent node has registered
curl -fsS http://localhost:8080/api/v1/nodes | jq '.[] | {id: .node_id, status: .status}'

# 3. All reasoners are discoverable (look for tags=["entry"])
curl -fsS http://localhost:8080/api/v1/discovery/capabilities \
  | jq '.reasoners[] | select(.node_id=="<slug>") | {name, tags}'
```

## Run a real reasoned answer

**Important:** the control plane wraps reasoner kwargs in an `input` field. Body shape is `{"input": {...kwargs...}}` — verified against `control-plane/internal/handlers/execute.go`.

```bash
curl -X POST http://localhost:8080/api/v1/execute/<slug>.<entry_reasoner_name> \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "<param1>": "<value>",
      "<param2>": <value>,
      "model": "openrouter/google/gemini-2.5-flash"
    }
  }' | jq
```

The optional `"model"` field overrides the AIConfig default for THIS request. Try different models:

```bash
# Same request, different model
curl -X POST http://localhost:8080/api/v1/execute/<slug>.<entry_reasoner_name> \
  -H 'Content-Type: application/json' \
  -d '{"input": {"<param1>": "...", "model": "openrouter/openai/gpt-4o"}}' | jq
```

## Showpiece — see the cryptographic workflow trail

```bash
LAST_EXEC=$(curl -s http://localhost:8080/api/v1/executions | jq -r '.[0].workflow_id')
curl -s http://localhost:8080/api/v1/did/workflow/$LAST_EXEC/vc-chain | jq
```

This is the verifiable credential chain — every reasoner that ran, with cryptographic provenance. No other agent framework gives you this.

## Stop

```bash
docker compose down
docker compose down --volumes  # also clears local control-plane state
```
```

## File 9: `CLAUDE.md`

See `references/project-claude-template.md` for the template. Generate it specific to this build.

## Generation order (do these in this order)

1. Decide the architecture (pattern + reasoner roles + which are `.ai()` vs `.harness()`)
2. Create the directory `examples/python_agent_nodes/<slug>/`
3. Write `main.py` with real reasoners (NOT a placeholder)
4. Write `requirements.txt`, `Dockerfile`, `.dockerignore`
5. Write `docker-compose.yml`
6. Write `.env.example`
7. Write `CLAUDE.md` (use the template from `references/project-claude-template.md`)
8. Write `README.md` with the actual curl payload for THIS use case
9. Validate (see next section)

## Validation (every build)

### Online validation (when Docker can pull images and you have a key)

```bash
# 1. Python syntax — must pass
python3 -m py_compile examples/python_agent_nodes/<slug>/main.py
# Plus any reasoner files if you split with routers:
python3 -m py_compile examples/python_agent_nodes/<slug>/reasoners/*.py

# 2. Compose file is valid
cd examples/python_agent_nodes/<slug>
OPENROUTER_API_KEY=sk-or-v1-FAKE docker compose config > /dev/null

# 3. Start the stack and run the smoke test
docker compose up --build -d
sleep 10 && curl -fs http://localhost:8080/api/v1/health
curl -X POST http://localhost:8080/api/v1/execute/<slug>.<entry_reasoner> \
  -H 'Content-Type: application/json' \
  -d '{"input": {"...": "..."}}'
docker compose logs <slug> --tail=50
docker compose down
```

### Offline validation (sandbox / CI / no docker pull)

If the environment cannot pull `agentfield/control-plane:latest` or doesn't have a real provider key, you **still validate**. These are the static checks that count as "validated":

```bash
# Syntax check
python3 -m py_compile examples/python_agent_nodes/<slug>/main.py
python3 -m py_compile examples/python_agent_nodes/<slug>/reasoners/*.py 2>/dev/null || true

# Compose syntax check (no image pull required)
cd examples/python_agent_nodes/<slug>
OPENROUTER_API_KEY=sk-or-v1-FAKE docker compose config > /dev/null
```

Then **run this visual-invariant checklist** against the generated files. Every box must be checked:

- [ ] `app.run(...)` in `__main__` (NOT `app.serve(...)`)
- [ ] Entry reasoner has `tags=["entry"]`
- [ ] Every `app.ai(...)` call's schema includes a `confident: bool` field if used as a gate, AND the call site has a fallback path
- [ ] Every reasoner that calls `app.ai(...)` accepts `model: str | None = None` and threads `model=model`
- [ ] Entry reasoner accepts `model` and propagates via `app.call(..., model=model)` to every child
- [ ] All `app.call(...)` use `f"{app.node_id}.X"` — no hardcoded node IDs
- [ ] No `requests.post()` / `httpx.post()` between reasoners (use `app.call`)
- [ ] No `app.harness(provider="...")` unless the Dockerfile installs the CLI AND main.py has a startup `shutil.which()` check
- [ ] No `input_schema=` / `output_schema=` parameters on `@app.reasoner()`
- [ ] README curl uses body shape `{"input": {...kwargs...}}` (NOT raw kwargs at top level)
- [ ] README canonical smoke test uses `POST /api/v1/execute/async/...` + polling `GET /api/v1/executions/:id` (NOT sync — multi-reasoner pipelines exceed the 90s sync timeout)
- [ ] README registration check uses `GET /api/v1/discovery/capabilities` as the primary gate (NOT `/api/v1/nodes` with filter params — its behaviour varies across control-plane builds)
- [ ] `Agent(agentfield_server=os.getenv("AGENTFIELD_SERVER", ...))` — exact parameter name
- [ ] `AGENT_CALLBACK_URL` set in compose to the in-network DNS name (`http://<service>:8001`)
- [ ] `auto_port=False` in `app.run()` so the port is deterministic
- [ ] CLAUDE.md exists with no `<placeholder>` tokens left in it
- [ ] `.env.example` lists `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- [ ] If reasoners are split across files: routers read `NODE_ID = os.getenv("AGENT_NODE_ID", "<slug>")` and use `f"{NODE_ID}.target"` — NOT `f"{router.node_id}.target"` (that attribute does not exist)
- [ ] No Pydantic instances or lists of Pydantic instances passed across `app.call(...)` boundaries. Either (a) reconstruct on the receiving side with `Model(**payload)`, or (b) render to prose in the caller and pass a string. See the "Fan-in handoff" section above
- [ ] LLM-to-LLM context is passed as natural-language strings, not raw JSON dicts
- [ ] Returning `dict` from an orchestrator reasoner is fine — Pydantic model returns are also fine — both work because schemas come from type hints

If any box fails, **fix before handing off**. A "scaffold that almost works" is worth zero.

### Mandatory live smoke test (before telling the user the build is ready)

**Static validation is necessary but not sufficient.** `py_compile` and `docker compose config` check syntax and shape — they do NOT exercise the live call graph, do NOT catch cross-boundary type mismatches, do NOT reveal runtime contract drift between reasoners. A build is not done until the canonical async curl has been fired against the live stack and returned `status: "succeeded"` with a real reasoned `result`.

Required sequence before handoff:

```bash
# 1. Bring the stack up
docker compose up --build -d

# 2. Wait for registration, using the durable primary check
for i in 1 2 3 4 5 6 7 8 9 10; do
  READY=$(curl -fsS http://localhost:8080/api/v1/discovery/capabilities 2>/dev/null \
    | jq -r '.capabilities[] | select(.agent_id=="<slug>") | .agent_id')
  [ -n "$READY" ] && break
  sleep 2
done
[ -z "$READY" ] && { echo "Agent never registered"; docker compose logs <slug> --tail=50; exit 1; }

# 3. Fire the canonical async curl from the README with realistic input
EXEC_ID=$(curl -sS -X POST http://localhost:8080/api/v1/execute/async/<slug>.<entry_reasoner> \
  -H 'Content-Type: application/json' \
  -d @./sample_payload.json | jq -r '.execution_id')

# 4. Poll until done
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
    *)
      sleep 2
      ;;
  esac
done

# 5. Tear down
docker compose down
```

**If the live smoke test fails, DO NOT hand off.** Read the error field and the agent container logs, find the actual stack trace, fix the bug, start over from step 1. The two most common runtime failures that only surface here:

- `AttributeError: '<object>' has no attribute '<X>'` — typically a cross-boundary data reconstitution bug (you passed a Pydantic model across `app.call` and the receiver tried to use it as one). Apply the fan-in handoff pattern above.
- `AttributeError: 'AgentRouter' has no attribute '<X>'` — you tried to use a router attribute that isn't in the proxied set. Check the router surface table in `choosing-primitives.md` and switch to env-based access or direct agent access.

**The general rule:** the only test that proves a distributed system works is the test that exercises the distribution. Run the canonical path against the live system before telling the user it's ready. Never hand off on static checks alone.

### Return-type note

Orchestrator reasoners that return heterogeneous results (e.g. `{"plan": ..., "reviews": [...], "verdict": ...}`) should declare `-> dict` as the return type. Single-purpose reasoners that produce one validated result should declare `-> SomePydanticModel`. Both work — schemas are derived from the type hint either way.
