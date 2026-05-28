# Choosing Primitives — Philosophy + Real SDK Surface

The most consequential architectural decision in any AgentField build. This file is one read because the philosophy IS the primitive choice — you cannot decide between `.ai()` and a `@reasoner` loop without first knowing what kind of reasoning you're trying to amplify. Read top to bottom before writing code.

---

## Part 1 — Composite Intelligence (the "why")

A single LLM call reasons at ~0.3–0.4 on a normalized scale where 1.0 is human-expert. **You cannot prompt your way to 0.8.** You can architect your way there.

A well-composed system of ten 0.3-grade reasoners can outperform a single 0.4-grade monolith by 5–10× on complex tasks — because the architecture itself encodes intelligence about how to break down problems, allocate cognitive work, combine partial solutions, and stay coherent across steps.

You are not a prompt engineer. You are a **systems architect**. Your job is to design the cognitive architecture; the LLMs are interchangeable parts.

### What this is NOT

- ❌ A single super-intelligent generalist that solves anything in one call
- ❌ A linear chain of LLM calls dressed up with "agent" branding (LangChain, CrewAI, AutoGen patterns)
- ❌ A pile of unbounded autonomous agents "thinking" their way to an answer
- ❌ A tool to orchestrate tools (that's what a script is for)

### What it IS

- ✅ A network of **specialized cognitive functions**, each tightly scoped
- ✅ **Architecture patterns** that elevate collective reasoning above any individual call
- ✅ **Decomposed atomic reasoning units** that can run in parallel
- ✅ **Guided autonomy**: agents have freedom inside a tight scope, not unbounded freedom
- ✅ **Dynamic routing**: the path adapts to what gets discovered, not a hardcoded DAG
- ✅ **Verifiable provenance**: every claim traces to its source

### The five foundational principles

**1. Granular decomposition is mandatory.** No complex problem is solved by a single agent in a single step. The constraint is a forcing function that produces parallelism, observability, and quality. If your "AI agent" is one 200-line function, you decomposed wrong.

**2. Guided autonomy, never unbounded.** A reasoner has freedom in HOW it accomplishes its goal, but **zero freedom** in WHAT the goal is. The orchestrator is a CEO, not a babysitter — it sets objectives and verifies outcomes.

**3. Dynamic, state-responsive orchestration.** The flow of control is not static. Agent A's output determines what subsystem B even looks like. This is the **meta-level** intelligence that distinguishes AgentField from chain frameworks: the chain shape itself is intelligence.

**4. Contextual fidelity & verifiable provenance.** The orchestrator is a context broker. Every reasoner gets exactly what it needs — no more, no less. Every claim carries a citation key that propagates to the final output.

**5. Asynchronous parallelism.** Decompose to parallelize. If your reasoner pipeline runs sequentially, your decomposition is wrong. Use `asyncio.gather` aggressively.

### The intelligence test

The whole point is **intelligence**. If something can be done programmatically — sorting, scoring, deduping, filtering, regex extraction, schema validation — **do it in code** (`@app.skill()`). LLMs are reserved for things that previously required a human expert: judgment, discovery, synthesis, routing decisions on ambiguous data, recognizing patterns that don't have clean rules.

If your "AI agent" is doing work a Python `for` loop could do, you're burning money and intelligence on the wrong layer.

### Why AgentField, not LangChain or CrewAI

LangChain and CrewAI give you **tools to build chains**. AgentField gives you a **control plane** that:

- Routes every inter-reasoner call through a server you can introspect, replay, and audit
- Tracks the live workflow DAG so you can see the system's reasoning shape
- Generates W3C verifiable credentials for every execution (cryptographic audit trail)
- Lets reasoners spawn sub-reasoners with dynamic prompts at runtime (meta-prompting)
- Enforces a clean separation between agent nodes (deployable units) and reasoners (cognitive units)
- Gives you per-call model overrides so a parent reasoner can route different sub-tasks to different LLMs

You are not building "an agent." You are deploying a **reasoning system** as production infrastructure.

---

## Part 2 — The Real Python SDK Surface (the "how")

Signatures here come from reading `sdk/python/agentfield/agent.py`, `router.py`, and `tool_calling.py` directly. Many docs describe an idealized API — this section is what actually works.

## The five primitives

| Primitive | What it really does | When to use |
|---|---|---|
| `@app.reasoner()` | Registers a function as a reasoner with the control plane. The function body is yours — make as many `app.ai()` / `app.call()` calls as you want | Wrap **every cognitive unit** in your system |
| `@app.skill()` | Registers a deterministic function. No LLM | Pure transforms, scoring, parsing, dedup, validation — anything code can do |
| `app.ai(...)` | Single call OR multi-turn tool-using LLM call (when `tools=` is passed). Returns text or a Pydantic schema | Classification, routing, structured analysis, **and** stateful tool-using reasoning when you give it tools |
| `app.call(target, **kwargs)` | Calls another reasoner/skill THROUGH the control plane. Tracks the workflow DAG | All inter-reasoner traffic. Never use direct HTTP |
| `app.harness(prompt, provider=...)` | **Delegates to an external coding-agent CLI** (claude-code, codex, gemini, opencode). Returns a `HarnessResult` | When you need a real coding agent to read/write files, run shell commands, or execute a non-trivial coding task as part of your pipeline |

## What `app.ai()` actually accepts

```python
result = await app.ai(
    *args,                     # positional: text, urls, file paths, bytes, dicts, lists (multimodal)
    system: str | None,        # system prompt
    user: str | None,          # user prompt (alternative to positional)
    schema: type[BaseModel] | None,  # Pydantic class for structured output
    model: str | None,         # PER-CALL model override (e.g. "gpt-4o", "openrouter/google/gemini-2.5-flash")
    temperature: float | None,
    max_tokens: int | None,
    stream: bool | None,
    response_format: "auto" | "json" | "text" | dict | None,
    tools: list | str | None,  # tool definitions for tool-calling, OR "discover" to auto-discover
    context: dict | None,
    memory_scope: list[str] | None,  # ["workflow", "session", "reasoner"] etc.
    **kwargs,                  # provider-specific extras
)
```

**Critical things most coding agents miss:**
- `model=` is per-call. You can override the AIConfig default on any specific call. **Always** thread `model` through from the entry reasoner so the user can A/B test models per request.
- `tools=` makes `app.ai()` a multi-turn tool-using LLM. This is how you build "stateful reasoning agents" — not via `app.harness()`. Pass `tools="discover"` to auto-discover available tools, or pass a list of tool definitions.
- `memory_scope=["workflow", "session", "reasoner"]` injects relevant memory state into the prompt automatically.
- `schema=` returns a validated Pydantic instance, not a dict. Call `.model_dump()` to serialize.

## What `app.harness()` actually accepts

```python
result = await app.harness(
    prompt: str,                         # task description
    schema: type[BaseModel] | None,      # optional structured output
    provider: "claude-code" | "codex" | "gemini" | "opencode" | None,
    model: str | None,                   # override the provider's default model
    max_turns: int | None,               # iteration cap
    max_budget_usd: float | None,        # cost cap
    tools: list[str] | None,             # which tools the coding agent is allowed to use
    permission_mode: "plan" | "auto" | None,
    system_prompt: str | None,
    env: dict[str, str] | None,
    cwd: str | None,
    **kwargs,
)
# Returns HarnessResult with .text, .parsed (validated schema), .result
```

**Use harness when:** you need a real coding agent (Claude Code, Codex, Gemini CLI) to perform a task that requires actual file I/O, shell access, or multi-step coding. Example: a "fix-this-failing-test" reasoner spawns a Claude Code harness to actually edit the test file.

**Do NOT use harness for:** in-process stateful LLM reasoning over a document. That's `app.ai(..., tools=[...])`. Harness is heavyweight — it spawns a subprocess running an entire agent CLI.

## What `app.call()` actually does

```python
result: dict = await app.call(
    target: str,           # "node_id.reasoner_name"
    *args,                 # positional args (auto-mapped to target's params for local calls)
    **kwargs,              # keyword args passed to the target reasoner
)
```

**Always returns a `dict`** — even if the target reasoner returns a Pydantic model. Convert manually:

```python
result_dict = await app.call(f"{app.node_id}.score", text=passage)
result = ScoreResult(**result_dict)
```

**Critical:** always reference reasoners as `f"{app.node_id}.reasoner_name"` so renaming the node via `AGENT_NODE_ID` env doesn't break the system. Hardcoding the node ID is a bug waiting to happen.

**Workflow tracking:** every `app.call` is recorded in the control plane's workflow DAG. Direct HTTP between reasoners bypasses this and is forbidden.

### Cross-boundary data does NOT auto-reconstitute — the most dangerous silent contract in the whole SDK

`app.call` crosses a serialization boundary. Even when the calling side and the receiving side both live in the same Python process, the payload is serialized to JSON and deserialized on the way back in. **This means every structured object crossing the boundary loses its runtime type identity.**

Concretely:

- If you build a Pydantic model on the caller side and pass it as a keyword argument to `app.call(...)`, the receiver gets a **plain dict** (or a list of plain dicts, or a dict of plain dicts), regardless of how the receiver's parameter is type-hinted.
- Type hints on the receiver document and validate the *shape* of the payload. They do **not** reconstruct the Pydantic instance for you. A parameter declared as `param: MyModel` will arrive as `dict`, not as `MyModel`. A parameter declared as `param: list[MyModel]` will arrive as `list[dict]`, not as `list[MyModel]`.
- Any downstream code that does `param.field_name`, `param.some_method()`, or `for item in param: item.other_field` will fail at runtime with `'dict' object has no attribute '...'`. Static checks (`py_compile`, type-checkers running on unannotated inputs, even most linters) will NOT catch it. Only running the live call path will.

**Two ways to deal with this. Pick one. Never skip both.**

**(a) Re-construct the model on the receiving side.** If the receiver genuinely needs the Pydantic instance — for example, to call a method or to pass it to a helper that type-checks — reconstruct it explicitly at the top of the function body:

```python
@router.reasoner()
async def downstream_reasoner(payload: dict, model: str | None = None) -> FinalResult:
    typed = UpstreamResult(**payload)             # explicit reconstruction
    # ... use `typed` normally from here ...
```

For list payloads: `typed_items = [UpstreamResult(**item) for item in payload]`.

**(b) Preferred for LLM-to-LLM handoffs — render to prose BEFORE the call.** When the downstream reasoner is going to feed the payload into an `app.ai()` prompt anyway, the data-flow rule already says the boundary should be a natural-language string. This ALSO dodges the serialization trap entirely. Build a prose renderer in `helpers.py`, call it on the caller side, pass a string:

```python
# Caller side — render structured results into prose before crossing the boundary
drafts = [UpstreamResult(**d) for d in await asyncio.gather(*upstream_calls)]
drafts_prose = render_bundle(drafts)   # plain Python helper, returns a str

# Receiver gets a string, not a list of dicts. No reconstruction needed,
# and the LLM inside the receiver reads better prose than serialized JSON.
verdict = await app.call(
    f"{app.node_id}.synthesizer",
    drafts_prose=drafts_prose,
    model=model,
)
```

**The general rule (applies to any framework that crosses a serialization boundary, not just this one):** a type hint is a *shape contract* on the wire, not a *type contract* in memory. Anything structured that crosses the wire arrives as the serialization format's native primitive (`dict`, `list`, `str`, `int`, `bool`, `None`). Runtime type information is lost. Plan for it.

**Red flags that mean you hit this trap:**

- `AttributeError: 'dict' object has no attribute 'X'`
- `TypeError: argument of type 'dict' is not iterable` (when the receiver expected `list[dict]`)
- `TypeError: argument after ** must be a mapping, not NoneType`
- Pydantic ValidationError complaining about missing required fields inside a list payload

Every one of these is the same bug: a structured payload crossed a reasoner boundary and the receiver tried to use it as if it still had its original Python type.

## The decision tree (real, not aspirational)

```
What is this reasoner doing?

├─ Pure deterministic transform (sort, parse, dedup, score-with-formula)?
│  → @app.skill()  (no LLM, free, replayable)
│
├─ Single classification with ≤4 flat fields, input fits comfortably in ~2k tokens?
│  → app.ai(system, user, schema=FlatModel)  (with confident: bool, with fallback)
│
├─ Stateful reasoning where the LLM needs to call tools, search, iterate?
│  → app.ai(system, user, tools=[...])  (multi-turn tool-using mode)
│
├─ Long input (a document, a transcript, a corpus) that needs navigation?
│  → @app.reasoner() that does LOOPED app.ai() calls with chunking,
│    OR app.ai(..., tools=["read_section", ...]) if you've defined the tools,
│    OR pre-process with a @app.skill() chunker then fan-out via asyncio.gather
│
├─ Need an actual coding agent to write/edit files / run shell?
│  → app.harness(prompt, provider="claude-code", tools=[...])
│
└─ Composing multiple reasoners?
   → @app.reasoner() that uses app.call() and asyncio.gather
```

**The bias:** decompose into many small `@app.reasoner()` units. Use `app.ai()` with explicit prompts. Use `tools=` when you need tool-calling. Reserve `app.harness()` for when you literally need a coding agent in the loop.

## The model-propagation pattern (mandatory in every build)

The user must be able to swap models per request without rebuilding the container. Implement it like this in **every** generated entry reasoner:

```python
@app.reasoner(tags=["entry"])
async def review_financials(
    company_name: str,
    business_summary: str,
    financial_snapshot: dict,
    analyst_question: str = "Should we proceed?",
    model: str | None = None,         # ← per-request model override
) -> dict:
    # 1. Use it in app.ai
    plan = await app.ai(
        system="You are a financial intake router.",
        user=f"...",
        schema=IntakePlan,
        model=model,                  # ← propagate
    )

    # 2. Pass it to child reasoners via app.call
    reviews = await asyncio.gather(*[
        app.call(
            f"{app.node_id}.{axis}_reviewer",
            company_name=company_name,
            business_summary=business_summary,
            model=model,              # ← propagate
        )
        for axis in plan.focus_areas
    ])

    # 3. Each child reasoner accepts and uses model the same way
```

And in every child reasoner:
```python
@app.reasoner()
async def profitability_reviewer(
    company_name: str,
    business_summary: str,
    model: str | None = None,         # ← accept it
) -> dict:
    review = await app.ai(
        system="You are a profitability reviewer.",
        user=f"...",
        schema=TrackReview,
        model=model,                  # ← use it
    )
    return review.model_dump()
```

The user can now pick the model per request:
```bash
curl -X POST http://localhost:8080/api/v1/execute/financial-reviewer.review_financials \
  -H 'Content-Type: application/json' \
  -d '{
    "company_name": "Acme",
    "business_summary": "...",
    "financial_snapshot": {...},
    "model": "openrouter/openai/gpt-4o"
  }'
```

If `model` is omitted, the AIConfig default from the env var `AI_MODEL` is used. **This pattern is non-negotiable.** Every generated build must support per-request model override.

## The router pattern (organize reasoners across files)

When a build has more than ~4 reasoners, split them into router files.

**Important detail from the SDK:** `AgentRouter(prefix="...")` **auto-namespaces** the reasoner IDs. A router with `prefix="clauses"` containing a reasoner `analyze_ip` registers as `clauses_analyze_ip`. Call it as `app.call(f"{app.node_id}.clauses_analyze_ip", ...)`.

**Three prefix variations and what they do:**

| Constructor call | Reasoner `analyze_ip` registers as | Use when |
|---|---|---|
| `AgentRouter(prefix="clauses")` | `clauses_analyze_ip` | You want grouped namespacing |
| `AgentRouter(prefix="")` (or omit `prefix`) | `analyze_ip` | You want raw function names — **the canonical default** |
| `@router.reasoner(name="explicit")` overrides any prefix | `explicit` | You want full control over the registered ID |

**Canonical default:** use `AgentRouter(prefix="", tags=["domain"])` so reasoner IDs match function names and your `app.call(f"{app.node_id}.func_name", ...)` calls stay readable. Only use `prefix=` when you have ID collisions across routers.

`reasoners/finance.py`:
```python
from agentfield import AgentRouter
from pydantic import BaseModel

# prefix="" → no auto-namespace; tags merge with per-decorator tags
router = AgentRouter(prefix="", tags=["finance"])

class TrackReview(BaseModel):
    axis: str
    score: int
    rationale: str

@router.reasoner()
async def profitability_reviewer(
    company_name: str,
    business_summary: str,
    model: str | None = None,
) -> TrackReview:                              # type-hinted return drives schema
    return await router.ai(                    # router.ai proxies to the attached agent
        system="You are a profitability reviewer.",
        user=f"Company: {company_name}\n{business_summary}",
        schema=TrackReview,
        model=model,
    )
```

`main.py`:
```python
import os
from agentfield import Agent, AIConfig
from reasoners.finance import router as finance_router
from reasoners.risk import router as risk_router

app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "financial-reviewer"),
    ai_config=AIConfig(model=os.getenv("AI_MODEL", "openrouter/google/gemini-2.5-flash")),
    dev_mode=True,
)

app.include_router(finance_router)
app.include_router(risk_router)

# Entry reasoner stays in main.py
@app.reasoner(tags=["entry"])
async def review_financials(...): ...

if __name__ == "__main__":
    app.run()                                   # auto-detects CLI vs server
```

**Router facts (verified against `router.py`):**

The router proxies a **fixed, enumerated set of callable attributes** from the attached agent. It is NOT a universal transparent proxy. Treat the router's surface as a closed list, not an open one.

| Attribute | Proxied? | How to use |
|---|---|---|
| `router.ai(...)` | ✅ | Same contract as `app.ai(...)` |
| `router.call(...)` | ✅ | Same contract as `app.call(...)` |
| `router.memory` | ✅ | Same contract as `app.memory` |
| `router.harness(...)` | ✅ | Same contract as `app.harness(...)` |
| `router.node_id` | ❌ | Does not proxy. Read the node id from env inside the router file: `NODE_ID = os.getenv("AGENT_NODE_ID", "<slug>")`, then call with `f"{NODE_ID}.target"` |
| Other data attributes on `app` | ❌ | Never assume they proxy. If you need them, read from env or pass them in explicitly |

**The general rule (applies to every framework, not just this one):** when a reference describes a surface contract with words like "every" or "all" or "transparently forwards", mentally replace that with "a specific documented subset". Verify the exact attribute you plan to use before you write code against it. Overstated contracts are the single most common source of subtle runtime breakage in framework-heavy code.

- Tags merge: `AgentRouter(tags=["finance"])` + `@router.reasoner(tags=["scoring"])` → reasoner has BOTH tags.
- `prefix` auto-namespaces IDs as `{prefix_segments}_{func_name}`.
- The canonical pattern is one router per domain file; one `Agent(...)` + multiple `include_router(...)` calls in `main.py`.

**When to use a router vs. keep everything in main.py:**
- ≤ 4 reasoners → main.py only
- 5–10 reasoners → split by domain into 2–3 router files
- > 10 reasoners → consider whether you've decomposed correctly OR whether you need multiple nodes

## Tags

Tags are **free-form** metadata attached to reasoners (verified against the control plane source — there are no reserved tag names). They surface in the discovery API:

```bash
curl -s http://localhost:8080/api/v1/discovery/capabilities \
  | jq '.reasoners[] | select(.tags[]? == "entry")'
```

**Conventions used by AgentField examples (not enforced, just convention):**
- `"entry"` — mark the public-facing entry reasoner. Always tag it.
- A domain tag (e.g., `"finance"`, `"risk"`, `"intake"`) — for filtering in discovery and the UI.

**Hard rule:** every entry reasoner gets `tags=["entry"]` so the user can find it via discovery without reading the source.

## `Agent(...)` constructor — verified signature

From `sdk/python/agentfield/agent.py:464`:

```python
app = Agent(
    node_id: str,                                 # REQUIRED. e.g. "customer-triage"
    agentfield_server: str | None = None,         # control plane URL. env: AGENTFIELD_SERVER. default http://localhost:8080
    version: str = "1.0.0",
    description: str | None = None,
    tags: list[str] | None = None,                # agent-LEVEL tags (distinct from per-reasoner tags)
    author: dict[str, str] | None = None,
    ai_config: AIConfig | None = None,            # default AIConfig.from_env(). Pass AIConfig(model="...") to set default
    harness_config: HarnessConfig | None = None,
    memory_config: MemoryConfig | None = None,
    dev_mode: bool = False,                       # verbose logs + DEBUG level. Always set True in scaffolds
    callback_url: str | None = None,              # else AGENT_CALLBACK_URL env, else auto-detect
    auto_register: bool = True,
    vc_enabled: bool | None = True,               # generate verifiable credentials for executions
    api_key: str | None = None,                   # X-API-Key header to control plane
    # ... other auth/DID parameters
)
```

**Critical things scaffolds get wrong:**
- The parameter is **`agentfield_server`** (not `agentfield_url`, not `server_url`). Verified in `agent.py:464`.
- Read it from env: `agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080")`.
- Set `dev_mode=True` in every scaffold so the user sees what's happening on first run.
- `Agent` subclasses FastAPI — you can use any FastAPI feature on it directly.

### `AGENT_CALLBACK_URL` env var

The agent node needs a URL the control plane can use to call back into it (for sync execution dispatch). In Docker Compose this is `http://<service-name>:<port>`. The SDK reads it from `AGENT_CALLBACK_URL`. You set it in the compose file:

```yaml
environment:
  AGENT_CALLBACK_URL: http://customer-triage:8001
```

If you don't set it, the SDK auto-detects, which works locally but is unreliable inside containers. **Always set it explicitly in the compose file** to the in-network DNS name of the service.

## `@app.reasoner()` real signature

Based on `agent.py:1612`, the decorator only accepts these parameters:

```python
@app.reasoner(
    path: str | None = None,             # default /reasoners/{func_name}
    name: str | None = None,             # override the registered ID
    tags: list[str] | None = None,
    *,
    vc_enabled: bool | None = None,      # inherits agent default
    require_realtime_validation: bool = False,
)
```

**Important things it does NOT accept:** `input_schema=`, `output_schema=`, `description=`, `version=`. **Schemas are derived from type hints.** The function's parameter type hints become the input schema; the return type hint becomes the output schema.

```python
class IntakeResult(BaseModel):
    contract_type: str
    confident: bool

@app.reasoner(tags=["entry"])
async def classify(text: str, model: str | None = None) -> IntakeResult:
    return await app.ai(system="...", user=text, schema=IntakeResult, model=model)
```

## `app.run()` is the entry point

`agent.py:4194` confirms `app.run()` auto-detects whether to launch in CLI mode (`af call`, `af list`, `af shell`) or server mode. **Always use `app.run()` in `__main__`**, not `app.serve()`:

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)
```

## Memory, vectors, and events — what's at your disposal

The SDK ships a persistent memory layer. Use it whenever a reasoner would otherwise re-compute, re-fetch, or lose something a later reasoner needs. Four scopes share a single API surface.

| Scope | Lifetime | Typical use |
|---|---|---|
| `global` | Cross everything | Knowledge bases, shared embeddings, constants outliving any run |
| `agent` | This node, all sessions | Cached lookups, warm data, node-wide state |
| `session` | One conversation / thread | Multi-turn chat state, per-user scratch |
| `run` | Single workflow execution | Intermediate results shared between reasoners in one run |

**Key-value API (same for every scope):**

```python
await app.memory.set(key, value, scope="run")
value = await app.memory.get(key, default=None, scope="run")
await app.memory.exists(key, scope="run")
await app.memory.delete(key, scope="run")
keys = await app.memory.list_keys(scope="agent")
```

**Vector memory — look up by meaning instead of by key.** Use when reasoners need retrieval over accumulated content (semantic cache, pattern library, "have I seen this before?"):

```python
await app.memory.set_vector(key, embedding, metadata={...}, scope="agent")
hits = await app.memory.search_vectors(
    query_embedding,
    top_k=5,
    scope="agent",
    filters={"domain": "..."},
)
```

**Event memory — consult the control plane's execution history.** Every reasoner execution is recorded as an event. Reasoners can query recent events for "has something similar already run?" checks, deduplication, and cross-run learning. The surface lives under `app.memory.events.*`; see `sdk/python/agentfield/memory_events.py` for the exact shape. Useful when a reasoner's work should build on what prior runs already discovered.

**When NOT to use memory:**

- If a value is trivially re-computable, recompute. Memory is for expensive or cross-boundary state.
- If two reasoners can pass a value directly via `app.call` kwargs, that is simpler.
- If state only lives inside one reasoner body, use a local variable.

**Rule of thumb:** memory is for state that crosses call boundaries OR must survive the current execution. Everything else is a local variable or a pass-through kwarg. When in doubt, start without memory and add it only when a concrete need (cost, latency, cross-run continuity) shows up.

## The `confident` flag pattern (mandatory for every `.ai()` gate)

Every `.ai()` schema includes a `confident: bool` field, and the call site checks it. **Three valid fallback options exist** when `confident` is false — pick the right one for the situation:

| Fallback option | When to use | Cost |
|---|---|---|
| **(a) Escalate to a deeper reasoner** | The system has another `@app.reasoner()` that can handle the harder case (chunked-loop, multi-call, more context) | Extra call |
| **(b) Deterministic safe default (RECOMMENDED for safety/regulated systems)** | The use case has a "safe" terminal state — `REFER_TO_HUMAN`, `REJECT`, `RETRY_LATER`, `NEEDS_HUMAN_REVIEW`. Return a Pydantic instance hard-coded to that safe state | Free |
| **(c) Escalate to `app.harness()`** | ONLY when `recommendation.harness_usable == true` from `af doctor`, AND the Dockerfile installs the CLI, AND there's a startup `shutil.which()` check | Heavy |

**Default for regulated, safety-critical, or judgment-based systems: option (b).** A confident-wrong automated decision is almost always worse than a referral. Build `fallback_*` constructors in `helpers.py` that return Pydantic instances hard-coded to the safe-default state.

### Pattern (a) — escalate to a deeper reasoner

```python
class IntakeDecision(BaseModel):
    contract_type: str
    complexity: str
    confident: bool

result = await app.ai(system="...", user="...", schema=IntakeDecision, model=model)

if not result.confident or result.complexity == "high":
    # Escalate to a deeper reasoner that can navigate more context
    result_dict = await app.call(
        f"{app.node_id}.deep_intake",
        document=full_document,
        partial=result.model_dump(),
        model=model,
    )
    result = DeepIntakeResult(**result_dict)
```

### Pattern (b) — deterministic safe default

```python
# In helpers.py:
def fallback_specialist_review(*, axis: str, reason: str) -> SpecialistReview:
    """Safe default Pydantic instance returned when an .ai() gate isn't confident."""
    return SpecialistReview(
        axis=axis,
        verdict="NEEDS_HUMAN_REVIEW",
        confidence_score=0.0,
        confident=False,
        rationale=reason,
        decisive_fact_ids=[],
    )

# In specialists.py:
review = await router.ai(system="...", user="...", schema=SpecialistReview, model=model)
if not review.confident:
    return fallback_specialist_review(
        axis=axis,
        reason=f"{axis} reviewer was not confident enough to automate a terminal view.",
    )
return review
```

This is the dominant pattern in real builds. The orchestrator at the top of the pipeline uses **deterministic governance overrides** (plain Python `if` statements) to convert any non-confident specialist into a `REFER_TO_HUMAN` final decision. The intelligence stays in the LLM; the safety stays in the code.

Every `.ai()` gate has a `confident` flag and one of these three fallback paths. No exceptions.

## What about long-document navigation?

The philosophy doc talks about "navigating documents" with a harness that has tools. In the actual SDK, you have three real options:

**Option A — `app.ai(tools=[...])` with custom tool definitions.** Define tools (e.g., `read_section(section_id)`, `search_document(query)`) the LLM can call iteratively. The `app.ai()` call becomes multi-turn automatically.

**Option B — Loop yourself in a `@app.reasoner()`.** Chunk the document with a `@app.skill()`, fan out `app.ai()` calls per chunk via `asyncio.gather`, then synthesize.

**Option C — `app.harness(provider="claude-code", tools=["read", "grep"])`.** Spawn a real coding agent CLI to navigate the document on the filesystem. Most powerful, also the most expensive.

Pick A for in-process tool-calling, B for embarrassingly-parallel chunked analysis, C for "I need a real agent to do file system work".

## The cost-of-being-wrong test

Before choosing `.ai()` without tools, ask: **"What does it cost the system if this call gets the wrong answer?"**

- Cheap to be wrong (a routing hint that gets corrected) → plain `.ai()` with `confident` flag
- Expensive to be wrong (a verdict the system commits to) → `.ai(tools=[...])` for iterative reasoning, or decompose into multiple narrower `.ai()` calls with adversarial verification

The financial cost of more reasoner calls is real but bounded. The reputation cost of a confidently-wrong answer propagating through your pipeline is unbounded.
