# Primitives Snapshot — Offline Fallback

> **This file is a frozen snapshot. Prefer the live docs at https://agentfield.ai/llms-full.txt or `/llm/docs/<slug>` when you have a network.** Stamp: 2026-06-02. See `live-docs.md` for the fetch flow.
>
> When you read from this file, add to your output: "(offline snapshot — verify the surface against agentfield.ai if anything seems off)".

This is a minimal cheat sheet of the five primitives so an offline build can still work. **It is intentionally incomplete.** Anything not here is a sign you need the live docs.

---

## The five primitives

| Primitive | What it is | When to use |
|---|---|---|
| `@app.reasoner()` | Registers a function as a reasoner with the control plane | Wrap every cognitive unit |
| `@app.skill()` | Registers a deterministic function (no LLM) | Sort, parse, dedupe, score-with-formula |
| `app.ai(...)` | Single call OR multi-turn tool-using LLM call when `tools=` is passed | Classification, routing, structured analysis, stateful tool-using |
| `app.call(target, **kwargs)` | Call another reasoner THROUGH the control plane. Returns `dict`. Tracks the workflow DAG | All inter-reasoner traffic |
| `app.harness(prompt, provider=...)` | Delegate to an external coding-agent CLI (claude-code / codex / gemini / opencode) | When you need a real coding agent to write files / run shell |

---

## `Agent(...)` constructor

```python
app = Agent(
    node_id="my-agent",                                          # REQUIRED
    agentfield_server=os.getenv("AGENTFIELD_SERVER",
                                "http://localhost:8080"),
    version="1.0.0",
    description=None,
    tags=None,
    ai_config=AIConfig(model=os.getenv("AI_MODEL", "<default>")),
    harness_config=None,
    memory_config=None,
    dev_mode=True,                                               # always True in scaffolds
    callback_url=None,                                           # else AGENT_CALLBACK_URL env
    auto_register=True,
    vc_enabled=True,                                             # generate VCs per execution
    api_key=None,
)
```

**Param name is `agentfield_server`** (not `agentfield_url`).

---

## `@app.reasoner()` decorator

Real signature accepts only: `path`, `name`, `tags`, `vc_enabled`, `require_realtime_validation`. It does **NOT** accept `input_schema=`, `output_schema=`, `description=`, `version=`. **Schemas are derived from type hints.**

```python
class IntakeResult(BaseModel):
    contract_type: str
    confident: bool

@app.reasoner(tags=["entry"])
async def classify(text: str, model: str | None = None) -> IntakeResult:
    return await app.ai(system="...", user=text, schema=IntakeResult, model=model)
```

---

## `app.ai(...)` signature

```python
result = await app.ai(
    *args,                     # positional: text, urls, paths, bytes, dicts, lists (multimodal)
    system=None,               # system prompt
    user=None,                 # user prompt (alternative to positional)
    schema=None,               # Pydantic class for structured output
    model=None,                # PER-CALL model override
    temperature=None,
    max_tokens=None,
    stream=None,
    response_format=None,      # "auto" / "json" / "text" / dict / None
    tools=None,                # list of tool defs, OR "discover" to auto-discover
    context=None,
    memory_scope=None,         # e.g., ["workflow", "session", "reasoner"]
    **kwargs,
)
```

- `model=` is per-call. **Always thread it through from the entry reasoner** so the user can A/B test per request.
- `tools=[...]` makes `app.ai()` a multi-turn tool-using LLM. Use this for stateful reasoning over a corpus.
- `schema=` returns a validated Pydantic instance.

---

## `app.call(...)` semantics

```python
result_dict = await app.call(
    target: str,           # "node_id.reasoner_name"
    *args,
    **kwargs,
)
```

- Always returns a `dict`, even if the target returns a Pydantic model.
- Reference with `f"{app.node_id}.X"` — never hardcoded.
- **Every cross-call is tracked in the workflow DAG.** Direct HTTP between reasoners is forbidden.

### Cross-boundary serialization (the silent contract)

`app.call` crosses a JSON boundary even inside the same process. A Pydantic model goes in; a plain dict comes out, **regardless of receiver type hints**. Type hints document the shape on the wire, not the type in memory.

Two correct patterns:

```python
# (a) Reconstruct on the receiver
@router.reasoner()
async def downstream(payload: dict, model: str | None = None) -> FinalResult:
    typed = UpstreamResult(**payload)              # explicit reconstruction
    # ... use `typed` normally
```

```python
# (b) Render to prose BEFORE the call (preferred for LLM-to-LLM)
drafts = [Result(**d) for d in await asyncio.gather(*calls)]
drafts_prose = render_bundle(drafts)               # plain Python helper

verdict = await app.call(
    f"{app.node_id}.synthesizer",
    drafts_prose=drafts_prose,                     # receiver gets a string
    model=model,
)
```

Red flags that mean you hit this trap:
- `AttributeError: 'dict' object has no attribute 'X'`
- `TypeError: argument after ** must be a mapping, not NoneType`
- Pydantic ValidationError about missing required fields inside a list payload

---

## `AgentRouter` proxy surface

`AgentRouter` is **NOT** a universal transparent proxy. It proxies a fixed enumerated set:

| Attribute | Proxied? |
|---|---|
| `router.ai(...)` | ✅ |
| `router.call(...)` | ✅ |
| `router.memory` | ✅ |
| `router.harness(...)` | ✅ |
| `router.node_id` | ❌ — read from env: `NODE_ID = os.getenv("AGENT_NODE_ID", "<slug>")` |
| Other agent attributes | ❌ — never assume they proxy |

Default canonical pattern: `AgentRouter(prefix="", tags=["domain"])`. `prefix="clauses"` auto-namespaces every reasoner ID as `clauses_<func_name>`.

---

## `app.harness(...)` signature

```python
result = await app.harness(
    prompt: str,
    schema: type[BaseModel] | None = None,
    provider: "claude-code" | "codex" | "gemini" | "opencode" | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
    tools: list[str] | None = None,
    permission_mode: "plan" | "auto" | None = None,
    system_prompt: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    **kwargs,
)
# Returns HarnessResult with .text, .parsed, .result
```

**Harness availability gate** — DO NOT use unless ALL true:

1. `af doctor` reports `recommendation.harness_usable: true` AND lists the chosen provider.
2. The Dockerfile installs the chosen provider's CLI.
3. `main.py` has a startup `shutil.which("<binary>")` guard that exits with a clear error.
4. The README explicitly says the container ships with the CLI.

If any of those four are false, refactor to `app.ai(tools=[...])` or a chunked-loop reasoner.

---

## Memory

| Scope | Lifetime |
|---|---|
| `global` | Cross everything |
| `agent` | This node, all sessions |
| `session` | One conversation thread |
| `run` | Single workflow execution |

```python
await app.memory.set(key, value, scope="run")
v = await app.memory.get(key, default=None, scope="run")
await app.memory.exists(key, scope="run")
await app.memory.delete(key, scope="run")
keys = await app.memory.list_keys(scope="agent")
```

Vector memory (`set_vector` / `search_vectors`) and event memory (`app.memory.events.*`) also exist. See live docs for current API.

---

## `app.run()` is the entry point

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)
```

Auto-detects CLI vs server mode. **Always use `app.run()`** — not `app.serve()`.

---

## The `confident` flag pattern (mandatory)

Every `.ai()` schema includes `confident: bool`. The call site checks it and has a fallback path. Three valid fallbacks:

| Option | When to use |
|---|---|
| Escalate to a deeper reasoner | The deeper reasoner can handle harder cases |
| Return a safe-default Pydantic instance (`REFER_TO_HUMAN`) | Regulated / safety-critical systems — **default recommendation** |
| Escalate to `app.harness()` | Only if the harness gate passes |

```python
result = await app.ai(system="...", user="...", schema=MySchema, model=model)
if not result.confident:
    # one of the three fallbacks
    ...
```

---

## Per-call model propagation

```python
@app.reasoner(tags=["entry"])
async def entry(payload: dict, model: str | None = None) -> dict:
    plan = await app.ai(system="...", user="...", schema=Plan, model=model)
    children = await asyncio.gather(*[
        app.call(f"{app.node_id}.child", payload=payload, axis=a, model=model)
        for a in plan.axes
    ])
    ...

@app.reasoner()
async def child(payload: dict, axis: str, model: str | None = None) -> dict:
    return (await app.ai(system="...", user=f"axis: {axis}", schema=Result, model=model)).model_dump()
```

`app.call()` has **no** native model override — thread `model` as a kwarg.

---

## Verifying primitives against current SDK

If any signature in this file appears wrong at runtime:

1. Fetch the relevant page from `agentfield.ai/llm/docs/build/building-blocks/<topic>` and re-check.
2. Read the SDK source directly: `sdk/python/agentfield/agent.py`, `router.py`, `tool_calling.py`.
3. Run `af agent kb guide --goal "use <primitive>"` for goal-oriented examples.

Snapshots drift. Live docs and the SDK source are the only durable truths.
