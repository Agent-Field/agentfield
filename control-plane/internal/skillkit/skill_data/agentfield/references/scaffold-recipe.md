# Scaffold Recipe — File-by-File Generation Contract

Every AgentField build produces this exact set of files. No omissions, no "I'll add that later."

**This file is the WHAT and the WHERE.** The HOW (which topology to choose, which patterns to compose) lives in `patterns-emerge.md` and `examples-map.md`. The actual `main.py` content depends on your design — there's no canonical example here because there's no canonical pattern. Pick the shape that emerges from the principles.

---

## Layout

```
<slug>/
├── main.py
├── reasoners/                  # only if > 4 reasoners
│   ├── __init__.py
│   ├── models.py               # Pydantic schemas (one place to type-check)
│   ├── helpers.py              # plain Python utilities (no decorators)
│   └── <domain>.py             # one or more AgentRouter files
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── requirements.txt
├── README.md
└── CLAUDE.md
```

`<slug>` is lowercase-hyphenated, derived from the use case (`customer-triage`, `incident-analyzer`, `meeting-summarizer`, etc.).

---

## Use `af init` to lay the foundation, then layer your real architecture on top

```bash
af init <slug> --language python --docker --defaults --non-interactive \
  --default-model <model_from_af_doctor>
```

This produces the **four infrastructure files** that don't need customization:

- `Dockerfile` — universal Python 3.11-slim
- `docker-compose.yml` — control-plane + agent service with healthcheck and `depends_on: service_healthy`
- `.env.example` — all four provider keys + `AI_MODEL` default
- `.dockerignore`

Plus the **language scaffold** that you WILL rewrite:

- `main.py`, `reasoners.py`, `requirements.txt`, `README.md`, `.gitignore`

And then you generate yourself:

- `CLAUDE.md` — from `references/project-claude-template.md`, AFTER you know your entry reasoner name
- A README with the real curl — replace the generic one

**Do not customize Dockerfile / docker-compose.yml / .dockerignore unless you have a real reason.** The `af init` output is correct as-is.

---

## File 1: `main.py` — your real architecture

The shape depends on your design. The **invariants** are below; the body is up to you.

### Hard invariants (every `main.py`)

```python
import asyncio
import os
from typing import Any

from agentfield import Agent, AIConfig
from pydantic import BaseModel, Field


# ----- Schemas (Pydantic, type-hinted) -----
# Schemas come from type hints. Do NOT pass input_schema= / output_schema= to @app.reasoner.

class MyResult(BaseModel):
    field_one: str
    confident: bool                    # MANDATORY on every .ai() gate


# ----- Agent (the only Agent(...) call in the system) -----

app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "<slug>"),
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    ai_config=AIConfig(model=os.getenv("AI_MODEL", "<model_from_af_doctor>")),
    dev_mode=True,
)


# ----- Reasoners (your real architecture goes here) -----

@app.reasoner()
async def child_a(payload: dict, model: str | None = None) -> MyResult:
    return await app.ai(system="...", user=str(payload), schema=MyResult, model=model)


@app.reasoner(tags=["entry"])
async def entry(payload: dict, model: str | None = None) -> dict:
    # Real composition. Decide based on principles, not template.
    result_dict = await app.call(f"{app.node_id}.child_a", payload=payload, model=model)
    result = MyResult(**result_dict)                                   # reconstruct across boundary
    if not result.confident:
        return {"status": "NEEDS_REVIEW", "reason": "child_a not confident"}
    # ... more orchestration
    return result.model_dump()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)
```

### Invariants checklist for `main.py`

- [ ] `node_id`, `agentfield_server`, `model` all read from env with defaults
- [ ] `dev_mode=True` so the user sees what's happening on first run
- [ ] `auto_port=False` so the port is deterministic and the curl works
- [ ] Exactly ONE entry reasoner with `tags=["entry"]`
- [ ] Schemas come from type hints — no `input_schema=` / `output_schema=` / `description=` on `@app.reasoner`
- [ ] Every `.ai()` gate has a `confident: bool` field and a fallback
- [ ] Every reasoner that calls `app.ai()` accepts `model: str | None = None` and threads it
- [ ] Entry reasoner accepts `model` and propagates via `app.call(..., model=model)`
- [ ] All `app.call(...)` use `f"{app.node_id}.X"` — no hardcoded node IDs
- [ ] No direct HTTP between reasoners
- [ ] `app.run()` in `__main__` — not `app.serve()`

---

## File 2: the `reasoners/` package (for >4 reasoners)

```
<slug>/
├── main.py
└── reasoners/
    ├── __init__.py
    ├── models.py
    ├── helpers.py
    └── <domain1>.py            # one router per logical grouping
```

### `reasoners/__init__.py`

```python
from .<domain1> import router as <domain1>_router
# ... other routers

__all__ = ["<domain1>_router", ...]
```

### `reasoners/models.py`

Every Pydantic schema used anywhere in the system. Single place to type-check, prevents circular imports.

### `reasoners/helpers.py`

Plain Python utilities — not decorated. Math, prose renderers (Pydantic → string for LLM context), schema construction, fallback constructors (`fallback_X(*, reason)` returns safe-default instances).

> **Why plain helpers vs `@app.skill()`?** `@app.skill()` makes a function discoverable through the control plane. Use it when the deterministic function might be called from another reasoner via `app.call` OR from an external caller. Internal helpers (math, prose rendering) are cleaner as plain Python — no decorator overhead, no registration.

### `reasoners/<domain>.py`

One `AgentRouter` per file:

```python
import os
import asyncio
from agentfield import AgentRouter

from .models import MyResult                  # Pydantic schemas

NODE_ID = os.getenv("AGENT_NODE_ID", "<slug>")        # router.node_id does NOT exist
router = AgentRouter(prefix="", tags=["<domain>"])    # prefix="" → no auto-namespace


@router.reasoner()
async def my_reasoner(payload: dict, model: str | None = None) -> MyResult:
    return await router.ai(system="...", user=str(payload), schema=MyResult, model=model)
```

In `main.py`:

```python
from reasoners import <domain1>_router

app.include_router(<domain1>_router)
```

### Router gotchas

- `router.ai`, `router.call`, `router.memory`, `router.harness` proxy to the attached agent.
- `router.node_id` and other data attributes **do not** proxy. Read from env with `NODE_ID = os.getenv("AGENT_NODE_ID", "<slug>")`.
- `prefix="clauses"` auto-namespaces reasoner IDs as `clauses_<func_name>`. Default `prefix=""` keeps them raw.

---

## File 3: `Dockerfile` (universal — do not customize)

`af init --docker` generates it. Shape:

```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt
COPY . /app/
EXPOSE 8001
CMD ["python", "main.py"]
```

Build context is the project directory itself (`context: .`), so the same scaffold works whether the project lives in `code/examples/` or standalone at `/tmp/my-build/`.

**Only edit if you need to install a harness CLI** (claude-code, codex, gemini, opencode). See `primitives-snapshot.md` → "Harness availability gate". Otherwise leave it alone.

---

## File 4: `docker-compose.yml` (universal — do not customize)

`af init --docker` generates it. Shape:

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
      AGENTFIELD_OPENROUTER_SITE_URL: ${AGENTFIELD_OPENROUTER_SITE_URL:-https://agentfield.ai}
      AGENTFIELD_OPENROUTER_APP_NAME: ${AGENTFIELD_OPENROUTER_APP_NAME:-AgentField AI}
      OR_SITE_URL: ${OR_SITE_URL:-https://agentfield.ai}
      OR_APP_NAME: ${OR_APP_NAME:-AgentField AI}
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY:-}
      AI_MODEL: ${AI_MODEL:-<model_from_af_doctor>}
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

Hard requirements:
- Control plane has a healthcheck; agent uses `condition: service_healthy`
- All four provider env vars are exposed with `:-` defaults
- `AGENT_CALLBACK_URL` is the in-network DNS name (not `localhost`)
- Default model is whatever `af doctor` recommended (or the user picked)

---

## File 5: `.env.example`

```bash
# Required: pick ONE provider
OPENROUTER_API_KEY=sk-or-v1-...
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...

# Model — must match the provider above
AI_MODEL=<model_from_af_doctor>

# Optional overrides
AGENT_NODE_ID=<slug>
AGENT_NODE_PORT=8001
AGENTFIELD_HTTP_PORT=8080
```

---

## File 6: `requirements.txt`

```
agentfield
pydantic>=2.0
```

Add libraries the reasoners actually need (httpx, beautifulsoup4, pdfplumber, etc.). Do not pin agentfield's version unless you need to.

---

## File 7: `.dockerignore`

```
__pycache__
*.pyc
.pytest_cache
.env
.venv
*.log
```

---

## File 8: `README.md`

Customize for the build. Must include:

- 2-sentence description
- "Run" section: `cp .env.example .env`, paste key, `docker compose up --build`
- "Verify" section: discovery/capabilities curl (primary registration check)
- "Try it" section: canonical **async** curl with realistic input (use the user's sample data if provided)
- "Showpiece" section: the verifiable workflow chain curl
- "Stop" section: `docker compose down`

Use the templates in `verification.md` for the curl content.

---

## File 9: `CLAUDE.md`

Generate from `project-claude-template.md`. Customize every `<placeholder>`.

---

## Generation order

1. Decide the topology (read `patterns-emerge.md` + closest example via `examples-map.md` + walk the five principles).
2. Run `af init <slug> --language python --docker --defaults --non-interactive --default-model <model>`.
3. **Rewrite `main.py`** with your real architecture.
4. If > 4 reasoners, build the `reasoners/` package (models, helpers, routers).
5. Write the customized `README.md` (real curl, real example data).
6. Write `CLAUDE.md` from the template.
7. Validate (next section).

---

## Validation

### Static (always run)

```bash
python3 -m py_compile main.py
python3 -m py_compile reasoners/*.py 2>/dev/null || true
OPENROUTER_API_KEY=sk-or-v1-FAKE docker compose config > /dev/null
```

### Live smoke test (always run before handoff)

See `verification.md` → "Mandatory live smoke test". A build is not done until the canonical async curl returns `status: "succeeded"` with a real reasoned `result`.

### Visual-invariant checklist

Re-verify before handoff. Every box must be checked:

- [ ] `app.run(...)` in `__main__` (NOT `app.serve(...)`)
- [ ] Entry reasoner has `tags=["entry"]`
- [ ] Every `app.ai(...)` gate's schema includes `confident: bool`, AND the call site has a fallback
- [ ] Every reasoner that calls `app.ai(...)` accepts `model: str | None = None` and threads `model=model`
- [ ] Entry reasoner accepts `model` and propagates via `app.call(..., model=model)`
- [ ] All `app.call(...)` use `f"{app.node_id}.X"` — no hardcoded node IDs
- [ ] No `requests.post()` / `httpx.post()` between reasoners
- [ ] No `app.harness(provider="...")` unless the Dockerfile installs the CLI AND `main.py` has `shutil.which()` check
- [ ] No `input_schema=` / `output_schema=` parameters on `@app.reasoner()`
- [ ] README curl uses body shape `{"input": {...kwargs...}}` (NOT raw kwargs at top level)
- [ ] README smoke test uses `POST /api/v1/execute/async/...` + polling (NOT sync — 90s timeout)
- [ ] README registration check uses `/api/v1/discovery/capabilities` (primary, durable)
- [ ] `Agent(agentfield_server=os.getenv("AGENTFIELD_SERVER", ...))` — exact parameter name
- [ ] `AGENT_CALLBACK_URL` set in compose to the in-network DNS name
- [ ] `auto_port=False` in `app.run()`
- [ ] CLAUDE.md exists with no `<placeholder>` tokens left in it
- [ ] `.env.example` lists all four provider keys
- [ ] If split into routers: each reads `NODE_ID = os.getenv("AGENT_NODE_ID", "<slug>")` — never `router.node_id`
- [ ] No Pydantic instances or lists thereof passed across `app.call` without reconstruction on the receiver OR prose rendering on the sender
- [ ] LLM-to-LLM context is prose, not raw JSON dicts
- [ ] Depth from entry to leaf is ≥ 3 (every "specialist" itself calls 2–4 sub-reasoners)

If any box fails, fix before handoff. "Almost works" is worth zero.
