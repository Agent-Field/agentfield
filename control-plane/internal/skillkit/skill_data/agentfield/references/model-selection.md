# Model Selection — env → ask → live-pick → fallback

The model is a per-build decision and a per-request override. **Never hardcode a model string in user-facing scaffolds.** Always make it flow from environment → AIConfig default → per-request kwarg.

This file is the flow for picking *which* model to bake as the default. The user can always override at request time via `{"input": {..., "model": "..."}}`.

---

## The flow

```
1. Run `af doctor --json`.
   ├─ recommendation.provider != "none" AND user did NOT specify a model
   │  → use recommendation.ai_model as the default.
   │
   ├─ recommendation.provider == "none" (no key set)
   │  → ASK the user which provider they want (AskUserQuestion).
   │  → If user picks OpenRouter (default) and doesn't name a model
   │    → go to step 3 (live-pick from OpenRouter)
   │  → Document the env var they need to set.
   │
   └─ User explicitly named a model in the brief
      → use it as-is. Skip the rest.

2. If the user only said "use the best open-source / cheap model"
   → step 3.

3. Live-pick from OpenRouter (when OpenRouter is the provider).

4. Fallback if /models is unreachable
   → use the frozen open-weight pick below.
```

---

## Asking the user when there's no key

When `af doctor` reports `provider_keys.*.set == false` for every provider, use `AskUserQuestion` to ask which provider they want. Default to OpenRouter — it's the cheapest path to access many models with one key.

Recommended question (single-select):

> "I don't see a provider key in your environment. Which provider do you want this agent to use? You'll paste the key into `.env` before running."

Options:
- **OpenRouter (recommended)** — one key, access to most open-weight and commercial models. `OPENROUTER_API_KEY`. Cheapest for experimentation.
- **OpenAI** — `OPENAI_API_KEY`. GPT-4o family.
- **Anthropic** — `ANTHROPIC_API_KEY`. Claude family.
- **Google** — `GOOGLE_API_KEY`. Gemini family.

If the user picked OpenRouter and didn't name a specific model, **proceed to live-pick** below.

---

## Live-pick from OpenRouter (for "best open-source" requests)

OpenRouter exposes its model catalog at `https://openrouter.ai/api/v1/models` (no auth required, just `GET`). Use `WebFetch` to pull it and filter.

### Filtering criteria

Default to:
- `pricing.prompt` ≤ **$1.00 per million tokens** (cheap)
- `context_length` ≥ **32768** (works for chunked-loop reasoners over modest documents)
- **Open-weight provider** — model `id` prefix is one of:
  - `qwen/` (Qwen 2.5 / Qwen 3 family)
  - `deepseek/` (DeepSeek V3, V3.1, R1)
  - `meta-llama/` (Llama 3.x, Llama 4)
  - `mistralai/` (Mistral, Mixtral, Codestral)
  - `google/gemma-` (Gemma)
  - `nousresearch/` (Hermes)
  - `microsoft/wizardlm` / `microsoft/phi-`
  - `cognitivecomputations/dolphin-`
  - `nvidia/nemotron-`
  - `01-ai/` (Yi)
- Reasonably recent: prefer models released in the last ~9 months when ties.

Filtering pseudocode (from `models = (await WebFetch("https://openrouter.ai/api/v1/models")).data`):

```python
OPEN_WEIGHT_PREFIXES = ("qwen/", "deepseek/", "meta-llama/", "mistralai/",
                       "google/gemma", "nousresearch/", "microsoft/wizardlm",
                       "microsoft/phi-", "01-ai/", "nvidia/nemotron-")

candidates = [
    m for m in models
    if m["id"].startswith(OPEN_WEIGHT_PREFIXES)
    and float(m["pricing"]["prompt"]) * 1_000_000 <= 1.00
    and m["context_length"] >= 32_768
]

# Rank: prefer higher context, then lower price, then more recent
candidates.sort(key=lambda m: (
    -m["context_length"],
    float(m["pricing"]["prompt"]),
    -m.get("created", 0),
))
top_three = candidates[:3]
```

Then present the top 3 to the user via `AskUserQuestion` and let them pick. Use the OpenRouter `id` field as the AgentField model string, prefixed with `openrouter/` (e.g., `openrouter/deepseek/deepseek-v3.1`).

### When live-pick is unreachable

If `WebFetch` to OpenRouter fails (offline, rate-limited, transient), fall back to one of these frozen choices and **say so** in the assumptions list:

| Frozen open-weight default | Notes |
|---|---|
| `openrouter/deepseek/deepseek-v3.1` | Strong reasoning, ~$0.30/M prompt, 64k context |
| `openrouter/qwen/qwen-2.5-72b-instruct` | Cheap, 32k context, broad capability |
| `openrouter/meta-llama/llama-3.3-70b-instruct` | Reliable baseline, 128k context |

Pick the first one available. If even those fail validation against the live model list when the smoke test runs, the user can override via `AI_MODEL` env or per-request `model=` kwarg.

---

## Why open-weight is the default for "best open-source"

When the user says "best open-source" they usually mean "Apache-licensed / open-weight model I can run anywhere if I leave OpenRouter." That excludes GPT-4o, Claude Sonnet, Gemini Pro — all commercial closed models.

If the user instead says "best cheap model" or "best reasoning model" without "open-source", widen the filter to include `openai/gpt-4o-mini`, `anthropic/claude-haiku-4-5`, `google/gemini-2.5-flash` — these are commercial but very cost-effective.

---

## Per-request model override (mandatory)

Regardless of the default, **every scaffold must support per-request model override** so the user can A/B test without rebuilding the container.

```python
# Entry reasoner
@app.reasoner(tags=["entry"])
async def entry(payload: dict, model: str | None = None) -> dict:
    plan = await app.ai(system="...", user="...", schema=Plan, model=model)
    children = await asyncio.gather(*[
        app.call(f"{app.node_id}.child_{i}", payload=payload, model=model)
        for i in plan.children
    ])
    ...

# Every child reasoner
@app.reasoner()
async def child_i(payload: dict, model: str | None = None) -> dict:
    return (await app.ai(system="...", user="...", schema=Result, model=model)).model_dump()
```

The user then runs:

```bash
curl -X POST http://localhost:8080/api/v1/execute/async/<slug>.entry \
  -d '{"input": {"...": "...", "model": "openrouter/google/gemini-2.5-flash"}}'
```

When `model` is omitted, the AIConfig default (from `AI_MODEL` env) is used. **`app.call()` has no native model override parameter — you MUST thread `model` as a regular reasoner kwarg.**

---

## Speed vs depth — pick fast for the default

The control plane's **sync** execute endpoint has a hard 90-second timeout. Multi-reasoner pipelines often blow past that with slow models. **Default to a fast model** so the canonical curl works on first try; let the user opt into a slower model per request if they want depth.

If the use case truly demands a slow reasoning model (legal, medical, complex analysis), use the **async** endpoint in the smoke test (already the canonical choice) and document the latency expectation in the README.
