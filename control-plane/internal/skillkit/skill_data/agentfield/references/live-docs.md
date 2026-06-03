# Live Docs — agentfield.ai is the source of truth

The SDK and the control plane evolve. This skill ships with a frozen snapshot in `primitives-snapshot.md` for the offline case, but **the live docs are authoritative**. Fetch them first, every time.

---

## The endpoints

agentfield.ai publishes machine-readable docs at four levels of detail. Pick the cheapest one that answers your question.

| Endpoint | Size | When to fetch |
|---|---|---|
| `https://agentfield.ai/llms.txt` | ~3 KB | **Always, first call.** Index + `Contract-Version` header. Tells you what changed. |
| `https://agentfield.ai/docs-ai.json` | ~30 KB | When you want a structured list of every doc page with slugs, titles, keywords, difficulty. Use this to pick which page(s) to fetch. |
| `https://agentfield.ai/llm/docs/<slug>` | ~5–20 KB | A single focused page. Use when you know exactly which topic you need. |
| `https://agentfield.ai/llms-full.txt` | ~370 KB / 10k lines | The full corpus. Use when you need broad context (e.g., redesigning a primitive). |
| `https://agentfield.ai/openapi.json` | varies | REST API schema. Use when wiring an external client to the control plane. |

`/llms.txt` includes a `Contract-Version: <date>-v<N>` header. Use it as a cache key.

---

## Suggested fetch order on every invocation

```
1. WebFetch https://agentfield.ai/llms.txt
   → Read Contract-Version + the "Documentation Pages" list
   → If Contract-Version matches your cached version → done, skip step 2
   → Otherwise → step 2

2. WebFetch https://agentfield.ai/docs-ai.json
   → Parse .docs[] for slug + keywords
   → Note the 3–6 page slugs that match your build (e.g., reasoners, routers, ai-config, harness, triggers, memory)

3. For each needed page:
   WebFetch https://agentfield.ai/llm/docs/<slug>
   → Or just pull /llms-full.txt once if you'll need a wide surface
```

**Cache.** Write the responses to `~/.agentfield/cache/<filename>` with a 24-hour TTL. Skip re-fetch if the file is newer than 24h AND the `Contract-Version` matches what you saw last time.

---

## What lives where (current as of 2026-03-24 contract)

The top-level structure of `llms.txt` lists these page categories:

| Page | Slug | When you'd read it |
|---|---|---|
| Quickstart | `/learn/quickstart` | First-time scaffold |
| Features | `/learn/features` | When the user asks "what does AgentField actually give me?" |
| Python SDK | `/reference/sdks/python` | Every Python build |
| TypeScript SDK | `/reference/sdks/typescript` | TS builds |
| Go SDK | `/reference/sdks/go` | Go builds |
| REST API | `/reference/sdks/rest-api` | External client wiring |
| CLI Reference | `/reference/sdks/cli` | When unsure which `af` subcommand to use |
| Deployment | `/reference/deploy` | Production deploy guidance |
| Testing | `/reference/testing` | Building a test suite around the agent |
| Troubleshooting | `/reference/troubleshooting` | When the smoke test fails |
| Building blocks: Agents | `/build/building-blocks/agents` | The `Agent(...)` constructor |
| Building blocks: Reasoners | `/build/building-blocks/reasoners` | `@app.reasoner` signature, type-hint schemas |
| Building blocks: Routers | `/build/building-blocks/routers` | `AgentRouter` + the proxy surface |
| Building blocks: Skills | `/build/building-blocks/skills` | `@app.skill` |
| Coordination: Cross-agent calls | `/build/coordination/cross-agent-calls` | `app.call` semantics + cross-boundary gotcha |

Slugs come from `docs-ai.json` `.docs[].url`. Don't paste them from this file; pull them fresh.

---

## What this skill no longer carries inline

Things that **used to** live in the bundled references but now live live:

- `Agent(...)` constructor signature → fetch `/llm/docs/build/building-blocks/agents`
- `app.ai(...)` parameter list → fetch `/llm/docs/build/building-blocks/reasoners`
- `app.call(...)` semantics + cross-boundary serialization → fetch `/llm/docs/build/coordination/cross-agent-calls`
- `app.memory` scopes + vector + event surfaces → fetch the memory page
- `AgentRouter` proxy surface (which attributes proxy) → fetch the routers page
- `@app.reasoner()` decorator real signature → fetch the reasoners page
- Trigger envelope shape per source → fetch the triggers page

The offline `primitives-snapshot.md` carries a frozen version of this content. **Use it only when `agentfield.ai` is unreachable.** Stamp every snapshot read with a warning in your output: "(offline snapshot from <date> — may be stale)".

---

## How to use the live docs during design

1. **Before scaffolding:** fetch `llms.txt` + the 3–5 pages most relevant to your build. Read them. Use real signatures.
2. **When you hit a contract question:** "Does the router proxy `node_id`?" — fetch `/llm/docs/build/building-blocks/routers`. Don't guess.
3. **When the live smoke test fails with an `AttributeError`:** fetch the relevant page. Surface drift is the #1 silent-bug source.
4. **When proposing the user a "next iteration":** scan the features page for capabilities they haven't used yet (memory scopes, VC chains, triggers).

---

## Examples are the second source of truth

Beyond docs, the `code/examples/` folder in this repo is a live catalog of real builds. Pages describe the API; examples show real composition. See `references/examples-map.md` for the index.

When designing, you should pick **one live example whose problem shape resembles yours**, grep it for the actual reasoner topology, and adapt — not copy the patterns described in this skill's prose.

---

## When the live docs are unreachable

In order of preference:

1. Read the cached copy under `~/.agentfield/cache/`.
2. Read `references/primitives-snapshot.md` — frozen offline fallback.
3. `af agent kb topics` / `af agent kb search "<topic>"` / `af agent kb guide --goal "<intent>"` — the af binary embeds a knowledge base of goal-oriented dev guides. See `references/cli-toolkit.md`.
4. Grep `code/examples/` for the closest analog and read the actual code.

If you fall through all four and still don't know what to do, **tell the user** rather than guess. The skill's value is correctness; guessing the SDK surface produces builds that crash at runtime.
