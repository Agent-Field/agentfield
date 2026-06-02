# Anti-Patterns — Refuse Without Negotiation

The hard rejections from `SKILL.md` in one-line form. Load when tempted to take a shortcut or when the user pushes back on a rejection.

When the user (or your own drift) reaches for one of these, name the rule, give the one-sentence reason, propose the AgentField alternative. Don't apologize, don't equivocate.

---

## Hard rejections

| ❌ | ✅ | Why |
|---|---|---|
| `httpx.post(other_agent_url, ...)` | `app.call(f"{app.node_id}.X", ...)` | The CP needs to see every call for DAG / VCs / replay / observability. Direct HTTP makes the system invisible. |
| One giant reasoner doing 5 things | 5 reasoners + orchestrator with `app.call` + `asyncio.gather` | Granular decomposition is the forcing function for parallelism, observability, replayability. A monolithic reasoner is a script with extra steps. |
| Static linear chain when the path depends on findings | Dynamic routing on intermediate state | Dynamic routing IS the meta-level intelligence that distinguishes AgentField. A static chain can be written in 30 lines of LangChain. |
| `app.ai(prompt=full_50_page_doc)` | Chunked-loop reasoner OR `app.ai(tools=[...])` OR `app.harness` | `.ai()` is single-shot. It cannot adapt or navigate. Stuffing a long doc truncates silently or blows the window. |
| `while not confident: ...` (no cap) | `for _ in range(MAX): ...` with explicit break | Unbounded loops are how you get a $400 bug report. |
| `app.ai(user=str(prev.model_dump()))` | `app.ai(user=format_as_prose(prev))` | LLMs reason over prose, not serialized JSON. JSON between code-and-LLM is fine; JSON between two LLMs is a smell. |
| `app.ai("sort these 50 by score")` | `sorted(items, key=...)` | Sorting is not intelligence. If a `for` loop or sort function would do it, do it. |
| Scaffold without a working live curl | The canonical async curl + poll in the README | The skill's promise is "docker compose up + curl returns real output". A failing handoff is a failure regardless of how nice the architecture looks. |
| Multi-container fleet for what one node would do | One agent node, many reasoners | Reasoners are cheaper than containers. Only use multi-container when there's a real boundary (separate teams / languages / scaling / trust domains). |
| `AIConfig(model="gpt-4o")` (hardcoded) | `AIConfig(model=os.getenv("AI_MODEL", "<default>"))` + per-request `model=` override | Users need to A/B test models per request without rebuilding the container. |
| `app.call("slug.X", ...)` (hardcoded node id) | `app.call(f"{app.node_id}.X", ...)` | Rename via `AGENT_NODE_ID` breaks hardcoded calls. |
| `.ai()` schema with no `confident` field, no fallback | Schema includes `confident: bool`, call site checks and escalates | A confidently-wrong propagated answer is the most expensive bug an AgentField system can ship. |
| `app.harness(provider="claude-code")` in default scaffold | `app.ai(tools=[...])` or chunked-loop reasoner | Default container has no `claude` CLI. Will crash on first run. Use harness only if `af doctor` reports `harness_usable: true` AND Dockerfile installs the CLI AND main.py has `shutil.which()` guard. |
| Long multi-step reasoning **inside** a triggered reasoner | Triggered reasoner is a router; fan out to `@app.reasoner` specialists via `app.call` + `asyncio.gather` | Triggers are entry surfaces. Keep them thin. |
| Hardcoded webhook secret | `secret_env="STRIPE_WEBHOOK_SECRET"` | CP reads env at request time; value never leaves the host. |
| `transform=` doing I/O or async work | Sync pure envelope-peeling only | Do I/O inside the reasoner. |
| `input_schema=` / `output_schema=` / `description=` on `@app.reasoner` | Schemas come from type hints | Those decorator params don't exist. |
| `app.serve()` in `__main__` | `app.run()` | `app.run()` auto-detects CLI vs server mode. |
| Pydantic instance across `app.call` expecting reconstitution | `Model(**payload)` on receiver, OR render prose on sender | Cross-boundary serialization drops the type. Receiver gets a `dict`, not the original instance. |
| Trusting prose contracts that say "every" / "all" / "transparently forwards" | Verify the exact attribute against an enumerated list (router proxy table, etc.) | Surface contracts are always narrower than the words describing them. |
| Treating `py_compile` + `docker compose config` as proof the build works | Run the live async smoke test before handoff | Static checks catch syntax, not contract drift. |

---

## Rationalization counters

When you (or the user) reach for one of these, recognize it and refuse.

| Rationalization | Counter |
|---|---|
| "Just for the demo, a chain is fine" | The demo IS the proof. A weak demo proves nothing. |
| "The LLM is smart enough to handle the whole document in one call" | The LLM is 0.3-grade. The architecture is 0.8-grade. Don't mix them up. |
| "I'll add the harness later if it doesn't work" | You'll never know it doesn't work — `.ai()` will silently truncate. Start with the right primitive. |
| "Routing is overkill, the workflow is always the same" | Then the workflow doesn't justify AgentField. Tell the user honestly. |
| "I'll skip the curl smoke test, the user will figure it out" | The user invoked a skill. The whole point is that they don't have to figure it out. |
| "CLAUDE.md is bureaucratic, the code is self-documenting" | Code documents WHAT. CLAUDE.md documents WHY this is the architecture and what NOT to undo. The next agent needs both. |
| "I'll ask 5 questions to be safe" | Ask 1–3 narrow ones only when the answer changes the architecture. State assumptions for the rest. |
| "I'll skip discovery/capabilities, I trust the build" | A curl that hangs at 30s tells you nothing about which step failed. Discovery tells you in 2s. |
| "I'll ship JSON directly to the next reasoner, it's cleaner" | Cleaner for you. Worse for the LLM. Convert to prose. |
| "More containers means better separation" | More containers means more YAML, more network hops, more failure modes. Use one node unless there's a real boundary. |
| "Static checks passed, the build is done" | Static checks prove syntax. Live execution proves the contract. Run the canonical async curl before handoff. |

---

## When the user explicitly demands a rejected pattern

Honor it — but only after you've named the rejection, explained why in one sentence, and they've confirmed they understand the tradeoff. Then build it their way and add a comment:

```python
# NOTE: User explicitly requested static chain over dynamic routing despite
# the canonical AgentField pattern being dynamic. See README "Tradeoffs" section.
```

The point is to refuse drift, not be a tyrant. Conscious choices are fine. Drift is not.
