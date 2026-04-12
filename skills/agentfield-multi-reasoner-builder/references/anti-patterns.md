# Anti-Patterns — Deep Dive

The 13 hard rejections and the rationalization counters are inlined in `SKILL.md` so they fire on every invocation. **This file is the deep-dive reference** — load it when the user pushes back on a rejection, when you need to explain WHY in more depth, or when you're tempted to negotiate with yourself.

When the user (or your own drift) pushes you toward one of these, name the rule, explain why in one sentence, and offer the AgentField-native alternative. Don't apologize, don't equivocate.

## Hard rejections

### 1. Direct HTTP between reasoners

❌ `httpx.post("http://other-agent:8002/run", ...)`
✅ `await app.call(f"{app.node_id}.other_reasoner", ...)`

**Why:** The control plane needs to see every call to track the workflow DAG, generate verifiable credentials, replay executions, and apply observability. Direct HTTP makes the system invisible.

---

### 2. One giant reasoner doing 5 things

❌ `async def review_everything(doc): ...` (200 lines, 4 LLM calls inside)
✅ Decompose into 5 reasoners that the orchestrator coordinates with `app.call` and `asyncio.gather`.

**Why:** Granular decomposition is the forcing function for parallelism, observability, replayability, and quality. A monolithic reasoner is just a script with extra steps.

---

### 3. Static linear chain where the path depends on discoveries

❌ `intake → analyze → score → report` (always, in this order, regardless of intake)
✅ Intake routes to different downstream reasoners based on what it found. If risk is high, spawn a deep-dive harness. If complexity is low, skip the adversary.

**Why:** Dynamic routing IS the meta-level intelligence that distinguishes AgentField from chain frameworks. A static chain can be written in 30 lines of LangChain.

---

### 4. `.ai()` on a long document

❌ `await app.ai(prompt=full_50_page_contract, schema=Result)`
✅ A `.harness()` that can navigate the document with `read_section` / `lookup_definition` tools.

**Why:** `.ai()` is single-shot. It cannot adapt, navigate, or escalate. Stuffing a long doc into the prompt either truncates silently, blows the context window, or produces shallow answers because the model never reads past page 3.

---

### 5. Unbounded loops

❌ `while not confident: result = await app.ai(...)`
✅ `for _ in range(MAX_ROUNDS): ...` with a hard cap and an explicit break condition.

**Why:** "Keep going until confident" is how you get a $400 bug report. Every loop has a cap. Period.

---

### 6. Structured JSON shoved into another LLM as "context"

❌ `await app.ai(user=str(previous_findings.model_dump()), ...)`
✅ `await app.ai(user=format_findings_as_prose(previous_findings), ...)`

**Why:** LLMs reason over natural language, not over JSON serialization. Structured output between code and a reasoner is correct. Structured output between two reasoners is a smell — convert it to prose with the relevant context.

---

### 7. Replicating programmatic work with an LLM

❌ `await app.ai(prompt="Sort these 50 items by score", ...)`
✅ `sorted(items, key=lambda x: x.score, reverse=True)`

**Why:** You are paying for intelligence. Sorting is not intelligence. If a `for` loop or a sort function would do it, do it. Save the LLM calls for things that previously required a human expert.

---

### 8. Scaffold without a working `curl`

❌ "Here are the files; you can figure out how to test it."
✅ A README with the exact verification ladder (health → nodes → capabilities → execute) and a curl that returns a real reasoned answer.

**Why:** The promise is `docker compose up` + curl. If the user can't run those two commands and see real output, the build failed regardless of how nice the architecture looks on paper.

---

### 9. Multi-container agent fleet when one node would do

❌ Five Docker services for "research agent", "writer agent", "editor agent", "fact-checker agent", "publisher agent"
✅ ONE agent node with five reasoners. Same orchestration capability, 5× less ops surface.

**Why:** Reasoners are cheaper than containers. Use multiple containers only when there's a real boundary (separate teams, separate language runtimes, separate scaling profiles, separate trust domains). Otherwise, one node with many reasoners is the right shape.

---

### 10. Hardcoded model strings

❌ `ai_config=AIConfig(model="gpt-4o")`
✅ `ai_config=AIConfig(model=os.getenv("AI_MODEL", "openrouter/google/gemini-2.5-flash"))` AND accept a `model` parameter on the entry reasoner that propagates via `app.call(..., model=model)`.

**Why:** Users need to swap models per-request to A/B test without rebuilding the container. Make the model dynamic at three layers: env default, container override, per-request override.

---

### 11. Hardcoded `node_id` in `app.call`

❌ `await app.call("financial-reviewer.score", ...)`
✅ `await app.call(f"{app.node_id}.score", ...)`

**Why:** When the user renames the node via `AGENT_NODE_ID`, hardcoded calls break. Always reference your own reasoners through `app.node_id`.

---

### 12. `.ai()` with no `confident` flag and no fallback

❌ Schema is `{decision: str, reason: str}` and the call site doesn't validate.
✅ Schema is `{decision: str, reason: str, confident: bool}` and the call site checks `if not result.confident: escalate_to_harness()`.

**Why:** Every `.ai()` has a failure mode. A failed `.ai()` that propagates a confidently-wrong answer is the single most expensive bug an AgentField system can ship.

---

## Rationalization counters

When you (or the user) start producing one of these, recognize it and refuse:

| Rationalization | Counter |
|---|---|
| "Just for the demo, a chain is fine" | The demo is the proof. A weak demo proves nothing. |
| "The LLM is smart enough to handle the whole document in one call" | The LLM is 0.3-grade. The architecture is 0.8-grade. Don't mix them up. |
| "I'll add the harness later if it doesn't work" | You'll never know it doesn't work because the .ai() will silently truncate. Start with harness. |
| "Routing is overkill, the workflow is always the same" | Then the workflow doesn't justify AgentField. Tell the user honestly. |
| "I'll skip the curl smoke test, the user will figure it out" | The user invoked a skill. The skill's whole point is they don't have to figure it out. |
| "The CLAUDE.md is bureaucratic, the code is self-documenting" | Code documents WHAT. CLAUDE.md documents WHY this is the architecture and what NOT to undo. The next agent needs both. |
| "Two grooming questions is barely anything" | One question. The point is to feel magical to the first-time user. Infer the rest. |
| "I'll skip the discovery API check, I trust the build" | A curl that hangs at 30s tells you nothing about which step failed. Discovery API tells you in 2s. |
| "I'll ship the JSON directly to the next reasoner, it's cleaner" | Cleaner for you. Worse for the LLM. Convert to prose. |
| "More containers means better separation" | More containers means more YAML, more network hops, more failure modes. Use one node unless you have a real reason. |

## When the user explicitly demands a rejected pattern

Some users will insist. Honor that — but only after you've named the rejection, explained why in one sentence, and they've confirmed they understand the tradeoff. Then build it their way and add a comment in the code:

```python
# NOTE: User explicitly requested static chain over dynamic routing despite
# the canonical AgentField pattern being dynamic. See README "Tradeoffs" section.
```

The point is not to be a tyrant — it's to refuse drift. Conscious choices are fine. Drift is not.
