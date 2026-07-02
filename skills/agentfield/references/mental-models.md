# Mental Models — How to Think Before You Architect

Five frames that generate the rules in `SKILL.md`. Rules tell you what to do; these tell you why, so you can extrapolate when the rules run out. Load once per design session, before drawing the topology.

---

## 1. Orchestrated functions, not a DAG

Reasoners are async functions that call other reasoners through `app.call()`. There is no graph to declare. Control flow is ordinary Python: `if`/`else`, loops, `asyncio.gather`, recursion. The DAG in the dashboard is a **trace** of what happened, not a spec you wrote.

One-line contrast: declare-the-graph frameworks (LangGraph, CrewAI) fix the topology before the run; when the path depends on discoveries, a pre-declared graph cannot express the system.

What this unlocks — use these actively, don't rediscover them:

- **Recursion with a depth cap.** A reasoner drills into nested structure by calling itself.

```python
@app.reasoner()
async def drill(section: str, depth: int = 0, model: str | None = None) -> dict:
    finding = await app.ai(system=SYS, user=section, schema=Finding, model=model)
    if finding.has_substructure and depth < MAX_DEPTH:
        subs = await asyncio.gather(*[
            app.call(f"{app.node_id}.drill", section=s, depth=depth + 1, model=model)
            for s in finding.subsections
        ])
    ...
```

- **Runtime topology.** How many specialists, which ones, in what order — computed from intermediate results, not fixed at import time. `N = len(plan.axes)`, then gather over N calls.
- **Meta-prompting.** A parent writes a child's prompt from what it just discovered. The child's behavior is a runtime artifact of the parent's reasoning, not a template you authored.
- **Conditional deepening.** Spend more calls only where signal appears. A cheap first-pass classifier decides which 3 of 20 documents deserve the expensive deep pass.

If none of your control flow depends on intermediate state, you didn't need this runtime. Tell the user honestly.

---

## 2. Composing intelligence

An individual LLM call reasons at maybe 0.3–0.4 on a normalized scale. Deliberate composition of small, constrained, verifiable reasoners pushes system-level reasoning to 0.7–0.8 for a specific domain. The intelligence lives in the composition, not the components.

Three consequences — apply all of them:

- **One cognitive job, 2–4 output fields per reasoner.** Small enough to verify, cheap enough to run on a mid-tier model. If you can't state the reasoner's contract in one sentence, it's two reasoners.
- **Quality comes from structure, not from a bigger prompt or a bigger model.** Parallel perspectives on the same input. Adversarial verification (HUNT→PROVE) when false positives are expensive. Cross-checks between independent paths. When output quality disappoints, your first move is to reshape the graph, not upgrade the model.
- **Errors localize.** A wrong answer in a 12-reasoner graph points at one slot you can inspect in the trace. A wrong answer inside one giant completion dissolves into the whole prompt. Decomposition is a debugging strategy, not just a style.

---

## 3. Guided autonomy

The orchestrator sets the question and verifies the answer. It never scripts the steps.

For harnesses this becomes the **competence-predictability inversion**: the more capable the delegate, the less you should control HOW it works and the more you must verify WHAT it returns. An `app.ai()` call gets a tight prompt; an `app.harness()` gets a goal, a budget, and a schema — never a step list.

The membrane you engineer is the contract:

- **Budget in:** `max_budget_usd`, `max_turns`, depth caps on recursion, iteration caps on loops.
- **Tool surface:** the delegate touches only what you hand it.
- **Schema out:** a validated Pydantic instance, or it didn't happen.

Bounded autonomy at every level: every loop, spawn, and recursion carries an explicit integer cap. Freedom inside the membrane; zero freedom about the membrane.

---

## 4. The autonomy spectrum

Every primitive is a point on one axis: how much process visibility you give up in exchange for capability, and what verification that trade demands.

| Point | Primitive | Process | How you verify |
|---|---|---|---|
| Typed function call | `app.ai()` | One shot, no tools, transparent | Instantly, on the schema |
| Delegated engineer | `app.harness()` | Multi-turn, tools, opaque | Only at the boundary: schema + budget + spot-check |
| Manager | `@app.reasoner()` calling reasoners | Ordinary Python you wrote | Per sub-call, in the trace |

Choosing a primitive is choosing a point on this spectrum **plus the verification that point requires**. Moving right (more autonomy) without adding verification is the real failure mode — not the autonomy itself. The decision tree in `SKILL.md` is this spectrum turned into questions.

---

## 5. Intelligence in the gaps, code everywhere else

Anything deterministic is Python between reasoner calls: scoring formulas, sorting, dedup, thresholds, clearing logic, format conversion. LLM slots are reserved for judgment, discovery, synthesis — things that previously required a human expert.

The test: if you can write the function, write the function. `sorted(items, key=...)` beats `app.ai("sort these")` on cost, latency, and correctness every time.

Corollary (the archei rule): data format follows the consumer.

- Code branches on it → structured JSON (`if result.risk == "critical"`).
- Another LLM reads it → prose (`render_findings_as_text(findings)`). LLMs reason over language, not serialized dicts.
- Both → hybrid: enums and scores as fields, reasoning as a string field.

Parsing an LLM's prose with regex means that field should have been JSON. Feeding `str(model_dump())` to an LLM means that field should have been prose.

---

## The composite check

Before scaffolding, ask the five in order:

1. Where does the path depend on discoveries? That's your dynamic dispatch.
2. Where does structure substitute for model size? Parallel views, adversaries, cross-checks.
3. What is each delegate's membrane? Budget, tool surface, schema.
4. Which spectrum point is each slot on, and how is it verified?
5. What is deterministic and therefore Python?

If the answers are "nowhere, nowhere, n/a, all typed calls, everything" — the problem is one LLM call plus plumbing. Say that to the user instead of building a pretend mesh.
