# The Generative Theory — Deriving an Orchestration From the Problem

Patterns are outputs of thinking, not inputs. Start from a named pattern (HUNT→PROVE, factory loops) and you can imitate builds that already exist; you cannot derive the right orchestration for a problem that looks like neither security auditing nor contract review. This document is the procedure that generates the topology from the problem itself. Run it once per design session, in order. The named shapes in `patterns-emerge.md` are what you may discover you have built afterwards.

The procedure:

1. **Decompose by cognitive jobs** — the expert's workflow becomes the reasoner set.
2. **Place each slot on the autonomy spectrum** — and accept that point's verification price.
3. **Assign each slot a verification rung** — priced by cost-of-being-wrong × cost-of-checking.
4. **Choose the dynamism rung** — the lowest one that lets discoveries steer, with a named signal and an integer budget.
5. **Apply the data-flow rule and the budget envelope** — code for certainty, JSON for code, prose for LLMs, caps on everything.

When quality disappoints after the build, there is a sixth move: the quality escalation ladder (§6). Structure first, model size last.

---

## 1. Decompose by cognitive jobs

Map how a domain expert actually works the problem — not how the data flows. Ask, concretely:

- What do they read **first**, and what do they deliberately ignore on the first pass?
- What do they **hold in mind** while working the rest? (Shared context a downstream slot will need — step 5.)
- What makes them **go deeper** on one part? (A dynamism signal — step 4.)
- When do they **stop**? (A budget or a coverage gate.)
- What do they **produce**, and who checks it? (The output contract and its verification — step 3.)
- Which of their moves are **mechanical** — sorting, counting, looking up, reconciling IDs? (Python, not a reasoner — step 5.)

Each distinct mental move becomes one reasoner: one job, 2–4 output fields, a one-sentence contract. Data-pipeline decomposition ("fetch → parse → analyze → report") produces stages; cognitive decomposition produces judgments ("triage severity", "scan for red flags", "check each flag against history", "decide disposition"). Only the second yields slots small enough to verify and independent enough to gather.

Worked micro-example — invoice intake for an accounting team, a problem no reference example covers (it threads through §2–§4 below so you can watch one derivation run end to end):

- The expert glances at vendor + amount first: "routine or unusual?" → one `.ai()` triage slot (`routine: bool`, `anomaly_kind: enum`, `confident: bool`).
- For unusual ones they pull the contract and check terms → a reasoner that reads the relevant sections. It runs only on signal — conditional deepening.
- They verify line-item math and PO references → pure Python: sums, ID lookups. Not a reasoner.
- Anything over $50k needs sign-off → a human gate on that path (rung 7, §3).
- They stop when disposition is decided → the entry reasoner's output contract.

Five minutes of this and the topology is mostly drawn — before any pattern name has entered the room.

---

## 2. The autonomy spectrum

Every intelligence slot sits somewhere between "typed function call" and "delegated engineer." Choosing the point is choosing how much process visibility you give up — and the verification price rises to match, because you can no longer inspect the steps, only the outcome. This is the competence-predictability inversion made operational: the more capable the delegate, the less you control HOW and the more you must verify WHAT.

| Point | Primitive | Process visibility | Verification price |
|---|---|---|---|
| Typed function call | `app.ai()` | One shot, transparent | Cheap — schema validates instantly |
| Manager | `@app.reasoner()` calling reasoners | Ordinary Python you wrote; every sub-call in the trace | Per sub-call, inspectable |
| Delegated engineer | `app.harness()` | Multi-turn, tools, opaque | Heavy — outcome-only, at the membrane |

Rules that follow:

- Pick the **leftmost** point that does the slot's job. Autonomy you don't need is verification debt you still pay.
- Moving right without adding verification is the real failure mode — not the autonomy itself.
- For a harness, the membrane is the whole contract: budget in (`max_budget_usd`, `max_turns`), tool surface, schema out. A goal and constraints, never a step list — you cannot audit steps you can't see, so don't pretend to specify them.

The invoice example, placed: triage is a typed function call (`app.ai()`, transparent, instantly checkable). The contract-terms check is a manager — a reasoner that pulls the relevant sections, asks 2–3 sub-questions, and assembles a judgment. Nothing in this problem needs a delegated engineer, so no slot moves right of manager and no harness-grade verification bill is incurred.

---

## 3. The verification ladder

Every slot gets a verification strategy, chosen by two costs: **how expensive is a wrong answer downstream** × **how cheaply can it be checked**. The ladder, cheapest to heaviest:

| Rung | Strategy | Mechanics | When |
|---|---|---|---|
| 1 | Accept | Nothing beyond the call | Low stakes, downstream tolerant of noise |
| 2 | Schema / shape | Types, enums, ranges — free via Pydantic `schema=` | Always. This is the floor, not a choice |
| 3 | Programmatic invariants | Code checks: sums reconcile, references resolve, cited passages exist in the source, the diff compiles / tests pass | Whenever an invariant is expressible in code — the cheapest real verification there is |
| 4 | Self-report + escalate | `confident: bool` in the schema; call site falls back to a deeper slot | A cheap slot may face inputs beyond its assumptions |
| 5 | Independent re-derivation | A second slot, different framing or model, answers the same question; compare programmatically or via a judge | Wrong answers costly, no code-checkable invariant exists |
| 6 | Adversarial refutation | A slot whose only job is to break the claim | False positives expensive AND one cognitive frame would rationalize its own guesses |
| 7 | Human gate | Pause the workflow for approval | Irreversible or accountability-bearing actions |

The two costs, crossed:

| | Cheap to check | Expensive to check |
|---|---|---|
| **Cheap to be wrong** | Rung 2–3 (it's nearly free — take it) | Rung 1–2 (don't pay for checks the stakes don't need) |
| **Expensive to be wrong** | Rung 3 (a code invariant is the jackpot cell) | Rung 4–7 — climb by severity: recoverable → 4/5, confidence-prone → 6, irreversible → 7 |

Choosing rules:

- **Pick the lowest rung the stakes allow.** Rung 6 on a formatting step is waste (~2× LLM cost per slot); rung 1 on a money-moving step ships a confidently-wrong answer into the world.
- **Rung 3 beats rungs 5–6 whenever it's available.** A citation checker in Python outranks a second LLM opinion and costs nothing per run. Look for an invariant before reaching for another model.
- **Verification lives at the membrane.** For a harness you cannot check the process, so check the outcome at the highest rung the stakes demand: run the tests (rung 3) when the output is code; a cheap `.ai()` judge or adversary (rung 5/6) when it's analysis.
- **Different paths can sit on different rungs.** Routine invoice → rung 2; the $50k anomaly → rung 7. Price each path, not the system.

The invoice example, priced: line-item math is rung 3 (Python reconciles the sums — free and exact). Triage is rung 4 (`confident: bool`; unconfident → escalate to the contract-terms reasoner). The contract-terms judgment on a large anomaly is rung 5 (a second slot re-derives the assessment from the contract alone; disagreements surface for review). The $50k payment action is rung 7. No slot needs rung 6 — there's no adversary-shaped failure mode here, and adding one would just double the bill.

The named things this skill already mandates are instances of this ladder, not extra rules: the mandatory `confident` flag + fallback is rung 4. HUNT→PROVE is rung 6 instantiated for security findings. Approval gates on irreversible actions are rung 7. When a design "always adds an adversary," it is paying rung-6 prices everywhere — refuse; read the stakes instead.

---

## 4. The dynamism ladder

Just-in-time orchestration is a decision, not an aesthetic. Topology can be, from least to most dynamic:

| Rung | Topology | The signal that justifies it | Budget it must carry |
|---|---|---|---|
| 1 | Fixed sequence | None needed | — |
| 2 | Conditional branches | A classification decides which subgraph runs | Enumerated branches |
| 3 | Runtime fan-out width | The data decides N (sections found, dimensions detected) | Max N |
| 4 | Meta-prompted children | A discovery decides *what to investigate and how to frame it* — the parent writes the child's prompt | Max spawns per parent |
| 5 | Recursive self-similar | Nested structure of unknown depth; the same judgment applies at each level | Depth cap |
| 6 | Self-modifying across runs | The system revises its own configuration between runs (rollouts, learned routing) | Rollout gate + rollback |

The rule: **choose the lowest rung that lets discoveries determine the path where they genuinely do.** Every rung above 1 must name the concrete signal that justifies it ("the number of exhibits is unknown until anatomy runs" → rung 3) and carry an integer budget. A rung chosen without a signal is ceremony; a real signal ignored by staying at rung 1 is a static chain that silently misses what it wasn't shaped to see.

"The DAG is a trace, not a spec" is the *consequence* of rungs 3–6: once the data decides the width, the prompts, or the depth, no pre-declared graph can express the system — which is exactly when this runtime earns its place over declare-the-graph frameworks. Reasoners are async functions calling each other through `app.call()`; control flow is ordinary Python (`if`/`else`, loops, `asyncio.gather`, recursion), so every rung is reachable without any framework construct. If the whole design honestly sits at rung 1–2, say so to the user; it may still be worth building (observability, VCs, model routing), but don't fake dynamism.

The invoice example, again: the signal is "routine or unusual?" — a classification that decides whether the contract-terms subgraph runs at all. That's rung 2. If one invoice can dispute clauses across several contracts and the count is unknown until triage, the terms check becomes a rung-3 fan-out (budget: max contracts pulled). Nothing here justifies meta-prompting or recursion, so the design stops climbing — and that restraint is itself a design decision the trace will show.

Rung 5 sketch — the signal and the budget are both explicit:

```python
@app.reasoner()
async def drill(section: str, depth: int = 0, model: str | None = None) -> dict:
    finding = await app.ai(system=SYS, user=section, schema=Finding, model=model)
    if finding.has_substructure and depth < MAX_DEPTH:      # signal + budget
        subs = await asyncio.gather(*[
            app.call(f"{app.node_id}.drill", section=s, depth=depth + 1, model=model)
            for s in finding.subsections
        ])
    ...
```

---

## 5. Code for certainty, intelligence for judgment

The value of an LLM slot is judgment — discovery, synthesis, the call a human expert would make. Everything else is code:

- Deterministic work (scoring formulas, sorting, dedup, thresholds, format conversion) is Python between reasoner calls. If you can write the function, write the function. `sorted(items, key=...)` beats `app.ai("sort these")` on cost, latency, and correctness every time.
- Rung-3 verification is this principle applied to checking: an invariant in code outranks an opinion from a model.

The data-flow rule (archei): format follows the consumer.

- Code branches on it → structured JSON (`if result.risk == "critical"`).
- Another LLM reads it → prose (`render_findings_as_text(findings)`). LLMs reason over language, not serialized dicts.
- Both → hybrid: enums and scores as fields, reasoning as a string field.

Parsing an LLM's prose with regex means that field should have been JSON. Feeding `str(model_dump())` to an LLM means that field should have been prose.

Budgets are the safety envelope around everything above. Steps 2–4 all create places where the system decides its own workload at runtime; budgets are what make that safe to ship. Every loop, spawn, recursion, and harness carries an explicit integer cap: `for _ in range(MAX)`, `max_budget_usd`, `max_turns`, depth caps, max spawns.

---

## 6. The quality escalation ladder

When output quality disappoints, escalate structure in this order — each move is cheaper and more diagnosable than the one after it:

1. **Sharpen the slot's contract.** Prompt precision, context fidelity — is the slot receiving exactly what it needs and nothing else?
2. **Decompose further.** A disappointing slot is usually doing two jobs; split it and see which half fails. Errors localize to one inspectable slot in the trace instead of dissolving into a giant completion.
3. **Add parallel perspectives.** Same question, different framings, merged programmatically or by a judge — rung 5 used generatively.
4. **Add adversarial verification.** A refuter (rung 6) when the failure mode is confident-wrong output.
5. **Escalate the slot's autonomy.** `.ai()` → tool-using `.ai()` → harness — only now, and pay the heavier membrane verification that added autonomy demands (§2).

Buy quality with structure before buying it with model size or tokens. Upgrading the model on an undiagnosed slot is spending money to avoid thinking. One LLM call reasons at ~0.3–0.4 on a normalized scale; it is deliberate composition — not a bigger completion — that pushes system-level reasoning to 0.7–0.8 for a specific domain.

---

## The derivation, end to end

1. Cognitive jobs → the reasoner set, and the Python between them.
2. Autonomy point per slot → primitives chosen, verification price accepted.
3. Verification rung per slot → stakes-priced checking, applied at the membrane.
4. Dynamism rung + named signal + budget → where the graph may decide its own shape.
5. Data-flow rule + budget envelope → the wiring.
6. (After the build) quality escalation ladder — structure before model size.

The honest exit: if step 1 yields one or two cognitive jobs, step 3 never rises above rung 2, and step 4 sits at rung 1 — the problem is one LLM call plus plumbing. Tell the user that instead of building a pretend mesh.

Only after steps 1–5 are done, open `patterns-emerge.md` and check whether the shape you derived has a name. Names help humans review the design; they are not design inputs.
