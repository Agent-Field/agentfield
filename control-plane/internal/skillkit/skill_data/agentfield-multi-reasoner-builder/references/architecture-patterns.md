# Architecture Patterns — The 8 AgentField Compositions

These are battle-tested patterns from real AgentField systems (`sec-af`, `af-swe`, `contract-af`, `af-deep-research`, `reactive-atlas`). Pick one, compose two, or invent your own — but never default to a static linear chain.

For each pattern: when to use it, the shape, and a real-system reference.

---

## 1. Parallel Hunters + Signal Cascade

**Shape:**
```
input ──┬──> hunter_A ──┐
        ├──> hunter_B ──┼──> findings_pool ──> downstream
        ├──> hunter_C ──┘
        └──> hunter_D
```

**When:** Any problem with multiple independent analysis dimensions that can be examined concurrently. Each hunter is a specialist that knows about ONE dimension deeply.

**Reference:** `examples/sec-af/` — parallel strategy hunters analyzing SEC filings; `examples/contract-af/` — parallel clause analysts (IP / liability / non-compete / data / termination).

**Code shape:**
```python
@app.reasoner()
async def review(document: str) -> dict:
    findings = await asyncio.gather(*[
        app.call(f"{app.node_id}.{h}", document=document)
        for h in ["profitability_hunter", "liquidity_hunter", "risk_hunter", "efficiency_hunter"]
    ])
    return await app.call(f"{app.node_id}.synthesizer", findings=findings)
```

**Common mistake:** Making the hunters do "everything" each. Each hunter is a NARROW specialist. If hunters overlap heavily, you decomposed wrong.

---

## 2. HUNT → PROVE Adversarial Tension

**Shape:**
```
input ──> hunters ──> candidate findings ──> provers ──> verified findings
                                           ↑
                                           structurally separated verification frame
```

**When:** When applying principle 2 (guided autonomy) reveals that the *discovery* frame and the *verification* frame must be structurally separate, because a single cognitive frame confuses finding answers with justifying them. Discovery reasoners are biased toward sensitivity — they should find everything plausible. Verification reasoners are biased toward specificity — they should refuse anything unproven. Keeping these frames in one head produces reasoners that rationalize their own initial guesses instead of stress-testing them.

**Why it works:** The tension between discovery and verification is where the intelligence lives. Neither frame alone produces a good answer. The pattern is most valuable when your problem requires verifiable confidence in a final decision and the cost of a confident-but-wrong output is meaningful.

```python
@app.reasoner()
async def adversarial_review(input: str) -> dict:
    candidates = await app.call(f"{app.node_id}.hunter_pool", input=input)
    verified = await asyncio.gather(*[
        app.call(f"{app.node_id}.prover", finding=f, original=input)
        for f in candidates
    ])
    return [v for v in verified if v["proven"]]
```

---

## 3. Streaming Pipeline (asyncio.Queue)

**Shape:**
```
upstream ──emits──> queue ──consumes──> downstream
                                          (starts working before upstream finishes)
```

**When:** Downstream reasoners can start working on partial results — and waiting for the full upstream batch wastes time and misses interaction effects.

**Reference:** `examples/sec-af/` — HUNT→PROVE streaming; `examples/contract-af/` — analysts → cross-reference + adversary streaming.

```python
findings_queue = asyncio.Queue()

async def producer(items):
    for item in items:
        finding = await app.call(f"{app.node_id}.analyze", item=item)
        await findings_queue.put(finding)
    await findings_queue.put(None)  # sentinel

async def consumer():
    seen = []
    while (f := await findings_queue.get()) is not None:
        # Check this finding against everything seen so far
        await app.call(f"{app.node_id}.cross_ref", new=f, prior=seen)
        seen.append(f)
```

---

## 4. Meta-Prompting (Harnesses Spawning Harnesses)

**Shape:**
```
parent_harness ──discovers X──> crafts a SPECIFIC prompt ──spawns──> child_harness ──> findings
       ↑                                                                                    │
       └────────────────── integrates findings ─────────────────────────────────────────────┘
```

**When:** The investigation path depends on what gets discovered. You cannot pre-define which sub-reasoners will run, because you don't know yet what's there.

**Reference:** `examples/contract-af/` — clause analysts spawning definition-impact analyzers when they discover a referenced defined term; cross-reference resolver spawning combination deep-dives.

**This is the pattern that no framework chain can replicate.** It's pure dynamic intelligence.

```python
@app.reasoner()
async def clause_analyst(clause: str, context: str) -> dict:
    initial = await app.harness(
        goal=f"Analyze this clause: {clause}",
        tools=["read_section", "lookup_definition"],
        max_iterations=10,
    )

    # The harness discovered a defined term that needs deeper analysis.
    # Craft a SPECIFIC prompt for a child harness at runtime.
    if initial.discovered_terms:
        for term in initial.discovered_terms:
            sub_prompt = (
                f"You are analyzing the cascading impact of the defined term '{term}' "
                f"in the context of clause: {clause}. "
                f"Read every section that references '{term}' and determine if any "
                f"interaction creates risk. Return: affected_sections, risk_level, rationale."
            )
            sub = await app.call(
                f"{app.node_id}.term_impact_analyzer",
                prompt=sub_prompt,
                term=term,
            )
            initial.term_impacts.append(sub)
    return initial.model_dump()
```

**Hard rule:** every meta-spawn point has a depth cap.

---

## 5. Three Nested Control Loops (Inner / Middle / Outer)

**Shape:**

| Loop | Scope | Trigger | Cap |
|---|---|---|---|
| **Inner** | Per-reasoner self-adaptation | Found a reference, escalation needed | `max_follows=3`, `max_escalations=1` |
| **Middle** | Cross-reasoner deep-dives | Critical combination, hidden interaction | `max_spawns=5` |
| **Outer** | Pipeline-wide coverage | Coverage gate detects a gap | `max_iterations=3` |

**When:** Long-running analysis where you can't predict upfront how deep you need to go. Coverage matters and edge cases are dangerous.

**Reference:** `examples/af-swe/` — inner coding loop / middle sprint loop / outer factory loop; `examples/contract-af/` — analyst loop / cross-ref loop / coverage loop.

**Hard rule:** every loop has an absolute cap. "Keep going until confident" is how you get a $400 bug report.

---

## 6. Fan-Out → Filter → Gap-Find → Recurse

**Shape:**
```
seed ──> [generate N candidates] ──> [filter to top K] ──> [gap analysis]
                                                                │
                                                                ├─ gaps found ──> recurse with new seeds
                                                                └─ no gaps    ──> done
```

**When:** Comprehensive coverage problems where you don't know the shape of the answer upfront — research, due diligence, audits, literature reviews.

**Reference:** `examples/af-deep-research/` — recursive research with quality-driven loops.

```python
@app.reasoner()
async def deep_research(question: str, max_rounds: int = 3) -> dict:
    seeds = [question]
    all_findings = []
    for round in range(max_rounds):
        findings = await asyncio.gather(*[
            app.call(f"{app.node_id}.investigator", seed=s) for s in seeds
        ])
        all_findings.extend(findings)
        gaps = await app.call(f"{app.node_id}.gap_finder", findings=all_findings, original=question)
        if not gaps.gaps:
            break
        seeds = gaps.gaps  # next round's seeds
    return await app.call(f"{app.node_id}.synthesizer", findings=all_findings)
```

---

## 7. Factory Control Loops

**Shape:** Three nested loops for long-running multi-step execution with adaptive replanning.

```
outer (factory)  ──> sprint planner   ──> goals
middle (sprint)  ──> task executor    ──> tasks
inner (coding)   ──> per-task agent   ──> code
                              │
                              └─ fails ──> outer replan
```

**When:** Multi-step execution that needs to replan based on intermediate results — code generation, document production, migration execution, multi-step research.

**Reference:** `examples/af-swe/`.

---

## 8. Reasoner Composition Cascade (the structural backbone — layer the other patterns on top)

**This is not a specific shape — it is the discipline of decomposing every reasoner into narrower sub-reasoners until the DAG has depth ≥ 3 and every reasoner reads like a composable software API.** Every other pattern in this file is a specific topology you layer on top of this discipline.

The cascade can manifest as a linear chain, a star with depth, a tree, a dynamic router, or anything else. What makes it a "cascade" is that the orchestrator is NOT the only thing calling `app.call` — every dimension reasoner is itself a small orchestrator that calls 2–4 sub-reasoners. Depth 3+ at every branch.

**Example shape — adversarial committee (only when false positives are expensive, see pattern 2):**

```
entry_reasoner
├── classifier_reasoner ─────────────────┐
│   ├── input_normalizer (skill)         │
│   └── intent_extractor (.ai)           │
│       └── slot_filler (.ai called by intent_extractor when ambiguous)
│
├── analysis_dimension_A_reasoner ────── ┤  ← all parallel via asyncio.gather
│   ├── deterministic_metric_calc (skill)
│   ├── pattern_judge (.ai)
│   │   └── citation_finder (.ai called by pattern_judge)
│   └── confidence_scorer (.ai)
│
├── analysis_dimension_B_reasoner ────── │
│   ├── different_metric_calc (skill)
│   ├── different_pattern_judge (.ai)
│   └── confidence_scorer (.ai REUSED — same reasoner)  ← reuse across branches!
│
├── analysis_dimension_C_reasoner ───────┤
│   └── (3 sub-calls, similar shape)
│
└── adversarial_synthesizer ─────────────┘  ← OPTIONAL: only earns its cost
    ├── steel_man_alternative (.ai)           when false positives are
    ├── disagreement_detector (.ai)           expensive (medical / legal /
    └── final_decision_reasoner (.ai)         financial / security / regulated)
        └── safety_override (deterministic skill)
```

**Example shape — linear refinement cascade (content, extraction, generation — no adversary needed):**

```
summarize_meeting (entry)
└── audio_transcriber
    └── speaker_diarizer
        └── turn_segmenter
            ├── action_item_extractor   ─┐  ← parallel
            └── decision_extractor       ┘
                └── summary_composer
                    └── follow_up_drafter
```

Same cascade discipline, totally different shape. Depth 6, linear with one parallelism wave. Perfect for content pipelines. No adversarial anything — just careful decomposition.

**Example shape — dynamic router cascade (classification, routing, workflow):**

```
handle_support_ticket (entry)
├── intake_classifier ──┐
│   ├── language_detector
│   ├── intent_extractor
│   └── urgency_grader
│
└── (conditional branches based on intake — only one fires per request)
    ├── refund_handler       → order_lookup → eligibility_checker → refund_executor (skill)
    ├── diagnosis_cascade    → symptom_classifier → known_issue_matcher → workaround_composer
    ├── escalation_composer  → priority_scorer → owner_resolver
    └── knowledge_base_search → semantic_query → answer_drafter
```

Same cascade discipline, third totally different shape. Depth 3–4 depending on branch. No parallelism across branches (mutually exclusive). No adversarial anything. Perfect for customer support, incident triage, intent-based routing.

**Each layer fans out via `asyncio.gather` (where parallelism is meaningful). Each reasoner has a single cognitive responsibility.** The orchestrator at the top is NOT the only thing that calls `app.call` — every dimension reasoner is itself a small orchestrator that calls 2–4 sub-reasoners.

**Used in:** Every serious AgentField system. The shape varies by problem — there is no canonical "correct" topology, only canonical decomposition discipline. The loan-underwriter and medical-triage examples happen to use the adversarial committee shape because their problems genuinely need an adversary. A web research agent would use the linear/fan-out shape and NOT include an adversary.

**Why it's the master pattern:**

1. **Reasoners as software APIs.** Each reasoner has a one-line API contract: *"Given X, return Y. Calls Z, W."* Other reasoners call it the way one microservice calls another.
2. **Composability over monolithic prompts.** A specialist reasoner like `pe_assessor` is NOT a 200-line `.ai()` prompt — it's an orchestrator that calls `wells_score_calculator`, `dyspnea_grader`, and `dvt_history_checker` and synthesizes their outputs. Each piece is testable, replaceable, reusable.
3. **Reuse across branches.** `confidence_scorer` is called from THREE different dimension reasoners. The flat-star pattern would have to copy-paste the logic three times. The composition cascade calls it once per branch — same code, three different contexts.
4. **Multi-layer parallelism.** `asyncio.gather` runs at the entry-reasoner layer (across dimensions A/B/C) AND inside each dimension reasoner (across its sub-calls). Total wall-clock time is dominated by the slowest path through the DAG, not by the sum.
5. **Observability has structure.** The control plane workflow DAG shows the actual call tree. The verifiable credential chain has hierarchy. A future debugger can ask "which sub-call inside `pe_assessor` flagged the concern" — the flat-star pattern can only tell you "pe_assessor returned X."
6. **Each reasoner is independently curl-able.** You can `POST /api/v1/execute/<slug>.wells_score_calculator` directly with synthetic input to debug or A/B test it. The flat-star only exposes the entry reasoner.

**Decomposition rules:**

- **30-line ceiling.** If a reasoner body is > 30 lines, it's probably 2 reasoners. Look for the seam — usually a "compute X then judge Y" boundary becomes "X is a `@app.skill`, Y is a `@app.reasoner` that calls X".
- **Single-judgment rule.** A reasoner makes ONE judgment call. If your reasoner is making three judgments ("is this concerning, is this acute, what's the risk score"), split into three reasoners.
- **Deterministic-vs-judgment split.** Anything that doesn't require LLM judgment (math, formula, regex, lookup, sort) is `@app.skill()` or a plain helper, not part of an `.ai()` reasoner body.
- **Reuse signal.** If the same logic appears in 2+ reasoners, extract it as its own reasoner and call it from both.
- **One-sentence API contract test.** Can you write a one-sentence contract for each reasoner ("Given a chief complaint string, return a list of red flag categories with confidence scores")? If not, the reasoner is doing too many things.

**Anti-patterns that mean you fell back to a flat star:**

- Your entry reasoner is the ONLY thing that calls `app.call`
- Your specialists each have a single fat `.ai()` call with a 500-token prompt
- Your DAG is depth 2 (`entry → specialists → done`)
- You can draw the architecture as a literal asterisk
- Two specialists have the same 50-line prompt with one line different — you should have had one parameterized sub-reasoner

**Concrete medical-triage example:**

A flat-star `red_flag_detector` reasoner with one big `.ai()` prompt → bad.

A `red_flag_detector` reasoner that calls `cardiac_red_flag_checker`, `stroke_red_flag_checker`, `bleeding_red_flag_checker`, `psych_red_flag_checker` in parallel via `asyncio.gather`, each of which is itself a focused `.ai()` with its own narrow prompt and confidence flag → good. The deeper structure means a future agent can swap the cardiac checker for one with a more accurate prompt without touching anything else.

**When you finish your design, count the depth.** If max depth from entry to leaf is < 3, redesign. A real composite-intelligence system has at least 3 layers of reasoner-calling-reasoner.

---

## 9. Reactive Document Enrichment

**Shape:**
```
event source (DB change stream / webhook) ──> enrichment pipeline ──> output
```

**When:** Work is triggered by data arriving — incidents, PRs, contracts on upload, form submissions, telemetry events.

**Reference:** `examples/reactive-atlas/` — MongoDB change streams → enrichment agents.

**The point:** the engine is domain-agnostic; the config defines the domain. The same pattern handles "new contract uploaded → enrich → score → route" as it handles "new incident filed → triage → assign → notify".

---

## How to think about these patterns

**These patterns are not a menu.** They are **names for emergent consequences of the five principles** in `SKILL.md` ("The unit of intelligence is the reasoner"). You do not pick a pattern and then build the system — you apply the principles to your specific problem, the topology emerges, and THEN you use these pattern names as vocabulary to describe what you built.

If you find yourself reaching for a pattern before you have walked the five principles, stop. Go back to the principles and let the shape fall out of the problem. The shape of the DAG is **always** derived from:

1. How your problem decomposes into atomic reasoning units (principle 1).
2. Where each unit's scope ends and the next begins, with the orchestrator as context broker (principles 2 + 4).
3. Where the graph's structure needs to change based on intermediate state (principle 3).
4. Which units depend on which others, and therefore which can run concurrently (principle 5).

The patterns below are **what those answers look like when they are drawn on paper**. Every problem produces its own specific shape — the same principles produce different topologies for different problems, and that is the point.

### When specific patterns tend to emerge

Apply the five principles honestly. The pattern you end up with will usually match one or a composition of the below. Use these as vocabulary, not as templates.

- **Parallel Hunters + Signal Cascade (pattern 1)** emerges when principle 1 (decomposition) identifies multiple independent analysis dimensions and principle 5 (async parallelism) runs them concurrently. The "hunters" are the independent specialists; the "signal cascade" is how their outputs feed downstream consumers.
- **HUNT→PROVE Adversarial Tension (pattern 2)** emerges when principle 2 (guided autonomy) forces you to separate the *discovery* frame from the *verification* frame, because getting them wrong in the same head produces biased reasoning. One cognitive frame finds candidate answers; a structurally different cognitive frame tries to falsify them. Consider this pattern when you need verifiable confidence and a single frame would confuse discovery with justification.
- **Streaming Pipeline (pattern 3)** emerges when principle 5 (async parallelism) identifies that downstream reasoners can start working on partial upstream results without waiting for the whole batch. The asyncio.Queue is the connective tissue.
- **Meta-Prompting (pattern 4)** emerges when principle 3 (dynamic orchestration) pushes the "which sub-reasoner to invoke and with what prompt" decision into the LLM itself, because the investigation path depends on what a prior reasoner discovered. This is the purest expression of the meta-level: a reasoner whose output IS the structure of the next subsystem.
- **Three Nested Control Loops (pattern 5)** emerges when principle 3 (dynamic orchestration) needs to happen at multiple scopes simultaneously — per-reasoner self-adaptation, cross-reasoner deep-dives, and pipeline-wide coverage iteration.
- **Fan-Out → Filter → Gap-Find → Recurse (pattern 6)** emerges when principle 1 (decomposition) meets principle 3 (dynamic orchestration) in an open-ended problem: you do not know the shape of the answer upfront, so the graph grows iteratively until a coverage gate is satisfied.
- **Factory Control Loops (pattern 7)** emerges when principle 3 (dynamic orchestration) spans long-running multi-phase execution — the plan itself must adapt as earlier phases reveal information about later ones.
- **Reasoner Composition Cascade (pattern 8)** is not a specific shape at all. It is the structural discipline of applying principle 1 (decomposition) recursively until every leaf is atomic and principle 5 (async parallelism) can fire at every layer. Every other pattern in this file is layered on top of this discipline.
- **Reactive Document Enrichment (pattern 9)** emerges when the trigger is not a direct API call but an event stream — any of the other patterns can live inside the enrichment chain, but the entry point changes from a reasoner invocation to a subscription.

### How to decide which emerge for your problem

Walk the five principles in order. Ask the questions from `SKILL.md`. The answers will naturally draw the graph for you:

- Where decomposition produces many independent sub-tasks at a given layer → some form of fan-out (patterns 1, 6).
- Where the verification frame needs to be structurally separated from the discovery frame → HUNT→PROVE (pattern 2).
- Where downstream work can begin on partial upstream output → streaming (pattern 3).
- Where the investigation path itself depends on intermediate state → meta-prompting (pattern 4).
- Where adaptation needs to happen at multiple scopes simultaneously → nested loops (pattern 5).
- Where the answer's shape is unknown upfront and a coverage gate drives iteration → fan-out / gap-find (pattern 6).
- Where long-running multi-phase execution must replan as it learns → factory loops (pattern 7).
- Where the trigger is an event stream rather than an API call → reactive enrichment (pattern 9).

**If none of these emerge for your problem, the problem may not need a multi-reasoner system at all — it may just be a sequence of deterministic steps dressed up as an agent.** If that is true, say so honestly. The value of AgentField is in problems where the architecture itself encodes intelligence.

## When NONE of these fit

If after walking the five principles no pattern emerges and the problem can be solved by a deterministic pipeline with one or two LLM calls, **tell the user honestly**. The goal is composite intelligence — if the problem does not demand composition, we should not force it.
