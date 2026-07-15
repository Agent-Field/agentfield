# Patterns Emerge — Post-Hoc Naming, Never Design Inputs

Read this file **after** your topology has emerged from the derivation procedure in `mental-models.md` — never before. Its job is to check whether the shape you derived has a name. Names help humans review a design ("this is parallel hunters feeding an adversary"); they are not inputs to design. Each entry below therefore carries a **derivation line** showing which combination of the ladders produces it — proof that the pattern is an output of the procedure, not an axiom.

The topology is never declared anywhere — it is the trace your control flow leaves at runtime. These names describe traces, not specs. And the framing underneath them all: **reasoners are APIs.** A derived design looks like a service mesh where any reasoner can call any other at any depth, the call graph is shaped by intermediate state, and the depth from entry to leaf is ≥ 3. If your design ends up as one orchestrator fanning out to N siblings and that's it, you stopped decomposing too early — see "The composition cascade" below.

How to use this file:

1. Finish the derivation. You have a shape.
2. Skim below for the entry whose shape matches what you built.
3. Open the live example pointed at in `examples-map.md`, grep its real code, and steal its decomposition discipline (not its topology — yours was derived for your stakes and signals).
4. Use the name in your CLAUDE.md and handoff so future reviewers can orient fast.

There is no "preferred" pattern. HUNT→PROVE earns its ~2× cost when false positives are expensive; a linear refinement cascade earns its place when no signal justifies more. The derivation already told you which — the name just labels it.

---

## The composition cascade — the structural backbone of every build

**Not a specific shape. The discipline of decomposing every reasoner into 2–4 sub-reasoners until the DAG has depth ≥ 3 and every leaf is atomic.** Every other pattern in this file is a specific topology layered on top of this discipline.

**Derivation:** cognitive-job decomposition (step 1) applied recursively — every job that itself contains sub-judgments becomes an orchestrator over them.

What "composition cascade" looks like in code: the orchestrator at the top of the pipeline is NOT the only thing calling `app.call`. Every "specialist" reasoner is itself a small orchestrator that calls 2–4 sub-reasoners. Those sub-reasoners may themselves call further sub-reasoners. The call graph forms recursively, like a service mesh.

**Reuse signal.** When the same logic appears in two specialists (e.g., scoring + confidence calibration), extract it into its own reasoner and have both specialists `app.call` it. The flat-star pattern would copy-paste; the composition cascade calls a single shared sub-reasoner from multiple callers.

**Decomposition rules of thumb:**

- **30-line ceiling.** A reasoner body > 30 lines is probably 2 reasoners glued together.
- **Single-judgment rule.** A reasoner makes ONE judgment call. If yours makes three ("is it severe, is it acute, what's the risk score"), split into three.
- **Deterministic vs judgment split.** Anything that doesn't require LLM judgment (math, regex, lookup, sort) is `@app.skill()` or a plain helper — never inside an `app.ai()` body.
- **One-sentence API contract test.** Write the contract for each reasoner. *"Given a chief complaint string, return a list of red-flag categories with confidence scores."* If you can't, the reasoner is doing too many things.

**Anti-patterns that mean you fell back to a flat star:**

- The entry reasoner is the only thing that calls `app.call`.
- Specialists each have one fat `app.ai()` call with a 500-token prompt.
- DAG depth is 2 (`entry → specialists → done`).
- Two specialists have the same 50-line prompt with one line different — should have been one parameterized sub-reasoner.

**Reference:** `healthcare-agents/` — 29 reasoners organized in 3 tiers (atomic / composers / orchestrators), where orchestrators call composers which call atomic; `contract-af/` — committee + specialists each calling deeper sub-reasoners; `sec-af/` — analyst layer fans into prover layer with sub-reasoners on both sides.

---

## When each pattern emerges

The shape comes from running the derivation procedure. Here are the named consequences, the questions that produce them, and the ladder rungs each one instantiates.

### Parallel hunters + signal cascade

**Question:** "Are there N independent analysis dimensions of the same input?"

**Derivation:** step-1 decomposition along independent expert lenses, run under `asyncio.gather`. Dynamism rung 1 when the lens set is fixed; rung 3 when the data decides how many hunters run.

Each "hunter" is a narrow specialist. They run concurrently via `asyncio.gather`. Findings funnel into a downstream synthesizer or a cross-reference resolver.

**Reference:** `sec-af` (strategy hunters across security categories), `contract-af` (clause analysts per legal axis), `reel-af` (4 topic hunters: specific-figure / reversal / temporal / cross-domain).

**Common mistake:** hunters that overlap heavily. If two hunters could be merged without losing depth, you decomposed wrong.

### HUNT → PROVE adversarial tension

**Question:** "Is the cost of a confident-but-wrong final answer meaningful, AND does a single cognitive frame confuse discovery with verification?"

**Derivation:** verification rung 6 (adversarial refutation) applied to a parallel discovery layer — nothing more exotic than that.

Discovery reasoners find candidates (biased toward sensitivity). Verification reasoners try to falsify them (biased toward specificity). Keeping them in one head produces reasoners that rationalize their initial guesses.

**Reference:** `sec-af` (HUNT phase → PROVE phase), `contract-af` (analysts → adversarial reviewer).

**Cost note:** ~2× the LLM cost of a single-frame pass. Only earns it when false positives are expensive. **Do NOT default to this pattern** for routing, extraction, content, or research — those don't need an adversary.

### Streaming pipeline (asyncio.Queue)

**Question:** "Can downstream reasoners begin working on partial upstream results without waiting for the full batch?"

**Derivation:** not from the ladders at all — a pure latency optimization layerable on any multi-stage shape. It changes when work runs, never what the topology is.

Upstream agents emit findings into a queue; downstream consumes them as they arrive. Total wall-clock time is dominated by the slowest path through the pipeline, not by the sum of phases.

**Reference:** `sec-af` (HUNT→PROVE streaming), `contract-af` (analysts → cross-ref + adversary streaming).

### Meta-prompting (reasoners spawning reasoners with runtime prompts)

**Question:** "Does the investigation path depend on what gets discovered? Can I not pre-define which sub-reasoners will run?"

**Derivation:** dynamism rung 4 — the discovery is the signal, the spawn cap is the budget.

A parent reasoner discovers something (a defined term, a suspicious combination, a referenced section) and **crafts a specific prompt at runtime** for a child reasoner with the discovery encoded in the prompt. This is pure dynamic intelligence — no static chain framework can replicate it.

**Reference:** `contract-af` (clause analysts spawning definition-impact analyzers when a defined term shows up; cross-reference resolver spawning combination deep-dives).

**Hard rule:** every meta-spawn point has a depth cap.

### Three nested control loops (inner / middle / outer)

**Question:** "Does adaptation need to happen at multiple scopes simultaneously — per-reasoner self-adaptation, cross-reasoner deep-dives, and pipeline-wide coverage iteration?"

**Derivation:** dynamism rungs 2–4 stacked at three scopes, each with its own integer budget; the outer loop is the quality-escalation ladder applied to coverage.

Three loops with three caps. Inner adapts per call. Middle adapts across calls in one phase. Outer iterates the whole pipeline until a coverage gate passes.

**Reference:** `af-swe` (inner coding loop → middle sprint loop → outer factory loop), `contract-af` (analyst loop → cross-ref loop → coverage loop).

**Hard rule:** every loop has an absolute cap. "Keep going until confident" is a budget hole.

### Fan-out → filter → gap-find → recurse

**Question:** "Is the shape of the answer unknown upfront, and does a coverage gate drive iteration?"

**Derivation:** dynamism rung 5 (recursive, depth-capped) + a programmatic filter between rounds (code for certainty) + a gap-finder as the recursion signal.

Generate N candidates → filter to top K → ask a gap-finder if anything important is still missing → if so, recurse with new seeds. The graph grows until a quality threshold or a hard iteration cap.

**Reference:** `af-deep-research` (recursive research with quality-driven loops).

### Factory control loops (multi-phase execution with replanning)

**Question:** "Is this long-running multi-phase work where the plan itself must adapt as earlier phases reveal information about later ones?"

**Derivation:** dynamism rung 4 across phases (each phase's results write the next phase's instructions) + verification rung 3 at every membrane (the output is code, so tests and compilation are the check).

Plan → execute → re-plan based on results → execute. Each phase is itself a sub-pipeline.

**Reference:** `af-swe` (PM → Architect → TL → Sprint Planner → parallel coders → QA → reviewer → merger → verifier).

### Linear refinement cascade

**Question:** "Is this a content / extraction / generation pipeline where each stage strictly refines the previous, with no adversary needed?"

**Derivation:** dynamism rungs 1–2 + verification rungs 2–3 — and that's correct, because no signal justifies more. Honest low rungs beat decorated high ones.

Sequential cascade with parallelism waves where independent sub-tasks appear. No HUNT/PROVE. No coverage gate. Just careful decomposition with depth ≥ 3.

**Reference:** `reel-af` article path (URL → essence → script → audio ∥ visuals ∥ accents ∥ beats → videos → stitch), `roboscribe-af` (multi-pass annotation refinement).

### Dynamic router cascade

**Question:** "Is the input classified into mutually-exclusive categories that each trigger a different downstream subgraph?"

**Derivation:** dynamism rung 2 (the classification is the signal, the branch set is the budget) + verification rung 4 on the classifier itself — a misroute poisons everything downstream.

Intake classifier routes to one of several conditional branches. Each branch is its own composition cascade.

**Reference:** Standard customer-support / triage / intent-routing shapes. Not yet a single example dedicated to this, but the pattern lives inside `contract-af`'s intake (which picks specialists) and inside `reactive-atlas` (which routes events to domain configs).

### Reactive document enrichment (event-driven)

**Question:** "Is the work triggered by data arriving rather than by an explicit user request?"

**Derivation:** any of the above shapes with a trigger as the entry surface — the arrival mechanism changes, the derivation doesn't.

The entry surface is a trigger (`@on_event`, `@on_schedule`) rather than a curl. The triggered reasoner is thin — it routes and fans out. The actual reasoning lives in `@app.reasoner` specialists downstream.

**Reference:** `reactive-atlas` (MongoDB change streams → enrichment agents; engine is domain-agnostic, config defines the domain).

See `triggers.md` before declaring any `@on_event` / `@on_schedule` / `triggers=[...]`.

---

## How names emerge

You don't pick. You run the derivation, and afterwards the shape you drew usually has a name. This table maps derivation observations to the name your trace will likely carry:

| What the derivation surfaced | Name your shape probably has |
|---|---|
| N independent analysis dimensions | Parallel hunters + signal cascade |
| Verification frame must be separate from discovery frame, false-positives expensive | HUNT → PROVE |
| Downstream can start before upstream finishes | Streaming pipeline |
| Investigation path depends on intermediate findings | Meta-prompting |
| Adaptation needed at multiple scopes simultaneously | Nested control loops |
| Answer shape unknown upfront, coverage gate drives iteration | Fan-out → filter → gap-find → recurse |
| Long-running multi-phase execution that must replan | Factory loops |
| Strict refinement, no adversary needed | Linear refinement cascade |
| Classification gates into mutually-exclusive branches | Dynamic router cascade |
| Triggered by external events, not direct calls | Reactive enrichment + triggers |

For all of the above, the **composition cascade** discipline applies. Every layer must have depth, parallelism where work is independent, and a mix of orchestrators and specialists.

---

## If none of these fit

Good — that's the procedure working, not failing. A derived topology with no name is still a correct topology; describe it in your CLAUDE.md in ladder terms ("rung-3 fan-out feeding a rung-5 cross-check") and move on. Names are conveniences, not requirements.

And if the derivation yields a deterministic pipeline with one or two LLM calls, **tell the user honestly.** AgentField earns its place when the architecture itself encodes intelligence. If your problem is "one LLM call + some plumbing", build it as one LLM call + some plumbing — don't force a pretend mesh on top.
