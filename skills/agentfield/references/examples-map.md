# Examples Map — Live Builds You Can Grep

These projects live in `code/examples/` of this repo. Each is a real, runnable AgentField multi-agent system. **Pick the closest analog to your problem and grep its code.** Pages describe the API; examples show real composition.

For each entry: domain, dominant shape, entry reasoner path, and one sentence on what makes it interesting.

---

## Composition-cascade flagships

### `healthcare-agents/`
- **Domain:** Clinical document analysis.
- **Shape:** 3-tier composition — **29 reasoners, 11 orchestrators**.
- **Entry:** `reasoners/orchestrators.py` (multiple, e.g., `analyze_patient`).
- **Why interesting:** the cleanest "atomic → composers → orchestrators" hierarchy. Orchestrators call composers, composers call atomic reasoners, atomic reasoners do one cognitive thing each. Reuse across orchestrators is heavy. Read this when your problem is "decompose one domain into many small judgments and assemble them."

### `contract-af/`
- **Domain:** Legal contract risk analysis.
- **Shape:** Parallel specialists + HUNT→PROVE committee + meta-prompting (clause analysts spawn definition-impact analyzers when they discover a defined term).
- **Entry:** `src/contract_af/...` (committee + specialists routers).
- **Why interesting:** the canonical example of pattern composition — adversarial committee on top of parallel hunters on top of meta-prompted sub-reasoners. Read this when false positives are expensive AND the investigation path depends on what's discovered.

### `sec-af/`
- **Domain:** Security audit (proves exploitability, not just patterns).
- **Shape:** RECON → HUNT (parallel strategy hunters) → DEDUP → PROVE (adversarial 4-agent chain) → REMEDIATE. Streaming between HUNT and PROVE.
- **Entry:** `src/sec_af/app.py`.
- **Why interesting:** the original HUNT→PROVE reference. Also a strong streaming-pipeline reference. Read this when verifying findings is more expensive than discovering them.

---

## Linear refinement / content-pipeline

### `reel-af/`
- **Domain:** URL or topic → vertical viral reel.
- **Shape:** Two entry reasoners. Article path is a deep linear cascade. Topic path is a multi-reasoner hunter cascade — 4 parallel hunters (specific-figure / reversal / temporal / cross-domain) → critic picks top 3 → 3 parallel narrators → pairwise judge picks the winner.
- **Entry:** `src/reel_af/app.py` — `reel_article_to_reel`, `reel_topic_to_reel`.
- **Why interesting:** shows two completely different topologies sharing the same downstream specialists. The topic path is a great "fan-out → critic → fan-out → judge" composition. Read this when your problem has multiple input paths into the same downstream work.

### `roboscribe-af/`
- **Domain:** Multi-pass annotation for robotics demonstration data.
- **Shape:** Multi-pass refinement cascade. Hierarchical sub-task annotation. Reads multimodal data (videos + actions + language).
- **Entry:** `src/...`.
- **Why interesting:** strongest example of decomposition over multimodal data. Read this when refining hierarchical annotations or doing multi-pass labeling.

### `podcast/`
- **Domain:** Generates podcast-style audio + video from a prompt.
- **Shape:** Small linear pipeline.
- **Entry:** `main.py` (one file).
- **Why interesting:** smallest example that still demonstrates `app.ai` with multimodal output. Read this when you want a minimal scaffold to learn from.

---

## Fan-out / recursive research

### `af-deep-research/`
- **Domain:** Autonomous research backend.
- **Shape:** Fan-out → filter → gap-find → recurse. Quality-driven loops with a coverage gate.
- **Entry:** `main.py` / `doc_generation_pipeline.py`.
- **Why interesting:** the canonical recursive-research pattern. Read this when comprehensive coverage matters and you don't know the shape of the answer upfront.

---

## Multi-step execution with replanning

### `af-swe/`
- **Domain:** Autonomous engineering team (one API call ships code end-to-end).
- **Shape:** Factory control loops — PM → Architect → TL → Sprint Planner → parallel coders (isolated worktrees) → QA → Reviewer → Merger → Verifier.
- **Entry:** see `docs/` for the entry reasoner.
- **Why interesting:** the deepest multi-phase execution pipeline. Each phase is a sub-pipeline that can replan based on earlier outputs. Read this when building anything that produces code, multi-phase reports, or long-running multi-step deliverables.

### `codekeep-af/`
- **Domain:** Autonomous code maintenance — drift detection, simplification, test gap analysis.
- **Shape:** Multiple entry reasoners, each fanning into specialists. Uses `app.harness` (claude-code provider).
- **Entry:** `src/codekeep_af/app.py`.
- **Why interesting:** real `app.harness` usage. Read this when your build genuinely needs a coding agent (file I/O, shell access). Note the Dockerfile pattern that installs the harness CLI.

### `pr-af/`
- **Domain:** Autonomous PR review (security, performance, test coverage, breaking changes).
- **Shape:** Parallel hunters + adversarial prover per finding + structured report writer.
- **Entry:** `src/...`.
- **Why interesting:** similar shape to sec-af, applied to PR review. Tighter feedback loop because PRs are smaller.

---

## Event-driven / reactive

### `reactive-atlas/`
- **Domain:** MongoDB collection → AI-enriched intelligence layer (no rule engines, no polling).
- **Shape:** Event triggers (Atlas change streams) → enrichment pipeline per domain → write back. The engine is domain-agnostic; YAML config defines the domain.
- **Entry:** `main.py` + `domains/<domain>.yaml`.
- **Why interesting:** the canonical event-driven pattern. The "config-as-product" approach is reusable. Read this when work is triggered by data arriving (incidents, PRs, contracts on upload, telemetry).

### `cloud-logs-af/`
- **Domain:** Cloud log analysis.
- **Shape:** Small, focused.
- **Entry:** `main.py`.
- **Why interesting:** great minimal scaffold of "agent that consumes logs and emits findings". Good to copy when you want a clean starting point.

### `cloudsecurity-af/`
- **Domain:** AI-native cloud infrastructure security scanner.
- **Shape:** Parallel hunters across cloud resources + verification.
- **Entry:** `main.py`.
- **Why interesting:** like sec-af but targets cloud infra rather than source code.

---

## Multimodal / data substrate

### `deeplake-collab/`
- **Domain:** Multi-reasoner builds on top of Deep Lake's multimodal substrate.
- **Shape:** Various — folder is a strategic playbook with multiple demos.
- **Why interesting:** read when your data is multimodal (images + text + video + tensors) and you want composite intelligence over it.

---

## Other / small

### `etl/`
- **Domain:** ETL example.
- **Shape:** Small pipeline.

### `demo/`
- **Domain:** Generic demo scaffold.

### `agent-test/`
- **Domain:** Test agent for SDK development.

### `SWESuite-af/`
- **Domain:** Benchmark suite for SWE-bench style evaluation.

---

## How to use this map

1. Read the user's problem.
2. Skim this map. Find the 1–2 examples whose **problem shape** most closely matches.
3. Open the actual code (use `Read` on the entry reasoner file).
4. Grep for `app.call(` and `@app.reasoner` / `@router.reasoner` to see the real call graph.
5. **Do not copy verbatim.** Adapt the decomposition discipline to your problem by walking the five principles again.

If no example matches, walk the principles fresh. The point of the map is to shortcut "what does a clean decomposition look like in code?" — not to substitute for thinking about your problem.
