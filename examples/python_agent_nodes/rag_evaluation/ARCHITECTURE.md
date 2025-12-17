# RAG Evaluation Agent Node - Architecture

Multi-reasoner evaluation system for RAG-generated responses with enterprise-grade metrics and configurable depth.

## Overview

This agent node provides comprehensive evaluation of RAG (Retrieval-Augmented Generation) responses using four distinct metrics, each with a unique multi-reasoner architecture designed to create impressive, observable workflow graphs.

## Core Metrics & Architectures

### 1. Faithfulness (Adversarial Debate Pattern)

**What it measures**: Is the answer grounded in the retrieved context?

**Architecture**: Prosecutor vs. Defender debate with Judge synthesis. This pattern reduces bias by forcing thorough consideration of both attack and defense perspectives.

```mermaid
graph TD
    A[evaluate_faithfulness_full] --> B[extract_claims]
    B -->|"Atomic Claims"| C[prosecute_claims]
    B -->|"Atomic Claims"| D[defend_claims]
    C -->|"Attack Arguments"| E[judge_faithfulness]
    D -->|"Defense Arguments"| E
    E --> F[Faithfulness Verdict]

    subgraph "Parallel Adversarial Phase"
        C
        D
    end

    style A fill:#4dabf7
    style C fill:#ff6b6b
    style D fill:#69db7c
    style E fill:#748ffc
```

**Reasoners**:
| Reasoner | Role |
|----------|------|
| `extract_claims` | Decompose response into atomic claims |
| `prosecute_claims` | Find unsupported/contradicted claims (ATTACK) |
| `defend_claims` | Find supporting evidence for claims (DEFEND) |
| `judge_faithfulness` | Weigh arguments, produce final verdict |

**AI Calls**: 4 (extract + prosecutor + defender + judge)

---

### 2. Answer Relevance (Multi-Jury Consensus Pattern)

**What it measures**: Does the answer actually address the question?

**Architecture**: Multiple jurors with different perspectives vote, foreman synthesizes. Captures multi-dimensional relevance.

```mermaid
graph TD
    A[evaluate_relevance_full] --> B[analyze_question]
    B -->|"Intent Analysis"| C[vote_literal_relevance]
    B -->|"Intent Analysis"| D[vote_intent_relevance]
    B -->|"Intent Analysis"| E[vote_scope_relevance]
    C -->|"Literal Vote"| F[synthesize_relevance_verdict]
    D -->|"Intent Vote"| F
    E -->|"Scope Vote"| F
    F --> G[Relevance Verdict]

    subgraph "Parallel Jury Deliberation"
        C
        D
        E
    end

    style A fill:#4dabf7
    style F fill:#ffd43b
```

**Reasoners**:
| Reasoner | Role |
|----------|------|
| `analyze_question` | Extract question intent and sub-questions |
| `vote_literal_relevance` | Does response literally answer the question? |
| `vote_intent_relevance` | Does response address underlying user need? |
| `vote_scope_relevance` | Is response appropriately scoped? |
| `synthesize_relevance_verdict` | Aggregate votes, handle disagreements |

**AI Calls**: 5 (analyze + 3 jurors + foreman)

---

### 3. Hallucination Detection (Hybrid ML+LLM Chain)

**What it measures**: Does the response contain fabricated or unsupported information?

**Architecture**: ML-first filtering with LLM escalation for uncertain cases. Achieves 60-80% cost reduction.

```mermaid
flowchart TD
    A[evaluate_hallucination_full] --> B[extract_statements]
    B --> C[verify_statements_ml]
    C --> D{Confidence?}
    D -->|High| E[Verified by ML]
    D -->|Low| F[Unverified by ML]
    D -->|Uncertain| G[verify_uncertain_statements]

    G --> H[verify_statement_llm_0]
    G --> I[verify_statement_llm_1]
    G --> J[verify_statement_llm_N]

    E --> K[synthesize_hallucination_report]
    F --> K
    H --> K
    I --> K
    J --> K
    K --> L[Hallucination Report]

    subgraph "ML Layer (Fast, Cheap)"
        B
        C
    end

    subgraph "LLM Escalation (Parallel)"
        H
        I
        J
    end

    style A fill:#4dabf7
    style C fill:#74c0fc
    style K fill:#69db7c
```

**Components**:
| Component | Type | Role |
|-----------|------|------|
| `extract_statements` | Reasoner+Skill | Extract factual statements (ML NER) |
| `verify_statements_ml` | Reasoner+Skill | Batch ML verification (embeddings + NLI) |
| `verify_statement_llm` | Reasoner | LLM verification for single uncertain statement |
| `synthesize_hallucination_report` | Reasoner | Aggregate findings, compute metrics |

**AI Calls**: 2 + N (extract + synthesize + N uncertain statements)

---

### 4. Constitutional Compliance (Principles-Based Pattern)

**What it measures**: Does the response adhere to configurable evaluation principles?

**Architecture**: Parallel principle checkers with domain-weighted aggregation. Fully customizable via YAML.

```mermaid
flowchart TB
    A[evaluate_constitutional_full] --> B[Load Constitution]

    B --> C[check_no_fabrication]
    B --> D[check_accurate_attribution]
    B --> E[check_completeness]
    B --> F[check_safety]
    B --> G[check_uncertainty_expression]

    C --> H[aggregate_constitutional]
    D --> H
    E --> H
    F --> H
    G --> H

    H --> I[Constitutional Report]

    subgraph "Parallel Principle Checks"
        C
        D
        E
        F
        G
    end

    style A fill:#4dabf7
    style H fill:#69db7c
```

**Reasoners**:
| Reasoner | Role |
|----------|------|
| `check_no_fabrication` | Check: Every claim traces to source |
| `check_accurate_attribution` | Check: Correct source attribution |
| `check_completeness` | Check: Addresses all question aspects |
| `check_safety` | Check: No harmful advice |
| `check_uncertainty_expression` | Check: Appropriate uncertainty |
| `aggregate_constitutional` | Weight scores, determine compliance |

**AI Calls**: 6 (5 principles + aggregation)

---

## Master Orchestration

The master orchestrator runs all four metrics in parallel, creating an impressive multi-branch workflow graph:

```mermaid
graph TD
    A[evaluate_rag_response<br/>Master Orchestrator] --> B[evaluate_faithfulness_full]
    A --> C[evaluate_relevance_full]
    A --> D[evaluate_hallucination_full]
    A --> E[evaluate_constitutional_full]

    subgraph "Faithfulness - Adversarial Debate"
        B --> B1[extract_claims]
        B1 --> B2[prosecute_claims]
        B1 --> B3[defend_claims]
        B2 --> B4[judge_faithfulness]
        B3 --> B4
    end

    subgraph "Relevance - Multi-Jury"
        C --> C1[analyze_question]
        C1 --> C2[literal_juror]
        C1 --> C3[intent_juror]
        C1 --> C4[scope_juror]
        C2 --> C5[foreman]
        C3 --> C5
        C4 --> C5
    end

    subgraph "Hallucination - Hybrid ML+LLM"
        D --> D1[extract_statements]
        D1 --> D2[ml_verify]
        D2 --> D3[llm_escalation]
        D3 --> D4[synthesize]
    end

    subgraph "Constitutional - Principles"
        E --> E1[check_P1]
        E --> E2[check_P2]
        E --> E3[check_P3]
        E --> E4[check_P4]
        E --> E5[check_P5]
        E1 --> E6[aggregate]
        E2 --> E6
        E3 --> E6
        E4 --> E6
        E5 --> E6
    end

    B4 --> F[Aggregate Results]
    C5 --> F
    D4 --> F
    E6 --> F
    F --> G[RAG Evaluation Report]

    style A fill:#4dabf7
    style F fill:#69db7c
    style G fill:#ffd43b
```

---

## Adaptive Depth Modes

The system supports three evaluation depths:

| Mode | Description | AI Calls | Latency | Use Case |
|------|-------------|----------|---------|----------|
| **Quick** | Single reasoner per metric | 4 | ~1s | Real-time validation |
| **Standard** | Multi-reasoner patterns | 10-14 | ~3s | Production evaluation |
| **Thorough** | Full depth on all metrics | 18+ | ~6s | Audits, compliance |

```mermaid
flowchart LR
    A[Input] --> B{Mode?}
    B -->|Quick| C[4 Parallel Single Reasoners]
    B -->|Standard| D[Multi-Reasoner Patterns]
    B -->|Thorough| E[Full Depth All Metrics]

    C --> F[Result]
    D --> F
    E --> F

    style C fill:#69db7c
    style D fill:#ffd43b
    style E fill:#ff6b6b
```

---

## Cost Analysis

| Evaluation Mode | Cost per 1000 evals | vs Pure LLM Savings |
|-----------------|---------------------|---------------------|
| Quick (ML-heavy) | $2-5 | 85% |
| Standard (Hybrid) | $8-15 | 70% |
| Thorough (LLM-heavy) | $25-40 | 40% |

The hybrid ML+LLM approach in hallucination detection achieves significant cost savings by using lightweight HuggingFace models for initial filtering:

- **all-MiniLM-L6-v2**: 22M params, ~20ms/inference for embeddings
- **DeBERTa-v3-base-MNLI**: For entailment checking
- **spaCy en_core_web_sm**: For named entity recognition

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `evaluate_rag_response` | Full evaluation with all 4 metrics |
| `evaluate_faithfulness_only` | Faithfulness only |
| `evaluate_relevance_only` | Relevance only |
| `evaluate_hallucination_only` | Hallucination detection only |
| `evaluate_constitutional_only` | Constitutional compliance only |

---

## Constitutional Configuration

Principles are configurable via YAML (`config/constitution.yaml`):

```yaml
principles:
  - id: no_fabrication
    name: "No Fabrication"
    description: "Every factual claim must trace to source material"
    weight: 1.0
    severity_if_violated: critical

domain_weights:
  medical:
    safety: 1.5  # Extra weight for medical safety
    no_fabrication: 1.2
```

Domain presets available:
- `medical.yaml` - Stricter safety, dosage accuracy
- `legal.yaml` - Citation accuracy, jurisdiction awareness
- `financial.yaml` - Numerical accuracy, risk disclosure

---

## File Structure

```
rag-evaluation/
├── main.py                      # Agent entry point
├── models.py                    # Pydantic schemas
├── ml_models.py                 # ML model loading
├── config/
│   ├── constitution.yaml        # Default principles
│   └── presets/
│       ├── medical.yaml
│       ├── legal.yaml
│       └── financial.yaml
├── reasoners/
│   ├── __init__.py              # Router registration
│   ├── orchestrator.py          # Master orchestrator
│   ├── faithfulness.py          # Adversarial debate
│   ├── relevance.py             # Multi-jury consensus
│   ├── hallucination.py         # Hybrid ML+LLM
│   └── constitutional.py        # Principles-based
├── ml_services/
│   ├── __init__.py
│   ├── embeddings.py            # MiniLM-L6-v2
│   ├── nli.py                   # DeBERTa-MNLI
│   └── ner.py                   # spaCy NER
└── ARCHITECTURE.md
```

---

## Usage Example

```python
# Full evaluation
result = await app.call(
    "rag-evaluation.evaluate_rag_response",
    question="What is the capital of France?",
    context="France is a country in Europe. Paris is the capital city of France.",
    response="The capital of France is Paris, a beautiful city known for the Eiffel Tower.",
    mode="standard",
    domain="general"
)

# Result includes:
# - faithfulness: Adversarial debate verdict
# - relevance: Multi-jury consensus verdict
# - hallucination: Hybrid ML+LLM report
# - constitutional: Principles compliance report
# - overall_score: Weighted aggregate
# - quality_tier: excellent/good/acceptable/poor/critical
# - recommendations: Improvement suggestions
```

---

## Design Philosophy

1. **Guided Autonomy**: AI makes decisions, but within structured frameworks
2. **Observable by Default**: Every reasoner creates workflow graph nodes
3. **Cost-Efficient**: Hybrid ML+LLM reduces costs 60-80%
4. **Configurable Depth**: Adapt evaluation intensity to use case
5. **Domain Customizable**: YAML-based principles for any domain
6. **Enterprise Ready**: Production-grade with full audit trails
