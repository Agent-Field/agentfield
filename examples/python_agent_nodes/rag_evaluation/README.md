# RAG Evaluation Agent

Multi-reasoner evaluation system for RAG-generated responses featuring adversarial debate, jury consensus, and hybrid ML+LLM verification.

## Features

- **4 Evaluation Metrics** with unique multi-reasoner architectures
- **Adversarial Debate** - Prosecutor vs Defender with Judge verdict (Faithfulness)
- **Multi-Jury Consensus** - 3 jurors vote on literal/intent/scope (Relevance)
- **Hybrid ML+LLM** - 60-80% cost reduction with HuggingFace models (Hallucination)
- **Configurable Principles** - YAML-based constitutional evaluation
- **Adaptive Depth** - Quick (4 calls), Standard (14 calls), Thorough (20+ calls)
- **Web UI** - Beautiful claim-level breakdown with multi-perspective analysis

## Quick Start (Docker)

The easiest way to run the full stack (Control Plane + RAG Agent + Web UI):

```bash
# 1. Copy environment template and add your API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 2. Start everything with Docker Compose
docker-compose up --build

# 3. Access the services:
# - Web UI: http://localhost:3000
# - Control Plane: http://localhost:8080
# - RAG Agent: http://localhost:8001
```

### Faster Docker Builds

For faster builds during development, skip the heavy ML dependencies:

```bash
# Fast build without ML dependencies (~30s vs ~3min)
INSTALL_ML_DEPS=false docker-compose up --build
```

**Note:** Without ML deps, the agent runs in "LLM-only" mode (no local embeddings). This is fine for most evaluation use cases.

## Quick Start (Local Development)

Running locally requires 3 components:

```bash
# Terminal 1: Start the control plane
af server

# Terminal 2: Start the RAG agent
cp .env.example .env  # Add your OPENROUTER_API_KEY
pip install -r requirements.txt
python main.py

# Terminal 3 (optional): Start the Web UI
cd ui
npm install
npm run dev
```

**Services:**
- Control Plane: http://localhost:8080
- RAG Agent: http://localhost:8001
- Web UI: http://localhost:3000 (optional)

> **Note:** The Web UI is optional - you can also use curl/API calls directly against the control plane.

## Execute Evaluation

```bash
curl -X POST http://localhost:8080/api/v1/execute/rag-evaluation.evaluate_rag_response \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "question": "What are the main causes of climate change?",
      "context": "Climate change is primarily caused by human activities, particularly the burning of fossil fuels such as coal, oil, and natural gas. This releases greenhouse gases like carbon dioxide (CO2) and methane into the atmosphere.",
      "response": "The main causes of climate change are burning fossil fuels (coal, oil, gas) which releases CO2.",
      "mode": "standard",
      "domain": "general"
    }
  }'
```

## Architecture

The system runs four evaluation metrics in parallel:

1. **Faithfulness** - Adversarial debate: extract claims -> prosecutor attacks -> defender defends -> judge verdict
2. **Relevance** - Multi-jury: analyze question -> 3 parallel jurors vote -> foreman synthesizes
3. **Hallucination** - Hybrid: ML verification (fast) -> LLM escalation (uncertain cases only)
4. **Constitutional** - Parallel principle checks -> weighted aggregation

See the [documentation](https://agentfield.dev/examples/complete-agents/rag-evaluator) for detailed diagrams and explanations.

## Endpoints

### Main
- **`/api/v1/execute/rag-evaluation.evaluate_rag_response`** - Full evaluation (recommended)
  - Params: `question`, `context`, `response`, `mode` (quick/standard/thorough), `domain` (general/medical/legal/financial)

### Individual Metrics
- `/api/v1/execute/rag-evaluation.evaluate_faithfulness_only` - Faithfulness only
- `/api/v1/execute/rag-evaluation.evaluate_relevance_only` - Relevance only
- `/api/v1/execute/rag-evaluation.evaluate_hallucination_only` - Hallucination only
- `/api/v1/execute/rag-evaluation.evaluate_constitutional_only` - Constitutional only

## Evaluation Modes

| Mode | AI Calls | Latency | Use Case |
|------|----------|---------|----------|
| **quick** | 4 | ~1s | Real-time validation |
| **standard** | 10-14 | ~3s | Production evaluation |
| **thorough** | 18+ | ~6s | Audits, compliance |

## Web UI

The web interface provides:
- **Claim-level breakdown** - See each claim evaluated with prosecution vs defense arguments
- **Multi-perspective analysis** - View results from 4 different evaluation approaches
- **Preset examples** - Try pre-loaded examples for different domains
- **Mode selection** - Quick/Standard/Thorough evaluation depth

## Configuration

Edit `config/constitution.yaml` to customize evaluation principles:

```yaml
principles:
  - id: no_fabrication
    name: "No Fabrication"
    weight: 1.0
    severity_if_violated: critical

domain_weights:
  medical:
    safety: 1.5
    no_fabrication: 1.2
```

Domain presets available in `config/presets/`:
- `medical.yaml` - Stricter safety, dosage accuracy
- `legal.yaml` - Citation accuracy, jurisdiction awareness
- `financial.yaml` - Numerical accuracy, risk disclosure

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTFIELD_SERVER` | Control plane URL | `http://localhost:8080` |
| `AI_MODEL` | LLM model | `openrouter/deepseek/deepseek-chat-v3-0324` |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |

## File Structure

```
rag_evaluation/
├── main.py                      # Agent entry point
├── models.py                    # Pydantic schemas
├── docker-compose.yml           # Full stack deployment
├── Dockerfile                   # Agent container
├── config/
│   ├── constitution.yaml        # Default principles
│   └── presets/                 # Domain-specific configs
├── reasoners/
│   ├── __init__.py              # Router registration
│   ├── orchestrator.py          # Master orchestrator
│   ├── faithfulness.py          # Adversarial debate
│   ├── relevance.py             # Multi-jury consensus
│   ├── hallucination.py         # Hybrid ML+LLM
│   └── constitutional.py        # Principles-based
├── ml_services/                 # Optional ML components
│   ├── embeddings.py            # MiniLM-L6-v2
│   ├── nli.py                   # DeBERTa-MNLI
│   └── ner.py                   # spaCy NER
├── ui/                          # Next.js web interface
├── requirements.txt             # Core dependencies
└── requirements-ml.txt          # ML dependencies (optional)
```
