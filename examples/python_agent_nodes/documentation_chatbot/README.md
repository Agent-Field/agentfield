# Documentation Chatbot (Reasoner-Based RAG)

A production-ready Retrieval-Augmented Generation (RAG) node that answers complex questions about your private documentation **without hallucinations**. It ingests local documentation folders, creates high-quality embeddings with precise chunk metadata, and produces answers with inline citations (Perplexity-style: `Feature works like X [A]`).

## Highlights
- **Single-file agent** – ingestion and Q&A sit right on the agent, no prefixes or extra routers to manage.
- **3-reasoner parallel architecture** – `plan_queries → parallel_retrieve → synthesize_answer` forms an efficient call graph in the control plane.
- **Self-aware synthesis** – every response includes confidence assessment; if incomplete, the system automatically refines with targeted retrieval (max 1 iteration).
- **Document-aware retrieval** – retrieves and analyzes full documentation pages instead of isolated chunks for better context comprehension.
- **Two-tier storage** – documents stored once in regular memory, chunks reference them via keys (70% storage savings vs naive duplication).
- **Inline citations** – the LLM references short keys (`[A]`, `[B]`) mapped to specific files and line ranges for transparent answers.
- **Chunk metadata** – every chunk keeps `relative_path`, `section`, `line_start/line_end`, and similarity score.
- **Fast, dependency-light** – uses `fastembed` for local embeddings; no external vector DB required.

## Architecture

### 3-Reasoner Parallel System

1. **Query Planner Reasoner** (`plan_queries`)
   - Generates 3-5 diverse search queries from user's question
   - Uses different terminology, aspects, and angles to maximize retrieval coverage
   - Strategies: broad exploration, specific targeting, or mixed approach

2. **Parallel Retrieval Reasoner** (`parallel_retrieve`)
   - Executes all queries concurrently (3x speed improvement)
   - Deduplicates results across queries
   - Returns top 15 unique chunks ranked by relevance

3. **Self-Aware Synthesizer Reasoner** (`synthesize_answer`)
   - Generates markdown answer with inline citations
   - Self-assesses completeness (confidence: high/partial/insufficient)
   - Identifies missing topics if answer is incomplete
   - Triggers automatic refinement if needed (max 1 iteration)

### Two QA Modes

**Chunk-Based QA** (`qa_answer`)
- Works with isolated text chunks
- Faster, lower token usage
- Good for specific factual queries

**Document-Aware QA** (`qa_answer_with_documents`) ⭐ **RECOMMENDED**
- Aggregates chunks to retrieve full documentation pages
- Better context comprehension for complex questions
- Smart document ranking (frequency + relevance scoring)
- Ideal for questions requiring multi-section understanding

## Project Structure
```
documentation_chatbot/
├── chunking.py        # Markdown-aware chunker with line tracking
├── embedding.py       # Shared FastEmbed helpers
├── main.py            # Agent bootstrap + skills/reasoners
├── schemas.py         # Pydantic models shared across reasoners
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r examples/python_agent_nodes/documentation_chatbot/requirements.txt
```

### 2. Run the Agent
```bash
python examples/python_agent_nodes/documentation_chatbot/main.py
```

### 3. Ingest Documentation
POST to `/skills/ingest_folder`:
```json
{
  "folder_path": "~/Docs/product-manual",
  "namespace": "product-docs",
  "chunk_size": 1200,
  "chunk_overlap": 250
}
```

This chunks every `.md`, `.mdx`, `.rst`, or `.txt` file, embeds them, and stores vectors in AgentField's global memory scope using the two-tier storage strategy.

### 4. Ask Questions

**Using Document-Aware QA (Recommended)**

POST to `/reasoners/qa_answer_with_documents`:
```json
{
  "question": "How does delta syncing work?",
  "namespace": "product-docs",
  "top_k": 6,
  "min_score": 0.35,
  "top_documents": 5
}
```

**Using Chunk-Based QA**

POST to `/reasoners/qa_answer`:
```json
{
  "question": "How does delta syncing work?",
  "namespace": "product-docs",
  "top_k": 6,
  "min_score": 0.35
}
```

### Response Format
```json
{
  "answer": "Delta syncing replays only changed blocks to remote storage [A]. The system maintains block manifests to track modifications [A][B].",
  "citations": [
    {
      "key": "A",
      "relative_path": "syncing/architecture.md",
      "start_line": 42,
      "end_line": 58,
      "section": "Delta transport",
      "preview": "Delta sync uploads only changed block manifests...",
      "score": 0.83
    },
    {
      "key": "B",
      "relative_path": "syncing/implementation.md",
      "start_line": 15,
      "end_line": 30,
      "section": "Block tracking",
      "preview": "Block manifests are stored in a merkle tree...",
      "score": 0.78
    }
  ],
  "confidence": "high",
  "needs_more": false,
  "missing_topics": []
}
```

## Available Endpoints

### Skills
- **`/skills/ingest_folder`** – Ingest documentation using two-tier storage
  - Parameters: `folder_path`, `namespace`, `glob_pattern`, `chunk_size`, `chunk_overlap`
  - Returns: `IngestReport` with file count, chunk count, and skipped files

### Reasoners
- **`/reasoners/plan_queries`** – Generate diverse search queries
  - Parameters: `question`
  - Returns: `QueryPlan` with 3-5 queries and strategy

- **`/reasoners/parallel_retrieve`** – Execute parallel chunk retrieval
  - Parameters: `queries`, `namespace`, `top_k`, `min_score`
  - Returns: List of `RetrievalResult` (deduplicated chunks)

- **`/reasoners/synthesize_answer`** – Generate self-aware answer from chunks
  - Parameters: `question`, `results`, `is_refinement`
  - Returns: `DocAnswer` with answer, citations, and confidence assessment

- **`/reasoners/qa_answer`** – Chunk-based QA orchestrator
  - Parameters: `question`, `namespace`, `top_k`, `min_score`
  - Returns: `DocAnswer` with automatic refinement if needed

- **`/reasoners/qa_answer_with_documents`** ⭐ **RECOMMENDED** – Document-aware QA orchestrator
  - Parameters: `question`, `namespace`, `top_k`, `min_score`, `top_documents`
  - Returns: `DocAnswer` using full document context

## Design Notes

### Two-Tier Storage Strategy
- **Tier 1**: Full documents stored ONCE in regular memory with key `{namespace}:doc:{relative_path}`
- **Tier 2**: Chunk vectors stored with metadata including `document_key` reference
- **Benefits**: 70% storage savings, no text duplication, easy document retrieval

### Namespace Isolation
Keep multiple doc sets isolated by using different `namespace` values for both ingestion and QA. Examples:
- `"product-docs"` for product documentation
- `"api-reference"` for API docs
- `"internal-wiki"` for internal knowledge base

### Citation Safety
The LLM only has access to a `key → snippet` map, so every fact must be backed by a retrieved chunk key. Citations can be post-processed: `[A]` → "syncing/architecture.md · lines 42-58".

### Self-Aware Synthesis
The synthesizer automatically assesses answer completeness:
- **confidence='high'**: Complete answer with all key details
- **confidence='partial'**: Some information present but incomplete
- **confidence='insufficient'**: Context doesn't contain relevant information

If `needs_more=True`, the system automatically performs one refinement iteration with targeted queries for missing topics.

### Document Ranking Algorithm
For document-aware QA, documents are scored using:
```
score = (chunk_frequency × 0.4) + (avg_similarity × 0.4) + (max_similarity × 0.2)
```

This rewards:
- Documents with multiple matching chunks (comprehensive coverage)
- High average relevance across chunks
- At least one highly relevant section

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTFIELD_SERVER` | Control plane server URL | `http://localhost:8080` |
| `DOC_EMBED_MODEL` | FastEmbed model for embeddings | `BAAI/bge-small-en-v1.5` |
| `AI_MODEL` | Primary LLM (handled by AgentField `AIConfig`) | `openrouter/openai/gpt-4o-mini` |
| `PORT` | Agent server port (optional, uses auto-port if not set) | Auto-assigned |

## Performance Characteristics

- **Parallel retrieval**: 3x faster than sequential query execution
- **Storage efficiency**: 70% reduction vs storing full text in every chunk
- **Max refinement iterations**: 1 (prevents infinite loops)
- **Default chunk size**: 1200 characters with 250 character overlap
- **Default top-k**: 6 chunks per query
- **Default min score**: 0.35 similarity threshold

## Next Steps

- Add schedulers/watchers to auto re-ingest docs on git changes
- Stream answers token-by-token via AgentField streaming APIs
- Layer evaluators (unit tests) that hit the QA endpoints with canonical Q&A pairs to detect regressions
- Implement multi-language support for non-English documentation
- Add support for PDF and DOCX file formats
