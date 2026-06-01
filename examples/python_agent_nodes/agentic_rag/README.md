# Agentic RAG - Hallucination-Proof Document Q&A

Production-ready RAG system with ensemble retrieval, iterative refinement, and claim verification.

## Features

- ✅ Lazy-indexed ensemble retrieval (semantic + keyword + type-specific)
- ✅ Confidence-driven iterative refinement
- ✅ Parallel claim verification (anti-hallucination)
- ✅ FastEmbed for semantic search
- ✅ Memory-backed caching

## Quick Start

### 1. Install Dependencies

```bash
cd examples/python_agent_nodes/agentic_rag
pip install -r requirements.txt
```

### 2. Set Environment

```bash
export OPENAI_API_KEY="your-key"
export AGENTFIELD_SERVER="http://localhost:8080"
```

### 3. Start Agent

```bash
python main.py
```

### 4. Query Document

```bash
curl -X POST http://localhost:8080/api/v1/execute/agentic-rag.query_document \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "file_path": "examples/python_agent_nodes/agentic_rag/sample_document.txt",
      "question": "When was the Dartmouth Conference?"
    }
  }'
```

## Example Output

```json
{
  "answer": "The Dartmouth Conference was held in 1956...",
  "citations": [
    {
      "chunk_id": "chunk_0_a3f2b1c8",
      "quote": "The field was officially founded in 1956...",
      "page_num": 0
    }
  ],
  "confidence_score": 0.92,
  "verification_summary": {
    "verified": 2,
    "uncertain": 0,
    "removed": 0
  },
  "completeness_score": 0.95,
  "gaps": []
}
```

## Architecture

7-phase pipeline:
1. **Smart Chunking** - Intelligent text segmentation
2. **Query Analysis** - Complexity assessment & routing
3. **Ensemble Retrieval** - 3 parallel strategies
4. **Iterative Refinement** - Confidence-driven loops
5. **Claim Verification** - Parallel fact-checking
6. **Answer Synthesis** - Verified claims only
7. **Quality Check** - Completeness assessment

## Files

- `main.py` - Agent with all reasoners
- `schemas.py` - Simple Pydantic models (2-4 fields)
- `skills.py` - Deterministic functions + embeddings
- `embedding_manager.py` - FastEmbed wrapper
- `sample_document.txt` - Test document

## License

Apache 2.0
