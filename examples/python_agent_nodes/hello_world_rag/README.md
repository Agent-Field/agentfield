# Hello World RAG

Minimal Retrieval-Augmented agent that demonstrates the new AgentField memory vector APIs without any external services.

## What It Does

| Component | Purpose |
| --- | --- |
| `ingest_document` skill | Accepts raw text or a file path, chunks it, creates toy embeddings, and stores them via `app.memory.set_vector`. |
| `answer_question` reasoner | Embeds the question, calls `app.memory.similarity_search`, and feeds the retrieved context into the LLM to synthesize an answer. |

## Running

```bash
cd examples/python_agent_nodes/hello_world
pip install -r requirements.txt
python main.py
```

Once the agent is registered with the control plane, you can call it via the AgentField CLI or direct HTTP:

```bash
# Ingest a document (either provide text or a path)
agentfield call hello-world-rag ingest_document \
  --json '{"document_id": "guide", "path": "docs/README.md"}'

# Ask a question
agentfield call hello-world-rag answer_question \
  --json '{"question": "What does the README say about deployment?"}'
```

The output includes both the synthesized answer and the supporting chunks so you can trace where the knowledge came from.
