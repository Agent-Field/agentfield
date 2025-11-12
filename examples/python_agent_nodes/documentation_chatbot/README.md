# Documentation Chatbot (Agentic RAG)

A production-ready Retrieval-Augmented Generation (RAG) node designed to answer complex questions about your private documentation **without hallucinations**. It ingests any local documentation folder, creates high-quality embeddings with precise chunk metadata, and produces answers that cite the exact page + line range inline (Perplexity-style: `Feature works like X [A]`).

## Highlights
- **Single-file agent** – ingestion and Q&A sit right on the agent, no prefixes or extra routers to manage.
- **Agentic planning** – `qa_plan → qa_generate_queries → qa_retrieve → qa_synthesize → qa_review ↺ qa_refine_queries` automatically forms a call graph in the control plane.
- **Self-critiquing answers** – every response is reviewed for grounding/completeness; if the critique flags gaps we re-retrieve and rewrite.
- **Inline citations** – the LLM only sees a `key -> snippet` context map, so it references short keys (`[A]`, `[B]`) that you can later swap for rich UI citations.
- **Chunk metadata** – every chunk keeps `relative_path`, `section`, `line_start/line_end`, and similarity score for transparent answers.
- **Fast, dependency-light** – uses `fastembed` for local embeddings; no external vector DB required.

## Project structure
```
documentation_chatbot/
├── chunking.py        # Markdown-aware chunker with line tracking
├── embedding.py       # Shared FastEmbed helpers
├── main.py            # Agent bootstrap + skills/reasoners
├── schemas.py         # Pydantic models shared across routers
├── requirements.txt
└── README.md
```

## Quick start
1. **Install deps** (optional virtualenv recommended):
   ```bash
   pip install -r examples/python_agent_nodes/documentation_chatbot/requirements.txt
   ```
2. **Run the agent**:
   ```bash
   python examples/python_agent_nodes/documentation_chatbot/main.py
   ```
3. **Ingest your docs** (POST to `/skills/ingest_folder`):
   ```json
   {
     "folder_path": "~/Docs/product-manual",
     "namespace": "product-docs"
   }
   ```
   This chunks every `.md`, `.mdx`, `.rst`, or `.txt` file, embeds them, and stores vectors in AgentField's global memory scope.
4. **Ask questions** (POST to `/reasoners/qa_answer`):
   ```json
   {
     "question": "How does delta syncing work?",
     "namespace": "product-docs"
   }
   ```
   Responses look like:
   ```json
   {
     "answer": "Delta syncing replays only changed blocks to remote storage [A][C].",
     "citations": [
       {
         "key": "A",
         "relative_path": "syncing/architecture.md",
         "start_line": 42,
         "end_line": 58,
         "section": "Delta transport",
         "preview": "Delta sync uploads only changed block manifests...",
         "score": 0.83
       }
     ],
     "plan": {
       "search_terms": ["delta syncing", "block manifest"],
       "must_include": ["delta"],
       "answer_style": "direct"
     }
   }
   ```

## Design notes
- **Namespace aware** – keep multiple doc sets isolated by sending a `namespace` argument to both ingestion + QA.
- **Citation safety** – the LLM only has access to the `key -> snippet` map, so every fact must be backed by a retrieved chunk key. We can later post-process `[A]` → "syncing/architecture.md · lines 42-58".
- **Multi-reasoner call graph** – `qa_answer` now chains `qa_focus_question → qa_plan → qa_generate_queries → qa_retrieve → qa_synthesize → qa_review` (with optional `qa_refine_queries` loops), so each phase is observable in the control plane.
- **Hallucination guardrails** – the review reasoner halts finalization if the draft answer lacks evidence, forcing a refined retrieval pass before returning to users.
- **Extensible** – swap `fastembed` for any embedding model (update `DOC_EMBED_MODEL` env var) or plug in extra reasoners for telemetry.

### Extra endpoints for debugging
- `/reasoners/qa_focus_question` – inspect how the agent distills conversation blobs into a clean question.
- `/reasoners/qa_plan` – view the retrieval instructions.
- `/reasoners/qa_generate_queries` – inspect the complementary search angles that broaden retrieval.
- `/reasoners/qa_refine_queries` – see the follow-up search prompts generated whenever the critic asks for more context.
- `/reasoners/qa_retrieve` – inspect the snippet window chosen for the current plan.
- `/reasoners/qa_synthesize` – see the raw markdown answer with inline citation keys.
- `/reasoners/qa_review` – get the critique verdict (needs_more_context, missing_terms, unsupported_claims).

## Environment variables
| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTFIELD_SERVER` | Control plane server URL | `http://localhost:8080` |
| `DOC_EMBED_MODEL` | FastEmbed model for chunk + question embeddings | `BAAI/bge-small-en-v1.5` |
| `AI_MODEL` | Primary LLM (handled by AgentField `AIConfig`) | `openrouter/meta-llama/llama-4-maverick` |

## Next steps
- Add schedulers/watchers to auto re-ingest docs on git changes.
- Stream answers token-by-token via AgentField streaming APIs.
- Layer evaluators (unit tests) that hit the `/reasoners/qa_answer` endpoint with canonical Q&A pairs to detect regressions.
