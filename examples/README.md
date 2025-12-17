# AgentField Examples

This directory contains example agents demonstrating AgentField's capabilities across multiple programming languages and use cases.

## Quick Reference Table

| Use Case | Python | TypeScript | Go |
|----------|--------|------------|-----|
| **Getting Started** | [hello_world](python_agent_nodes/hello_world/) | [init-example](ts-node-examples/init-example/) | [go_agent_hello_world](go_agent_nodes/go_agent_hello_world/) |
| **Basic RAG** | [hello_world_rag](python_agent_nodes/hello_world_rag/) | [discovery-memory](ts-node-examples/discovery-memory/) | - |
| **Production RAG** | [agentic_rag](python_agent_nodes/agentic_rag/) | - | - |
| **Documentation Q&A** | [documentation_chatbot](python_agent_nodes/documentation_chatbot/) | - | - |
| **RAG Evaluation** | [rag_evaluation](python_agent_nodes/rag_evaluation/) | - | - |
| **Deep Research** | [deep_research_agent](python_agent_nodes/deep_research_agent/) | - | - |
| **Image Generation** | [image_generation_hello_world](python_agent_nodes/image_generation_hello_world/) | - | - |
| **Multi-Agent Simulation** | [simulation_engine](python_agent_nodes/simulation_engine/) | [simulation](ts-node-examples/simulation/) | - |
| **Serverless Deployment** | [serverless_hello](python_agent_nodes/serverless_hello/) | [serverless-hello](ts-node-examples/serverless-hello/) | - |
| **Verifiable Credentials** | - | [verifiable-credentials](ts-node-examples/verifiable-credentials/) | - |

## Examples by Language

### Python Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| [hello_world](python_agent_nodes/hello_world/) | Minimal agent foundation | Skills, Reasoners, Call graphs, Pydantic schemas |
| [hello_world_rag](python_agent_nodes/hello_world_rag/) | Simple vector memory RAG | Document ingestion, Semantic search, Memory APIs |
| [image_generation_hello_world](python_agent_nodes/image_generation_hello_world/) | Multi-provider image generation | DALL-E, Gemini, Prompt enhancement, Vision APIs |
| [agentic_rag](python_agent_nodes/agentic_rag/) | Production-grade document Q&A | Ensemble retrieval, Hallucination prevention, Citations |
| [deep_research_agent](python_agent_nodes/deep_research_agent/) | Recursive research planning | Task decomposition, Web search (Tavily), Parallel execution |
| [documentation_chatbot](python_agent_nodes/documentation_chatbot/) | Enterprise RAG system | 3-reasoner architecture, Markdown-aware chunking, Inline citations |
| [rag_evaluation](python_agent_nodes/rag_evaluation/) | Multi-metric QA assessment | Faithfulness, Relevance, Hallucination detection, Constitutional checks |
| [simulation_engine](python_agent_nodes/simulation_engine/) | Domain-agnostic multi-agent simulation | 100+ parallel reasoners, Scenario analysis, Sentiment modeling |
| [serverless_hello](python_agent_nodes/serverless_hello/) | Serverless deployment pattern | Lambda/Cloud Functions handler, Cross-agent calling |

### TypeScript Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| [init-example](ts-node-examples/init-example/) | Basic TypeScript setup | Router-based organization, OpenAI integration, Dev mode |
| [discovery-memory](ts-node-examples/discovery-memory/) | Memory & Discovery APIs | Vector storage, Agent discovery, Workflow progress |
| [simulation](ts-node-examples/simulation/) | Multi-router simulation | 5 parallel routers, OpenRouter support, Schema organization |
| [verifiable-credentials](ts-node-examples/verifiable-credentials/) | Cryptographic audit trails | DIDs, W3C VCs, Ed25519 signatures, Compliance |
| [serverless-hello](ts-node-examples/serverless-hello/) | Serverless handler pattern | AWS Lambda, Express fallback, Reasoner relay |

### Go Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| [go_agent_hello_world](go_agent_nodes/go_agent_hello_world/) | Full Go SDK demonstration | CLI + Control plane, 3-reasoner chain, Environment config |

## Use Case Deep Dives

### Retrieval-Augmented Generation (RAG)

Start with **hello_world_rag** to understand fundamentals, progress to **agentic_rag** for production patterns with anti-hallucination, and use **rag_evaluation** to assess quality.

```
hello_world_rag → agentic_rag → documentation_chatbot → rag_evaluation
     (basic)        (production)      (enterprise)        (quality)
```

### Multi-Agent Simulation

Both Python and TypeScript implementations demonstrate enterprise scenario simulation with parallel agent execution:

- **Python**: `simulation_engine` - 5-layer architecture with 100+ concurrent reasoners
- **TypeScript**: `simulation` - Router-based organization with multi-level parallelism

### Serverless Deployment

Deploy agents to AWS Lambda, Google Cloud Functions, or other serverless platforms:

- **Python**: `serverless_hello` - Universal `handler(event, context)` pattern
- **TypeScript**: `serverless-hello` - Express fallback for local testing

## Technology Stack

| Category | Technologies |
|----------|--------------|
| **LLM Providers** | OpenAI, OpenRouter, Gemini, LiteLLM |
| **Embeddings** | FastEmbed, OpenAI Embeddings, HuggingFace |
| **Web Search** | Tavily API |
| **Deployment** | Docker, AWS Lambda, Cloud Functions, Vercel |
| **Schemas** | Pydantic (Python), Zod (TypeScript) |

## Getting Started

1. Choose an example based on your use case and preferred language
2. Follow the README in each example directory for setup instructions
3. Most examples require environment variables for LLM API keys

```bash
# Example: Running the Python hello_world
cd python_agent_nodes/hello_world
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
python main.py
```
