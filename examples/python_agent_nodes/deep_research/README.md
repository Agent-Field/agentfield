# Deep Research Agent

A research agent that uses recursive planning to break down research questions into subtasks, forms a topological graph for parallel execution, deduplicates tasks, and synthesizes comprehensive reports.

## Features

- **Recursive Planning** – Breaks down questions into subtasks with configurable depth
- **Task Deduplication** – Merges redundant tasks
- **Smart Search Strategy** – Decides synthesis-only vs enhanced search for parent tasks
- **Topological Execution** – Parallel execution with dependency management
- **Web Search** – Tavily API integration with citation tracking

## Quick Start

### 1. Install & Setup

```bash
pip install -r examples/python_agent_nodes/deep_research/requirements.txt
export TAVILY_API_KEY="your-tavily-api-key"
```

### 2. Run Agent

```bash
python examples/python_agent_nodes/deep_research/main.py
```

### 3. Execute Research

```bash
curl -X POST http://localhost:8080/reasoners/planning_execute_deep_research \
  -H "Content-Type: application/json" \
  -d '{
    "research_question": "What is AgentField.ai?",
    "max_depth": 3,
    "max_tasks_per_level": 5
  }'
```

## Architecture

1. **Plan Creation** – Recursively breaks question into distinct tasks
2. **Deduplication** – Merges redundant tasks
3. **Dependency Identification** – Maps task dependencies
4. **Execution** – Topological order:
   - Leaf tasks → web search
   - Parent tasks → synthesis or enhanced search (with context)
5. **Synthesis** – Final report from all findings

## Endpoints

### Main
- **`/reasoners/planning_execute_deep_research`** – Complete workflow (recommended)
  - Params: `research_question`, `max_depth` (default: 3), `max_tasks_per_level` (default: 5)

### Planning
- `/reasoners/planning_create_research_plan` – Create plan only
- `/reasoners/planning_deduplicate_tasks` – Deduplicate tasks
- `/reasoners/planning_identify_dependencies` – Identify dependencies
- `/reasoners/planning_decide_search_strategy` – Decide search strategy

### Research
- `/reasoners/research_execute_research_task` – Execute leaf task (search)
- `/reasoners/research_execute_research_task_with_context` – Execute with dependency context
- `/reasoners/research_synthesize_from_dependencies` – Synthesize from children

## Task Types

- **Leaf Tasks** (no deps) → Web search
- **Parent Tasks** (has deps) → Synthesis-only or enhanced search with context

## Environment Variables

| Variable            | Description               | Default                                      |
| ------------------- | ------------------------- | -------------------------------------------- |
| `TAVILY_API_KEY`    | Tavily API key (required) | -                                            |
| `AGENTFIELD_SERVER` | Control plane URL         | `http://localhost:8080`                      |
| `AI_MODEL`          | LLM model                 | `openrouter/deepseek/deepseek-v3.1-terminus` |
