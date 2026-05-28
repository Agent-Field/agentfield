# Generalized Multi-Agent Simulation System

A domain-agnostic, schema-flexible simulation system that can model any enterprise scenario (pricing, customer support, marketing, policy, etc.) using LLM-powered multi-reasoner architecture with maximum parallelism.

## Highlights

- **Domain-Agnostic**: Works for any enterprise scenario without hardcoded domain logic
- **Simple Schemas**: All reasoners use flat schemas (max 3-4 fields) for easy LLM generation
- **Maximum Parallelism**: 100+ concurrent reasoner calls for fast execution
- **Right Context to Right AI**: Each reasoner gets minimal, focused context
- **Full Observability**: Every decision includes reasoning traces

## Architecture

### Multi-Layer Parallel System

1. **Scenario Analysis Layer** (3 parallel reasoners)
   - `extract_entities` - Identifies actors, organizations, regions
   - `extract_actions` - Identifies possible actions/decisions
   - `extract_outputs` - Identifies desired output metrics

2. **Actor Generation Layer** (100+ parallel reasoners)
   - `generate_actor_trait` - Creates one trait per actor (runs 100x in parallel)
   - `assign_actor_to_group` - Assigns actors to hierarchy levels

3. **Behavior Simulation Layer** (400+ parallel reasoners)
   - `evaluate_action` - Actor evaluates one action (runs for all actor-action pairs)
   - `generate_reasoning` - Generates reasoning text (parallel)
   - `calculate_sentiment` - Calculates sentiment score (parallel)

4. **Interaction Layer** (Optional, parallel)
   - `model_influence` - Models influence between actors
   - `propagate_opinion` - Propagates opinions through network

5. **Aggregation Layer** (Many parallel reasoners)
   - `calculate_metric` - Calculates one metric (runs for each metric)
   - `aggregate_by_level` - Aggregates one hierarchy level (parallel per level)
   - `generate_insight` - Generates one insight (parallel)

## Quick Start

### 1. Install Dependencies

```bash
pip install agentfield
```

### 2. Run the Agent

```bash
python examples/python_agent_nodes/simulation_engine/main.py
```

### 3. Run a Simulation

POST to `/reasoners/run_simulation`:

```json
{
  "scenario_description": "A company is experiencing supply chain delays and needs to decide on discount levels. They want to understand how customers in North America, Europe, and Asia would react to discounts of 5%, 10%, 15%, or 20%.",
  "scenario_context": "{\"company\": \"TechCorp\", \"regions\": [\"North America\", \"Europe\", \"Asia\"], \"discount_options\": [5, 10, 15, 20]}",
  "output_requirements": "{\"include_recommendations\": true, \"breakdown_by_region\": true, \"include_sentiment_analysis\": true}"
}
```

### Response Format

```json
{
  "summary": "{\"total_actors\": 10, \"total_actions\": 4, \"total_responses\": 40, \"hierarchy_levels\": [\"North America\", \"Europe\", \"Asia\"]}",
  "statistics": "{\"positive_rate\": 0.72, \"average_sentiment\": 0.65}",
  "actor_responses": "[{\"actor_id\": \"actor_0001\", \"decision\": \"purchase with 15% discount\", \"confidence\": 0.85, ...}]",
  "hierarchical_breakdown": "{\"North America\": {...}, \"Europe\": {...}, \"Asia\": {...}}",
  "reasoning_traces": "[...]",
  "recommendations": "{\"insights\": [\"Customers prefer 15% discount\", ...]}"
}
```

## Example Scenarios

### Example 1: Pricing Disruption

**Scenario**: Company needs to decide on discount levels for supply chain disruption.

**Input**:
```json
{
  "scenario_description": "A company is experiencing supply chain delays and needs to decide on discount levels. They want to understand how customers in North America, Europe, and Asia would react to discounts of 5%, 10%, 15%, or 20%.",
  "scenario_context": "{\"company\": \"TechCorp\", \"regions\": [\"North America\", \"Europe\", \"Asia\"], \"discount_options\": [5, 10, 15, 20]}",
  "output_requirements": "{\"include_recommendations\": true, \"breakdown_by_region\": true}"
}
```

**What Happens**:
1. System extracts entities: customers, regions, company
2. System extracts actions: evaluate discount options
3. System generates 10 diverse customer actors (parallel)
4. System evaluates each customer's reaction to each discount (40 parallel evaluations)
5. System generates reasoning for each decision (40 parallel)
6. System calculates sentiment for each decision (40 parallel)
7. System aggregates by region (3 parallel aggregations)
8. System calculates metrics and generates insights (parallel)

### Example 2: Customer Support

**Scenario**: SaaS company wants to simulate support ticket routing strategies.

**Input**:
```json
{
  "scenario_description": "A SaaS company wants to simulate how different support ticket routing strategies affect customer satisfaction. They have 3 routing options: round-robin, skill-based, and priority-based.",
  "scenario_context": "{\"ticket_volume\": 1000, \"support_agents\": 20, \"routing_strategies\": [\"round-robin\", \"skill-based\", \"priority-based\"]}",
  "output_requirements": "{\"include_satisfaction_metrics\": true, \"compare_strategies\": true}"
}
```

**What Happens**:
1. System automatically adapts to this completely different domain
2. Extracts entities: customers, support agents, routing strategies
3. Extracts actions: ticket creation, routing decisions, resolution
4. Generates diverse customer and agent actors
5. Simulates ticket flow through each routing strategy
6. Calculates satisfaction metrics for each strategy
7. Generates comparison insights

## Design Principles

### 1. Simple, Flat Schemas

Every reasoner uses schemas with maximum 3-4 fields, no nesting:

```python
class ActionEvaluation(BaseModel):
    decision: str
    confidence: float

class Metric(BaseModel):
    metric_name: str
    metric_value: float
```

This ensures even smaller LLM models can generate valid outputs easily.

### 2. Maximum Parallelism

Instead of one complex reasoner doing everything, we have many small reasoners:

- **Old approach**: 1 reasoner processes 100 actors sequentially
- **New approach**: 100 reasoners process 1 actor each, all in parallel

This provides:
- 100x faster execution (theoretical)
- Better error isolation (one failure doesn't block others)
- Scalability (add more actors = more parallel calls)

### 3. Right Context to Right AI

Each reasoner gets only the context it needs:

- `evaluate_action` gets: actor_id, action_description, actor_traits (minimal)
- `generate_reasoning` gets: decision, actor_id (minimal)
- `calculate_metric` gets: metric_name, data_list (minimal)

This:
- Reduces token usage
- Improves accuracy (focused prompts)
- Enables parallel execution (no shared state)

### 4. Domain-Agnostic Design

No hardcoded domain knowledge. LLMs interpret scenarios dynamically:

- Pricing scenarios → LLM understands discounts, customers, regions
- Support scenarios → LLM understands tickets, agents, routing
- Marketing scenarios → LLM understands campaigns, audiences, channels

The same reasoners work for all domains.

## Available Endpoints

### Main Orchestrator

- **`/reasoners/run_simulation`** - Run a complete simulation
  - Parameters: `SimulationRequest` (scenario_description, scenario_context, output_requirements)
  - Returns: `SimulationResult` with all aggregated results

### Scenario Analysis

- **`/reasoners/extract_entities`** - Extract entities from scenario
- **`/reasoners/extract_actions`** - Extract actions from scenario
- **`/reasoners/extract_outputs`** - Extract output requirements

### Actor Generation

- **`/reasoners/generate_actor_trait`** - Generate one trait for one actor
- **`/reasoners/assign_actor_to_group`** - Assign actor to hierarchy level

### Behavior Simulation

- **`/reasoners/evaluate_action`** - Evaluate action from actor's perspective
- **`/reasoners/generate_reasoning`** - Generate reasoning for a decision
- **`/reasoners/calculate_sentiment`** - Calculate sentiment from text

### Interactions

- **`/reasoners/model_influence`** - Model influence between actors
- **`/reasoners/propagate_opinion`** - Propagate opinion through network

### Aggregation

- **`/reasoners/calculate_metric`** - Calculate one metric from data
- **`/reasoners/aggregate_by_level`** - Aggregate by hierarchy level
- **`/reasoners/generate_insight`** - Generate one insight

## Performance Characteristics

- **Parallel Execution**: 100+ reasoners execute concurrently
- **Batch Processing**: Actors processed in batches of 50-100
- **Error Isolation**: One reasoner failure doesn't block others
- **Scalability**: Linear scaling with number of actors (more actors = more parallel calls)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTFIELD_SERVER` | Control plane server URL | `http://localhost:8080` |
| `AI_MODEL` | Primary LLM model | `openrouter/openai/gpt-4o-mini` |
| `PORT` | Agent server port | Auto-assigned |

## Technical Details

### Schema Design Rules

- Maximum 3-4 fields per schema (enforced)
- No nested objects - use JSON strings when needed
- Flat structures only - all fields at root level
- Simple types: str, float, int, bool, List[str] only

### Parallel Execution Strategy

- **Batch Processing**: Group independent reasoners into batches
- **Context Passing**: Each reasoner gets minimal, focused context
- **asyncio.gather**: Execute 100+ reasoners concurrently per batch
- **Error Isolation**: One reasoner failure doesn't block others

### Memory Usage

- Intermediate results stored in memory during simulation
- Final results returned as JSON strings
- No persistent storage required (stateless design)

## Next Steps

- Add time-stepping support for multi-step simulations
- Implement actor-to-actor communication networks
- Add result streaming for large simulations
- Create scenario templates for common use cases
- Add evaluation metrics for simulation quality
