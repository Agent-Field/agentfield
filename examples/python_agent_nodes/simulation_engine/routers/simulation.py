"""Main simulation orchestrator router."""

from __future__ import annotations

from typing import List

from agentfield import AgentRouter

from schemas import SimulationResult

# Import reasoners from other routers
from .aggregation import aggregate_and_analyze
from .decision import simulate_batch_decisions
from .entity import generate_entity_batch
from .scenario import decompose_scenario, generate_factor_graph

simulation_router = AgentRouter(prefix="simulation")


@simulation_router.reasoner()
async def run_simulation(
    scenario: str,
    population_size: int,
    context: List[str] = [],
    parallel_batch_size: int = 20,
    exploration_ratio: float = 0.1,
) -> SimulationResult:
    """
    Scalable orchestrator with proper batching at each phase.

    Handles large N by:
    1. Generating entities in optimized batches (5 per AI call)
    2. Simulating decisions in parallel batches (20 concurrent)
    3. Sampling data for analysis (max 30 examples to AI)

    For small scale testing, use population_size: 20-50 and parallel_batch_size: 10
    """
    print(f"🚀 Starting simulation: {population_size} entities")

    # Phase 1: Understand scenario (single call, always fast)
    print("\n📋 Phase 1: Analyzing scenario...")
    scenario_analysis = await decompose_scenario(scenario, context)
    print(f"   Entity type: {scenario_analysis.entity_type}")
    print(f"   Decision type: {scenario_analysis.decision_type}")
    print(f"   Options: {scenario_analysis.decision_options}")

    # Phase 2: Build factor graph (single call, always fast)
    print("\n🕸️  Phase 2: Building factor graph...")
    factor_graph = await generate_factor_graph(scenario, scenario_analysis, context)
    print(f"   Tracking {len(factor_graph.attributes)} attributes")

    # Phase 3: Generate entities in optimized batches
    print(f"\n👥 Phase 3: Generating {population_size} entities...")

    # Generate in smart batches (5 entities per AI call, parallelize calls)
    entities_per_batch = 20  # Process 20 entities at a time (4 parallel AI calls of 5 each) - reduced for small scale
    all_entities = []

    num_batches = (population_size + entities_per_batch - 1) // entities_per_batch
    for batch_num in range(num_batches):
        start_id = batch_num * entities_per_batch
        batch_size = min(entities_per_batch, population_size - start_id)

        print(
            f"   Batch {batch_num + 1}/{num_batches}: Generating {batch_size} entities..."
        )
        entities = await generate_entity_batch(
            start_id, batch_size, scenario_analysis, factor_graph, exploration_ratio
        )
        all_entities.extend(entities)

    print(f"   ✅ Generated {len(all_entities)} entities")

    # Phase 4: Simulate decisions in controlled parallel batches
    print("\n🎯 Phase 4: Simulating decisions...")
    all_decisions = await simulate_batch_decisions(
        all_entities,
        scenario,
        scenario_analysis,
        context,
        parallel_batch_size=parallel_batch_size,
    )

    print(f"   ✅ Simulated {len(all_decisions)} decisions")

    # Phase 5: Aggregate with sampled data
    print("\n📊 Phase 5: Aggregating results and generating insights...")
    insights = await aggregate_and_analyze(
        scenario, scenario_analysis, factor_graph, all_entities, all_decisions, context
    )

    print("\n✨ Simulation complete!")
    print(f"\n🎯 KEY INSIGHT: {insights.key_insight}")
    print("\n📈 OUTCOME DISTRIBUTION:")
    for decision, pct in insights.outcome_distribution.items():
        print(f"   {decision}: {pct*100:.1f}%")

    return SimulationResult(
        scenario=scenario,
        context=context,
        population_size=population_size,
        scenario_analysis=scenario_analysis,
        factor_graph=factor_graph,
        entities=all_entities,
        decisions=all_decisions,
        insights=insights,
    )
