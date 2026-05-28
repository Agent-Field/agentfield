"""Decision simulation router for simulation engine."""

import asyncio
from typing import List

from agentfield import AgentRouter

from schemas import EntityDecision, EntityProfile, ScenarioAnalysis

decision_router = AgentRouter(prefix="decision")


@decision_router.reasoner()
async def simulate_entity_decision(
    entity: EntityProfile,
    scenario: str,
    scenario_analysis: ScenarioAnalysis,
    context: List[str] = [],
) -> EntityDecision:
    """
    Simulates decision with error handling and simplified prompts.
    Only shows key attributes (5-7) instead of all attributes to reduce JSON parsing issues.
    """
    try:
        context_str = "\n".join([f"- {c}" for c in context]) if context else ""

        # Only show top 5-7 key attributes, not all attributes
        key_attrs = scenario_analysis.key_attributes[:7]  # Max 7 attributes
        if not key_attrs:
            # Fallback: use first 5 attributes if key_attributes not set
            key_attrs = list(entity.attributes.keys())[:5]

        # Build simplified attributes string (only key attributes)
        key_attributes_str = "\n".join(
            [
                f"  • {k}: {entity.attributes.get(k, 'N/A')}"
                for k in key_attrs
                if k in entity.attributes
            ]
        )

        context_section = f"ADDITIONAL CONTEXT:\n{context_str}\n\n" if context else ""

        prompt = f"""You are simulating the decision-making of a specific {scenario_analysis.entity_type}.

WHO YOU ARE:
{entity.profile_summary}

KEY ATTRIBUTES (most relevant for this decision):
{key_attributes_str}

SCENARIO YOU'RE FACING:
{scenario}

{context_section}AVAILABLE DECISIONS:
{', '.join(scenario_analysis.decision_options)}

TASK:
Based on who you are, decide how you would respond to this scenario.

1. decision: Choose one option from the available decisions list.

2. confidence: Rate confidence 0.0-1.0. How certain are you?

3. key_factor: What single attribute influenced this decision most? (max 50 words)

4. trade_off: What was the main trade-off you considered? (max 50 words)

5. reasoning: Optional brief explanation (1-2 sentences, max 100 words).

Be concise and realistic."""

        result = await decision_router.ai(prompt, schema=EntityDecision)
        result.entity_id = entity.entity_id
        return result

    except Exception as e:
        print(f"⚠️  Failed entity {entity.entity_id}: {str(e)[:100]}")
        # Return a default decision instead of failing
        return EntityDecision(
            entity_id=entity.entity_id,
            decision=scenario_analysis.decision_options[0]
            if scenario_analysis.decision_options
            else "unknown",
            confidence=0.0,
            key_factor="Error during decision generation",
            trade_off="Unable to evaluate",
            reasoning="Failed to generate decision",
        )


@decision_router.reasoner()
async def simulate_batch_decisions(
    entities: List[EntityProfile],
    scenario: str,
    scenario_analysis: ScenarioAnalysis,
    context: List[str] = [],
    parallel_batch_size: int = 20,
) -> List[EntityDecision]:
    """
    Process with error handling, rate limiting, and global concurrency control.
    - Uses return_exceptions=True to prevent one failure from killing the batch
    - Adds delays between batches to avoid rate limits
    - Filters out failed entities
    """
    all_decisions = []

    # Process in batches to control concurrency
    num_batches = (len(entities) + parallel_batch_size - 1) // parallel_batch_size

    for batch_num, i in enumerate(range(0, len(entities), parallel_batch_size)):
        batch = entities[i : i + parallel_batch_size]

        # Each entity gets its own AI call, but we do them in parallel
        # Use return_exceptions=True so one failure doesn't kill the batch
        tasks = [
            simulate_entity_decision(entity, scenario, scenario_analysis, context)
            for entity in batch
        ]

        # Use return_exceptions=True to handle failures gracefully
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and None values
        valid_decisions = []
        for result in batch_results:
            if isinstance(result, EntityDecision):
                valid_decisions.append(result)
            elif isinstance(result, Exception):
                print(f"⚠️  Exception in batch: {str(result)[:100]}")
            # None values are already filtered

        all_decisions.extend(valid_decisions)

        # Progress reporting
        print(
            f"   Batch {batch_num + 1}/{num_batches}: Completed {len(all_decisions)}/{len(entities)} decisions..."
        )

        # Add delay between batches to avoid rate limits
        # Only delay if not the last batch
        if i + parallel_batch_size < len(entities):
            await asyncio.sleep(0.5)  # 0.5 second delay between batches

    print(
        f"   ✅ Successfully generated {len(all_decisions)}/{len(entities)} decisions"
    )
    return all_decisions
