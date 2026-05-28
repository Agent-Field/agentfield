"""Entity generation router for simulation engine."""

import json
from typing import List

from agentfield import AgentRouter
from pydantic import BaseModel, Field

from schemas import EntityProfile, FactorGraph, ScenarioAnalysis

entity_router = AgentRouter(prefix="entity")


class MiniBatchSchema(BaseModel):
    """Schema for generating multiple entities in one call"""

    entities: List[dict] = Field(description="List of entity attribute dictionaries")


@entity_router.reasoner()
async def generate_entity_batch(
    start_id: int,
    batch_size: int,
    scenario_analysis: ScenarioAnalysis,
    factor_graph: FactorGraph,
    exploration_ratio: float = 0.1,
) -> List[EntityProfile]:
    """
    Generate multiple entities in ONE AI call to save tokens.
    Generate 5-10 entities per call, then parallelize those calls.
    """
    # Generate multiple entities per AI call (but not too many)
    entities_per_call = 5  # Sweet spot for quality vs efficiency
    num_calls = (batch_size + entities_per_call - 1) // entities_per_call

    async def generate_mini_batch(call_num: int) -> List[EntityProfile]:
        start = call_num * entities_per_call
        count = min(entities_per_call, batch_size - start)

        # Determine exploration mode for this mini-batch
        exploration_mode = start < int(batch_size * exploration_ratio)

        mode_instruction = ""
        if exploration_mode:
            mode_instruction = """EXPLORATION MODE: Generate entities with unusual or edge-case attributes.
Sample from distribution tails or create surprising but realistic combinations."""
        else:
            mode_instruction = """STANDARD MODE: Generate typical, realistic entities following
normal distributions and common attribute combinations."""

        prompt = f"""Generate {count} synthetic {scenario_analysis.entity_type} entities for simulation.

AVAILABLE ATTRIBUTES:
{json.dumps(factor_graph.attributes, indent=2)}

ATTRIBUTE RELATIONSHIPS:
{factor_graph.attribute_graph}

SAMPLING GUIDANCE:
{factor_graph.sampling_strategy}

{mode_instruction}

TASK:
Generate exactly {count} diverse entities. For each entity, create:
- A complete set of attributes (all attributes from the list above)
- Values that are realistic and internally consistent
- Follow correlations and dependencies described
- Ensure diversity across the {count} entities

Return a list of {count} dictionaries, where each dictionary contains:
- All attribute names as keys
- Appropriate values (numbers, strings, booleans as needed)

Make entities feel realistic and distinct from each other."""

        # Use MiniBatchSchema to get multiple entities at once
        class CallBatchSchema(BaseModel):
            entities: List[dict] = Field(
                description=f"List of exactly {count} entity attribute dictionaries"
            )

        try:
            result = await entity_router.ai(prompt, schema=CallBatchSchema)

            # Convert to EntityProfile objects
            profiles = []
            for i, entity_attrs in enumerate(result.entities):
                entity_id = f"E_{start_id + start + i:06d}"

                # Generate a quick summary for each entity
                attrs_str = ", ".join(
                    [f"{k}={v}" for k, v in list(entity_attrs.items())[:5]]
                )
                summary = f"{scenario_analysis.entity_type.title()} with {attrs_str}..."

                profile = EntityProfile(
                    entity_id=entity_id,
                    attributes=entity_attrs,
                    profile_summary=summary,
                )
                profiles.append(profile)

            return profiles
        except Exception as e:
            print(f"⚠️  Failed to generate mini-batch {call_num}: {str(e)[:100]}")
            # Return empty list on failure - will be filtered out later
            return []

    # Parallelize the mini-batch calls
    import asyncio

    tasks = [generate_mini_batch(i) for i in range(num_calls)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results and filter out exceptions
    all_entities = []
    for i, batch_result in enumerate(results):
        if isinstance(batch_result, Exception):
            print(f"⚠️  Exception in entity batch {i}: {str(batch_result)[:100]}")
        elif isinstance(batch_result, list):
            all_entities.extend(batch_result)
        else:
            print(f"⚠️  Unexpected result type in batch {i}: {type(batch_result)}")

    if len(all_entities) < batch_size:
        print(
            f"⚠️  Warning: Only generated {len(all_entities)}/{batch_size} entities due to errors"
        )

    return all_entities[:batch_size]  # Trim to exact size
