"""Aggregation and analysis router for simulation engine."""

import json
import random
from collections import defaultdict
from typing import List

from agentfield import AgentRouter

from schemas import (
    EntityDecision,
    EntityProfile,
    FactorGraph,
    ScenarioAnalysis,
    SimulationInsights,
)

aggregation_router = AgentRouter(prefix="aggregation")


@aggregation_router.reasoner()
async def aggregate_and_analyze(
    scenario: str,
    scenario_analysis: ScenarioAnalysis,
    factor_graph: FactorGraph,
    entities: List[EntityProfile],
    decisions: List[EntityDecision],
    context: List[str] = [],
) -> SimulationInsights:
    """
    Only pass intelligent summaries to AI, not all raw data.
    Pre-compute statistics, create attribute distributions, and sample representative examples.
    """
    # Compute basic statistics (no AI needed)
    total = len(decisions)
    decision_counts = {}
    confidence_by_decision = {}

    for d in decisions:
        decision_counts[d.decision] = decision_counts.get(d.decision, 0) + 1
        if d.decision not in confidence_by_decision:
            confidence_by_decision[d.decision] = []
        confidence_by_decision[d.decision].append(d.confidence)

    outcome_dist = {k: v / total for k, v in decision_counts.items()}
    avg_confidence = {k: sum(v) / len(v) for k, v in confidence_by_decision.items()}

    # Create intelligent summaries instead of passing all data

    # 1. Attribute distribution summaries (for each attribute, show value frequencies)
    attribute_summaries = {}
    for attr_name in factor_graph.attributes.keys():
        # Get all values for this attribute
        attr_values = [
            e.attributes.get(attr_name) for e in entities if attr_name in e.attributes
        ]

        # Count frequencies
        value_counts = defaultdict(int)
        for val in attr_values:
            if val is not None:
                # Convert to string for counting, handle different types
                val_str = str(val)
                value_counts[val_str] += 1

        # Create summary (top 5 most common values)
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        top_values = sorted_values[:5]
        total_with_attr = len(attr_values)

        if total_with_attr > 0:
            summary_parts = [
                f"{val} ({count}/{total_with_attr}, {count*100/total_with_attr:.1f}%)"
                for val, count in top_values
            ]
            attribute_summaries[attr_name] = {
                "distribution": ", ".join(summary_parts),
                "total": total_with_attr,
            }

    # 2. Decision patterns by attribute (which attributes correlate with which decisions)
    decision_by_attribute = defaultdict(lambda: defaultdict(int))
    for entity, decision in zip(entities, decisions):
        for attr_name, attr_value in entity.attributes.items():
            if attr_value is not None:
                attr_str = str(attr_value)
                decision_by_attribute[attr_name][(attr_str, decision.decision)] += 1

    # Create summary of strongest correlations
    attribute_decision_patterns = {}
    for attr_name in factor_graph.attributes.keys():
        if attr_name in decision_by_attribute:
            patterns = decision_by_attribute[attr_name]
            # Find the strongest pattern for this attribute
            if patterns:
                top_pattern = max(patterns.items(), key=lambda x: x[1])
                (attr_val, decision_type), count = top_pattern
                total_for_attr = sum(patterns.values())
                attribute_decision_patterns[attr_name] = (
                    f"When {attr_name}={attr_val}: {count}/{total_for_attr} chose '{decision_type}' "
                    f"({count*100/total_for_attr:.1f}%)"
                )

    # 3. Sample representative examples intelligently
    sample_size = min(30, len(entities))  # Max 30 examples to AI
    samples_per_decision = max(3, sample_size // len(decision_counts))

    sampled_examples = []
    for decision_type in decision_counts.keys():
        # Get entities that made this decision
        matching_decisions = [d for d in decisions if d.decision == decision_type]
        matching_entity_ids = {d.entity_id for d in matching_decisions}
        matching_entities = {
            e.entity_id: e for e in entities if e.entity_id in matching_entity_ids
        }

        # Sample some of them
        sample_count = min(samples_per_decision, len(matching_entities))
        if sample_count > 0:
            sampled_ids = random.sample(list(matching_entities.keys()), sample_count)

            for entity_id in sampled_ids:
                entity = matching_entities[entity_id]
                decision = next(
                    d for d in matching_decisions if d.entity_id == entity_id
                )
                sampled_examples.append(
                    {
                        "attributes": entity.attributes,
                        "decision": decision.decision,
                        "key_factor": decision.key_factor,
                        "trade_off": decision.trade_off,
                        "reasoning": decision.reasoning or "",
                        "confidence": decision.confidence,
                    }
                )

    # 4. Create segment summaries (group by common attribute combinations)
    # Find entities with similar attribute patterns
    segment_examples = []
    # Group by 2-3 key attributes to create segments
    key_attributes = list(factor_graph.attributes.keys())[:3]  # Top 3 attributes

    if key_attributes:
        segment_groups = defaultdict(list)
        for entity, decision in zip(entities, decisions):
            # Create a segment key from top attributes
            segment_key = tuple(
                str(entity.attributes.get(attr, "unknown")) for attr in key_attributes
            )
            segment_groups[segment_key].append((entity, decision))

        # Get one example from each major segment
        for segment_key, group in list(segment_groups.items())[:10]:  # Top 10 segments
            if group:
                entity, decision = group[0]
                segment_examples.append(
                    {
                        "segment": f"{', '.join(f'{k}={v}' for k, v in zip(key_attributes, segment_key))}",
                        "count": len(group),
                        "example_decision": decision.decision,
                        "example_attributes": entity.attributes,
                    }
                )

    # Prepare context
    context_str = "\n".join([f"- {c}" for c in context]) if context else ""

    # Send intelligent summaries, not raw data!
    attribute_summaries_str = json.dumps(attribute_summaries, indent=2)
    attribute_patterns_str = "\n".join(
        [f"  • {k}: {v}" for k, v in list(attribute_decision_patterns.items())[:10]]
    )
    segment_summaries_str = json.dumps(segment_examples, indent=2)
    sampled_examples_str = json.dumps(sampled_examples, indent=2)

    if context:
        context_block = "CONTEXT:\n" + context_str + "\n\n"
    else:
        context_block = ""

    prompt = f"""Analyze simulation results from {total} {scenario_analysis.entity_type} entities.

SCENARIO:
{scenario}

{context_block}ATTRIBUTES TRACKED:
{', '.join(factor_graph.attributes.keys())}

OUTCOME DISTRIBUTION (from {total} entities):
{json.dumps(outcome_dist, indent=2)}

AVERAGE CONFIDENCE BY DECISION:
{json.dumps(avg_confidence, indent=2)}

ATTRIBUTE DISTRIBUTIONS (summary of value frequencies):
{attribute_summaries_str}

STRONGEST ATTRIBUTE-DECISION PATTERNS:
{attribute_patterns_str}

SEGMENT SUMMARIES (grouped by key attributes):
{segment_summaries_str}

REPRESENTATIVE EXAMPLES ({len(sampled_examples)} of {total} entities):
{sampled_examples_str}

TASK:
Analyze these results and provide insights:

1. outcome_distribution: Return this exact dictionary: {outcome_dist}

2. key_insight: ONE sentence capturing the most important finding.

3. detailed_analysis: 4-5 paragraphs covering:
   - Overall pattern and dominant outcome
   - Distinct segments and their behaviors
   - Key drivers (which attributes predicted decisions)
   - Surprising or counterintuitive findings
   - Implications and recommendations

4. segment_patterns: 2-3 paragraphs analyzing how different entity types decided:
   - Group entities by meaningful attribute combinations
   - Describe each segment's typical decision and why
   - Note certainty levels by segment
   - Identify interesting edge cases

5. causal_drivers: 2-3 paragraphs on attribute influence:
   - Which attributes most strongly influenced decisions
   - Specific examples from the data
   - Interaction effects between attributes
   - Rank drivers by importance

Be specific and reference the summarized data provided."""

    result = await aggregation_router.ai(prompt, schema=SimulationInsights)

    # Override with our precise computed values
    result.outcome_distribution = outcome_dist

    return result
