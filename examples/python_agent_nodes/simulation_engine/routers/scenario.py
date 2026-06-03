"""Scenario analysis router for simulation engine."""

from typing import List

from agentfield import AgentRouter

from schemas import FactorGraph, ScenarioAnalysis

scenario_router = AgentRouter(prefix="scenario")


@scenario_router.reasoner()
async def decompose_scenario(
    scenario: str, context: List[str] = []
) -> ScenarioAnalysis:
    """
    Analyzes the scenario to understand what we're simulating.
    Returns entity type, decision type, and deep analysis.
    """
    context_str = (
        "\n".join([f"- {c}" for c in context])
        if context
        else "No additional context provided."
    )

    prompt = f"""You are analyzing a simulation scenario to understand what needs to be modeled.

SCENARIO:
{scenario}

CONTEXT:
{context_str}

TASK:
Analyze this scenario deeply and provide:

1. entity_type: What type of entity/person are we simulating? (e.g., "customer", "voter", "employee", "consumer")

2. decision_type: What kind of decision are they making?
   - "binary_choice" (yes/no, stay/leave)
   - "multi_option" (choose from several options)
   - "continuous_value" (a number or amount)

3. decision_options: List all possible decisions/outcomes the entity could make. Be specific and exhaustive.

4. analysis: Write a comprehensive analysis (3-4 paragraphs) covering:
   - What are the key factors that would influence this decision?
   - What causal relationships exist? (e.g., "income affects price sensitivity")
   - What attributes of the entity would matter most?
   - What psychological, economic, or social dynamics are at play?
   - Are there different segments/archetypes of entities we should consider?
   - What hidden variables or second-order effects might exist?

Be thorough in your analysis - this will guide the entire simulation.

5. key_attributes: Identify the top 5-7 attributes that will MOST influence this decision.
   These should be the most predictive factors. Examples: income, price_sensitivity, tenure, loyalty, alternatives.
   Return as a list of attribute names (e.g., ["price_sensitivity", "income", "tenure", "loyalty", "alternatives"])."""

    result = await scenario_router.ai(prompt, schema=ScenarioAnalysis)

    # If key_attributes not provided, use a default set based on common patterns
    if not result.key_attributes:
        # Default key attributes for common scenarios
        if "price" in scenario.lower() or "cost" in scenario.lower():
            result.key_attributes = [
                "price_sensitivity",
                "income",
                "budget_constraint",
                "perceived_value",
                "alternatives",
            ]
        elif "upgrade" in scenario.lower() or "switch" in scenario.lower():
            result.key_attributes = [
                "loyalty",
                "tenure",
                "satisfaction",
                "alternatives",
                "switching_cost",
            ]
        else:
            # Generic defaults
            result.key_attributes = [
                "loyalty",
                "satisfaction",
                "alternatives",
                "tenure",
                "value",
            ]

    return result


@scenario_router.reasoner()
async def generate_factor_graph(
    scenario: str, scenario_analysis: ScenarioAnalysis, context: List[str] = []
) -> FactorGraph:
    """
    Creates the factor graph: what attributes matter and how they relate.
    """
    context_str = (
        "\n".join([f"- {c}" for c in context])
        if context
        else "No additional context provided."
    )

    prompt = f"""You are designing the factor graph for a simulation.

SCENARIO:
{scenario}

CONTEXT:
{context_str}

PREVIOUS ANALYSIS:
Entity Type: {scenario_analysis.entity_type}
Decision Type: {scenario_analysis.decision_type}
Possible Decisions: {', '.join(scenario_analysis.decision_options)}

Key Insights from Analysis:
{scenario_analysis.analysis}

TASK:
Design the factor graph that defines what attributes each {scenario_analysis.entity_type} should have.

1. attributes: Create a dictionary of all relevant attributes. For each attribute, provide a clear description.
   Include attributes across these categories:
   - Demographic (age, location, income, etc.)
   - Behavioral (usage patterns, preferences, history)
   - Psychographic (values, attitudes, personality traits)
   - Contextual (external factors, constraints, alternatives available)

   Keep attribute names simple and lowercase (e.g., "age", "income_level", "price_sensitivity")
   Make descriptions clear and specific.

2. attribute_graph: Write a detailed explanation (2-3 paragraphs) of:
   - How attributes influence each other (correlations and dependencies)
   - How attributes influence the final decision
   - What are strong vs weak predictors
   - Any interaction effects (e.g., "age matters more for low-income entities")
   - Which attributes cluster together to form natural segments

3. sampling_strategy: Describe how to sample these attributes to create realistic entities:
   - What are typical ranges/distributions for each attribute?
   - Which attributes are correlated and should be sampled together?
   - Are there natural segments/archetypes we should ensure are represented?
   - What makes a "realistic" vs "unrealistic" combination of attributes?

Be specific and detailed - this defines the entire simulation space."""

    result = await scenario_router.ai(prompt, schema=FactorGraph)

    return result
