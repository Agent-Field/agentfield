"""Pydantic schemas for the simulation engine."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScenarioAnalysis(BaseModel):
    """Simple schema for scenario decomposition"""

    entity_type: str = Field(
        description="Type of entity being simulated (e.g., 'customer', 'voter', 'employee')"
    )
    decision_type: str = Field(
        description="Type of decision (e.g., 'binary_choice', 'multi_option', 'continuous_value')"
    )
    decision_options: List[str] = Field(
        description="List of possible decisions/outcomes"
    )
    analysis: str = Field(
        description="Detailed analysis of the scenario including key factors, causal relationships, and what matters"
    )
    key_attributes: List[str] = Field(
        default=[],
        description="Top 5-7 attributes that matter most for this decision (identified from analysis)",
    )


class FactorGraph(BaseModel):
    """Schema for the causal attribute graph"""

    attributes: Dict[str, str] = Field(
        description="Dictionary of attribute_name: description. Each attribute that matters for this entity."
    )
    attribute_graph: str = Field(
        description="Detailed description of how attributes relate to each other and to the decision, including correlations, dependencies, and causal chains"
    )
    sampling_strategy: str = Field(
        description="Description of how to sample these attributes to get realistic, diverse entities"
    )


class EntityProfile(BaseModel):
    """Schema for a single entity's attributes"""

    entity_id: str
    attributes: Dict[str, Any] = Field(
        description="Dictionary of attribute_name: value for this entity"
    )
    profile_summary: str = Field(
        description="2-3 sentence human-readable summary of who this entity is"
    )


class EntityBatch(BaseModel):
    """Schema for generating multiple entities at once"""

    entities: List[Dict[str, Any]] = Field(
        description="List of entity attribute dictionaries"
    )


class EntityDecision(BaseModel):
    """Schema for entity's decision - simplified to avoid JSON parsing issues"""

    entity_id: str
    decision: str = Field(
        description="The chosen decision/action from the available options"
    )
    confidence: float = Field(description="Confidence in this decision, 0.0 to 1.0")
    key_factor: str = Field(
        description="Single most important attribute that influenced this decision (max 50 words)"
    )
    trade_off: str = Field(description="Main trade-off considered (max 50 words)")
    reasoning: Optional[str] = Field(
        default="",
        description="Brief explanation (1-2 sentences, max 100 words) - optional",
    )


class SimulationInsights(BaseModel):
    """Schema for final simulation results"""

    outcome_distribution: Dict[str, float] = Field(
        description="Percentage for each decision option"
    )
    key_insight: str = Field(
        description="One sentence summary of the most important finding"
    )
    detailed_analysis: str = Field(
        description="Comprehensive analysis (4-5 paragraphs) covering: overall patterns, segment differences, causal drivers, surprising findings, and implications"
    )
    segment_patterns: str = Field(
        description="Description of how different types of entities decided differently, organized by meaningful segments"
    )
    causal_drivers: str = Field(
        description="Analysis of which attributes most strongly predicted decisions, with specific examples and correlations"
    )


class SimulationResult(BaseModel):
    """Complete simulation result"""

    scenario: str
    context: List[str]
    population_size: int
    scenario_analysis: ScenarioAnalysis
    factor_graph: FactorGraph
    entities: List[EntityProfile]
    decisions: List[EntityDecision]
    insights: SimulationInsights
