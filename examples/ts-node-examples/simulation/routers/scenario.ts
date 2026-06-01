import { z } from 'zod';
import { AgentRouter } from '@agentfield/sdk';
import {
  FactorGraphSchema,
  ScenarioAnalysisSchema,
  ScenarioInputSchema,
  type FactorGraph,
  type ScenarioAnalysis,
  type ScenarioInput
} from '../schemas.js';
import { formatContext, parseWithSchema } from '../utils.js';

export const scenarioRouter = new AgentRouter({ prefix: 'scenario' });

scenarioRouter.reasoner<ScenarioInput, ScenarioAnalysis>(
  'decomposeScenario',
  async (ctx) => {
    const { scenario, context = [] } = ctx.input;
    const fallbackAnalysis = (): ScenarioAnalysis => ({
      entityType: 'customer',
      decisionType: 'multi_option',
      decisionOptions: ['option_a', 'option_b'],
      analysis: 'Fallback analysis used due to invalid model response.',
      keyAttributes: []
    });

    const prompt = `You are analyzing a simulation scenario to understand what needs to be modeled.

SCENARIO:
${scenario}

CONTEXT:
${formatContext(context)}

TASK:
Analyze this scenario deeply and provide:

1. entityType: What type of entity/person are we simulating? (e.g., "customer", "voter", "employee", "consumer")

2. decisionType: What kind of decision are they making?
   - "binary_choice" (yes/no, stay/leave)
   - "multi_option" (choose from several options)
   - "continuous_value" (a number or amount)

3. decisionOptions: List all possible decisions/outcomes the entity could make. Be specific and exhaustive.

4. analysis: Write a comprehensive analysis (3-4 paragraphs) covering:
   - What are the key factors that would influence this decision?
   - What causal relationships exist? (e.g., "income affects price sensitivity")
   - What attributes of the entity would matter most?
   - What psychological, economic, or social dynamics are at play?
   - Are there different segments/archetypes of entities we should consider?
   - What hidden variables or second-order effects might exist?

Be thorough in your analysis - this will guide the entire simulation.

5. keyAttributes: Identify the top 5-7 attributes that will MOST influence this decision.
   These should be the most predictive factors. Examples: income, price_sensitivity, tenure, loyalty, alternatives.
   Return as a list of attribute names (e.g., ["price_sensitivity", "income", "tenure", "loyalty", "alternatives"]).

RESPONSE FORMAT: Return ONLY valid JSON matching the schema. No prose, no markdown.`;

    // Lower temperature for more consistent structured output
    const rawAnalysis = await ctx.ai(prompt, { schema: ScenarioAnalysisSchema, temperature: 0.4 });
    const analysis =
      parseWithSchema(rawAnalysis, ScenarioAnalysisSchema, 'Scenario analysis', fallbackAnalysis, true) ??
      fallbackAnalysis();

    if (!Array.isArray(analysis.keyAttributes) || analysis.keyAttributes.length === 0) {
      const scenarioText = scenario.toLowerCase();
      if (scenarioText.includes('price') || scenarioText.includes('cost')) {
        analysis.keyAttributes = ['price_sensitivity', 'income', 'budget_constraint', 'perceived_value', 'alternatives'];
      } else if (scenarioText.includes('upgrade') || scenarioText.includes('switch')) {
        analysis.keyAttributes = ['loyalty', 'tenure', 'satisfaction', 'alternatives', 'switching_cost'];
      } else {
        analysis.keyAttributes = ['loyalty', 'satisfaction', 'alternatives', 'tenure', 'value'];
      }
    }

    return analysis;
  },
  { inputSchema: ScenarioInputSchema, outputSchema: ScenarioAnalysisSchema }
);

scenarioRouter.reasoner(
  'generateFactorGraph',
  async (ctx) => {
    const { scenario, scenarioAnalysis, context = [] } = ctx.input as {
      scenario: string;
      scenarioAnalysis: ScenarioAnalysis;
      context?: string[];
    };

    const prompt = `You are designing the factor graph for a simulation.

SCENARIO:
${scenario}

CONTEXT:
${formatContext(context)}

PREVIOUS ANALYSIS:
Entity Type: ${scenarioAnalysis.entityType}
Decision Type: ${scenarioAnalysis.decisionType}
Possible Decisions: ${scenarioAnalysis.decisionOptions.join(', ')}

Key Insights from Analysis:
${scenarioAnalysis.analysis}

TASK:
Design the factor graph that defines what attributes each ${scenarioAnalysis.entityType} should have.

1. attributes: Create a dictionary of all relevant attributes. For each attribute, provide a clear description.
   Include attributes across these categories:
   - Demographic (age, location, income, etc.)
   - Behavioral (usage patterns, preferences, history)
   - Psychographic (values, attitudes, personality traits)
   - Contextual (external factors, constraints, alternatives available)

   Keep attribute names simple and lowercase (e.g., "age", "income_level", "price_sensitivity")
   Make descriptions clear and specific.

2. attributeGraph: Write a detailed explanation (2-3 paragraphs) of:
   - How attributes influence each other (correlations and dependencies)
   - How attributes influence the final decision
   - What are strong vs weak predictors
   - Any interaction effects (e.g., "age matters more for low-income entities")
   - Which attributes cluster together to form natural segments

3. samplingStrategy: Describe how to sample these attributes to create realistic entities:
   - What are typical ranges/distributions for each attribute?
   - Which attributes are correlated and should be sampled together?
   - Are there natural segments/archetypes we should ensure are represented?
   - What makes a "realistic" vs "unrealistic" combination of attributes?

Be specific and detailed - this defines the entire simulation space.

RESPONSE FORMAT: Return ONLY valid JSON matching the schema. No prose, no markdown.
Example:
{
  "attributes": {
    "age": "Age in years",
    "mars_tenure": "Years on Mars",
    "professional_role": "Job/role in the colony",
    "psychological_resilience": "0-1 resilience score",
    "risk_tolerance": "0-1 risk tolerance",
    "trust_in_leadership": "0-1 trust score",
    "community_orientation": "0-1 orientation to community",
    "hoarding_tendency": "0-1 tendency to hoard",
    "resource_usage_efficiency": "0-1 efficiency score"
  },
  "attributeGraph": "How these attributes interact...",
  "samplingStrategy": "How to sample realistic values..."
}`;

    // Lower temperature for more consistent structured output
    const factorGraph = await ctx.ai<FactorGraph>(prompt, { schema: FactorGraphSchema, temperature: 0.4 });
    return factorGraph;
  },
  {
    inputSchema: z.object({
      scenario: z.string(),
      scenarioAnalysis: ScenarioAnalysisSchema,
      context: z.array(z.string()).optional()
    }),
    outputSchema: FactorGraphSchema
  }
);
