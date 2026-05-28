import { z } from 'zod';
import { AgentRouter } from '@agentfield/sdk';
import {
  EntityDecisionSchema,
  EntityProfileSchema,
  FactorGraphSchema,
  ScenarioAnalysisSchema,
  SimulationInsightsSchema,
  type EntityDecision,
  type EntityProfile,
  type FactorGraph,
  type ScenarioAnalysis,
  type SimulationInsights
} from '../schemas.js';
import { formatContext, parseWithSchema } from '../utils.js';

export const aggregationRouter = new AgentRouter({ prefix: 'aggregation' });

aggregationRouter.reasoner(
  'aggregateAndAnalyze',
  async (ctx) => {
    const { scenario, scenarioAnalysis, factorGraph, entities, decisions, context = [] } = ctx.input as {
      scenario: string;
      scenarioAnalysis: ScenarioAnalysis;
      factorGraph: FactorGraph;
      entities: EntityProfile[];
      decisions: EntityDecision[];
      context?: string[];
    };

    const total = decisions.length || 1;
    const decisionCounts: Record<string, number> = {};
    const confidenceByDecision: Record<string, number[]> = {};

    decisions.forEach((decision) => {
      decisionCounts[decision.decision] = (decisionCounts[decision.decision] ?? 0) + 1;
      confidenceByDecision[decision.decision] = confidenceByDecision[decision.decision] ?? [];
      confidenceByDecision[decision.decision].push(decision.confidence);
    });

    const outcomeDistribution = Object.fromEntries(
      Object.entries(decisionCounts).map(([k, v]) => [k, v / total])
    );
    const avgConfidence = Object.fromEntries(
      Object.entries(confidenceByDecision).map(([k, v]) => [k, v.reduce((a, b) => a + b, 0) / v.length])
    );

    const attributeSummaries: Record<string, { distribution: string; total: number }> = {};
    Object.keys(factorGraph.attributes).forEach((attrName) => {
      const values = entities
        .map((e) => e.attributes[attrName])
        .filter((val) => val !== undefined && val !== null)
        .map((val) => String(val));

      const freq: Record<string, number> = {};
      values.forEach((val) => {
        freq[val] = (freq[val] ?? 0) + 1;
      });
      const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 5);
      const summaryParts = sorted.map(
        ([val, count]) => `${val} (${count}/${values.length}, ${(count * 100) / values.length}%)`
      );
      if (values.length > 0) {
        attributeSummaries[attrName] = { distribution: summaryParts.join(', '), total: values.length };
      }
    });

    const decisionByAttribute: Record<string, Record<string, number>> = {};
    entities.forEach((entity) => {
      const decision = decisions.find((d) => d.entityId === entity.entityId);
      if (!decision) return;
      Object.entries(entity.attributes).forEach(([attrName, attrValue]) => {
        const key = `${String(attrValue)}|${decision.decision}`;
        decisionByAttribute[attrName] = decisionByAttribute[attrName] ?? {};
        decisionByAttribute[attrName][key] = (decisionByAttribute[attrName][key] ?? 0) + 1;
      });
    });

    const attributeDecisionPatterns: Record<string, string> = {};
    Object.entries(decisionByAttribute).forEach(([attrName, patterns]) => {
      const entries = Object.entries(patterns);
      if (entries.length === 0) return;
      const [topKey, count] = entries.sort((a, b) => b[1] - a[1])[0];
      const [attrVal, decisionType] = topKey.split('|');
      const totalForAttr = entries.reduce((acc, [, v]) => acc + v, 0);
      attributeDecisionPatterns[attrName] = `When ${attrName}=${attrVal}: ${count}/${totalForAttr} chose '${decisionType}' (${(
        (count * 100) /
        totalForAttr
      ).toFixed(1)}%)`;
    });

    const sampleSize = Math.min(30, entities.length);
    const decisionsByType = decisions.reduce<Record<string, EntityDecision[]>>((acc, decision) => {
      acc[decision.decision] = acc[decision.decision] ?? [];
      acc[decision.decision].push(decision);
      return acc;
    }, {});

    const sampledExamples: Array<Record<string, unknown>> = [];
    Object.entries(decisionsByType).forEach(([decisionType, decisionList]) => {
      const matchingEntities = decisionList
        .map((d) => entities.find((e) => e.entityId === d.entityId))
        .filter((e): e is EntityProfile => Boolean(e));

      const sampleCount = Math.min(
        Math.max(3, Math.floor(sampleSize / Math.max(1, Object.keys(decisionsByType).length))),
        matchingEntities.length
      );
      for (let i = 0; i < sampleCount; i++) {
        const idx = Math.floor(Math.random() * matchingEntities.length);
        const entity = matchingEntities[idx];
        const decision = decisionList.find((d) => d.entityId === entity.entityId)!;
        sampledExamples.push({
          attributes: entity.attributes,
          decision: decision.decision,
          keyFactor: decision.keyFactor,
          tradeOff: decision.tradeOff,
          reasoning: decision.reasoning ?? '',
          confidence: decision.confidence
        });
      }
    });

    const keyAttributes = Object.keys(factorGraph.attributes).slice(0, 3);
    const segmentGroups: Record<string, { entity: EntityProfile; decision: EntityDecision }[]> = {};
    if (keyAttributes.length) {
      entities.forEach((entity) => {
        const decision = decisions.find((d) => d.entityId === entity.entityId);
        if (!decision) return;
        const segmentKey = keyAttributes.map((attr) => String(entity.attributes[attr] ?? 'unknown')).join('|');
        segmentGroups[segmentKey] = segmentGroups[segmentKey] ?? [];
        segmentGroups[segmentKey].push({ entity, decision });
      });
    }

    const segmentExamples = Object.entries(segmentGroups)
      .slice(0, 10)
      .map(([segmentKey, group]) => ({
        segment: keyAttributes.map((attr, idx) => `${attr}=${segmentKey.split('|')[idx]}`).join(', '),
        count: group.length,
        exampleDecision: group[0].decision.decision,
        exampleAttributes: group[0].entity.attributes
      }));

    const prompt = `Analyze simulation results from ${total} ${scenarioAnalysis.entityType} entities.

SCENARIO:
${scenario}

${context.length ? `CONTEXT:\n${formatContext(context)}\n\n` : ''}ATTRIBUTES TRACKED:
${Object.keys(factorGraph.attributes).join(', ')}

OUTCOME DISTRIBUTION (from ${total} entities):
${JSON.stringify(outcomeDistribution, null, 2)}

AVERAGE CONFIDENCE BY DECISION:
${JSON.stringify(avgConfidence, null, 2)}

ATTRIBUTE DISTRIBUTIONS (summary of value frequencies):
${JSON.stringify(attributeSummaries, null, 2)}

STRONGEST ATTRIBUTE-DECISION PATTERNS:
${Object.entries(attributeDecisionPatterns)
      .slice(0, 10)
      .map(([k, v]) => `  • ${k}: ${v}`)
      .join('\n')}

SEGMENT SUMMARIES (grouped by key attributes):
${JSON.stringify(segmentExamples, null, 2)}

REPRESENTATIVE EXAMPLES (${sampledExamples.length} of ${total} entities):
${JSON.stringify(sampledExamples, null, 2)}

TASK:
Analyze these results and provide insights:

1. outcomeDistribution: Return this exact dictionary: ${JSON.stringify(outcomeDistribution)}
2. keyInsight: ONE sentence capturing the most important finding.
3. detailedAnalysis: 4-5 paragraphs covering:
   - Overall pattern and dominant outcome
   - Distinct segments and their behaviors
   - Key drivers (which attributes predicted decisions)
   - Surprising or counterintuitive findings
   - Implications and recommendations
4. segmentPatterns: 2-3 paragraphs analyzing how different entity types decided:
   - Group entities by meaningful attribute combinations
   - Describe each segment's typical decision and why
   - Note certainty levels by segment
   - Identify interesting edge cases
5. causalDrivers: 2-3 paragraphs on attribute influence:
   - Which attributes most strongly influenced decisions
   - Specific examples from the data
   - Interaction effects between attributes
   - Rank drivers by importance

Be specific and reference the summarized data provided.

RESPONSE FORMAT: Return ONLY valid JSON matching the schema. No prose, no markdown.`;

    // Lower temperature for more consistent structured output
    const raw = await ctx.ai(prompt, { schema: SimulationInsightsSchema, temperature: 0.4 });
    const fallback = () => ({
      outcomeDistribution,
      keyInsight: 'Fallback insight',
      detailedAnalysis: 'Fallback analysis due to invalid model response.',
      segmentPatterns: 'Fallback segment patterns.',
      causalDrivers: 'Fallback causal drivers.'
    });
    const parsed =
      parseWithSchema(raw, SimulationInsightsSchema, 'Aggregation insights', fallback, true) ?? fallback();
    const insights = typeof parsed === 'object' && parsed !== null ? parsed : fallback();

    return { ...insights, outcomeDistribution };
  },
  {
    inputSchema: z.object({
      scenario: z.string(),
      scenarioAnalysis: ScenarioAnalysisSchema,
      factorGraph: FactorGraphSchema,
      entities: z.array(EntityProfileSchema),
      decisions: z.array(EntityDecisionSchema),
      context: z.array(z.string()).optional()
    }),
    outputSchema: SimulationInsightsSchema
  }
);
