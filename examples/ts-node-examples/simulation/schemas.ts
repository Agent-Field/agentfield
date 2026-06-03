import { z } from 'zod';

// Helper: coerce string numbers to actual numbers (handles "0.8" -> 0.8)
const coercedNumber = z.coerce.number();

// Helper: coerce record values to numbers (for outcomeDistribution)
const coercedNumberRecord = z.record(z.coerce.number());

export const ScenarioInputSchema = z.object({
  scenario: z.string(),
  context: z.array(z.string()).optional()
});
export type ScenarioInput = z.infer<typeof ScenarioInputSchema>;

export const ScenarioAnalysisSchema = z.object({
  entityType: z.string(),
  decisionType: z.string(),
  decisionOptions: z.array(z.string()).min(1), // At least one option
  analysis: z.string(),
  keyAttributes: z.array(z.string()).default([]) // Default to empty if missing
});
export type ScenarioAnalysis = z.infer<typeof ScenarioAnalysisSchema>;

export const FactorGraphSchema = z.object({
  attributes: z.record(z.string()),
  attributeGraph: z.string(),
  samplingStrategy: z.string()
});
export type FactorGraph = z.infer<typeof FactorGraphSchema>;

export const EntityProfileSchema = z.object({
  entityId: z.string(),
  attributes: z.record(z.any()),
  profileSummary: z.string()
});
export type EntityProfile = z.infer<typeof EntityProfileSchema>;

export const EntityDecisionSchema = z.object({
  entityId: z.string(),
  decision: z.string(),
  confidence: coercedNumber.min(0).max(1), // Coerce "0.8" -> 0.8, clamp 0-1
  keyFactor: z.string(),
  tradeOff: z.string(),
  reasoning: z.string().optional().default('') // Default empty if missing
});
export type EntityDecision = z.infer<typeof EntityDecisionSchema>;

export const SimulationInsightsSchema = z.object({
  outcomeDistribution: coercedNumberRecord, // Coerce string numbers in record
  keyInsight: z.string(),
  detailedAnalysis: z.string(),
  segmentPatterns: z.string(),
  causalDrivers: z.string()
});
export type SimulationInsights = z.infer<typeof SimulationInsightsSchema>;

export const SimulationResultSchema = z.object({
  scenario: z.string(),
  context: z.array(z.string()),
  populationSize: z.number(),
  scenarioAnalysis: ScenarioAnalysisSchema,
  factorGraph: FactorGraphSchema,
  entities: z.array(EntityProfileSchema),
  decisions: z.array(EntityDecisionSchema),
  insights: SimulationInsightsSchema
});
export type SimulationResult = z.infer<typeof SimulationResultSchema>;

export const RunSimulationInputSchema = z.object({
  scenario: z.string(),
  populationSize: z.number().int().positive(),
  context: z.array(z.string()).default([]),
  parallelBatchSize: z.number().int().positive().default(20),
  explorationRatio: z.number().min(0).max(1).default(0.1)
});
export type RunSimulationInput = z.infer<typeof RunSimulationInputSchema>;
