import { z } from 'zod';
import { AgentRouter } from '@agentfield/sdk';
import {
  EntityProfileSchema,
  FactorGraphSchema,
  ScenarioAnalysisSchema,
  type EntityProfile,
  type FactorGraph,
  type ScenarioAnalysis
} from '../schemas.js';
import { flattenAttributes, parseWithSchema } from '../utils.js';

export const entityRouter = new AgentRouter({ prefix: 'entity' });

entityRouter.reasoner(
  'generateEntityBatch',
  async (ctx) => {
    const { startId, batchSize, scenarioAnalysis, factorGraph, explorationRatio = 0.1 } = ctx.input as {
      startId: number;
      batchSize: number;
      scenarioAnalysis: ScenarioAnalysis;
      factorGraph: FactorGraph;
      explorationRatio?: number;
    };

    const entitiesPerCall = 5;
    const numCalls = Math.ceil(batchSize / entitiesPerCall);

    const generateMiniBatch = async (callNum: number) => {
      const start = callNum * entitiesPerCall;
      const count = Math.min(entitiesPerCall, batchSize - start);
      const explorationMode = start < Math.floor(batchSize * explorationRatio);

      const modeInstruction = explorationMode
        ? `EXPLORATION MODE: Generate entities with unusual or edge-case attributes.
Sample from distribution tails or create surprising but realistic combinations.`
        : `STANDARD MODE: Generate typical, realistic entities following
normal distributions and common attribute combinations.`;

      const prompt = `Generate ${count} synthetic ${scenarioAnalysis.entityType} entities for simulation.

AVAILABLE ATTRIBUTES:
${JSON.stringify(factorGraph.attributes, null, 2)}

ATTRIBUTE RELATIONSHIPS:
${factorGraph.attributeGraph}

SAMPLING GUIDANCE:
${factorGraph.samplingStrategy}

${modeInstruction}

TASK:
Generate exactly ${count} diverse entities. For each entity, create:
- A complete set of attributes (all attributes from the list above)
- Values that are realistic and internally consistent
- Follow correlations and dependencies described
- Ensure diversity across the ${count} entities

Return a list of ${count} dictionaries, where each dictionary contains:
- All attribute names as keys
- Appropriate values (numbers, strings, booleans as needed)

Make entities feel realistic and distinct from each other.

RESPONSE FORMAT: Return ONLY a JSON array of objects (no wrapper), length ${count}, where each object has all attributes. No prose, no markdown, no code fences.`;

      // Use lenient schema: accept 1+ items (model may return fewer than requested)
      const CallBatchSchema = z.array(z.record(z.any())).min(1);

      try {
        // Lower temperature for more consistent structured output
        const raw = (await ctx.ai(prompt, { schema: CallBatchSchema, temperature: 0.4 })) as unknown;
        // Accept either an array of entities or an object with { entities }
        const parsedArray =
          parseWithSchema(raw, CallBatchSchema, 'Entity mini-batch', () => [], false) ??
          (Array.isArray((raw as any)?.entities) ? ((raw as any).entities as unknown[]) : []);

        if (!Array.isArray(parsedArray) || parsedArray.length === 0) {
          throw new Error('Entity mini-batch schema validation failed');
        }

        return parsedArray.map((entityAttrs, idx) => {
          const flatAttrs = flattenAttributes(entityAttrs as Record<string, any>);
          const entityId = `E_${(startId + start + idx).toString().padStart(6, '0')}`;
          const summaryAttributes = Object.entries(flatAttrs)
            .slice(0, 5)
            .map(([k, v]) => `${k}=${v}`)
            .join(', ');

          return {
            entityId,
            attributes: flatAttrs,
            profileSummary: `${scenarioAnalysis.entityType} with ${summaryAttributes}...`
          } satisfies EntityProfile;
        });
      } catch (err) {
        console.warn(`⚠️  Failed to generate mini-batch ${callNum}: ${String(err).slice(0, 120)}`);
        return [] as EntityProfile[];
      }
    };

    const miniBatches = await Promise.all(Array.from({ length: numCalls }, (_, i) => generateMiniBatch(i)));
    return miniBatches.flat().slice(0, batchSize);
  },
  {
    inputSchema: z.object({
      startId: z.number(),
      batchSize: z.number(),
      scenarioAnalysis: ScenarioAnalysisSchema,
      factorGraph: FactorGraphSchema,
      explorationRatio: z.number().optional()
    }),
    outputSchema: z.array(EntityProfileSchema)
  }
);
