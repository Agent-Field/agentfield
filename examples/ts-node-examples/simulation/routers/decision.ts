import { z } from 'zod';
import { AgentRouter } from '@agentfield/sdk';
import {
  EntityDecisionSchema,
  EntityProfileSchema,
  ScenarioAnalysisSchema,
  type EntityDecision,
  type EntityProfile,
  type ScenarioAnalysis
} from '../schemas.js';
import { formatContext } from '../utils.js';

export const decisionRouter = new AgentRouter({ prefix: 'decision' });

decisionRouter.reasoner(
  'simulateEntityDecision',
  async (ctx) => {
    const { entity, scenario, scenarioAnalysis, context = [] } = ctx.input as {
      entity: EntityProfile;
      scenario: string;
      scenarioAnalysis: ScenarioAnalysis;
      context?: string[];
    };

    const keyAttrs = scenarioAnalysis.keyAttributes.slice(0, 7);
    const keyAttributesStr = keyAttrs
      .filter((attr) => entity.attributes[attr] !== undefined)
      .map((attr) => `  • ${attr}: ${entity.attributes[attr]}`)
      .join('\n');
    const contextSection = context.length ? `ADDITIONAL CONTEXT:\n${formatContext(context)}\n\n` : '';

    const prompt = `You are simulating the decision-making of a specific ${scenarioAnalysis.entityType}.

WHO YOU ARE:
${entity.profileSummary}

KEY ATTRIBUTES (most relevant for this decision):
${keyAttributesStr}

SCENARIO YOU'RE FACING:
${scenario}

${contextSection}AVAILABLE DECISIONS:
${scenarioAnalysis.decisionOptions.join(', ')}

TASK:
Based on who you are, decide how you would respond to this scenario.

1. decision: Choose one option from the available decisions list.
2. confidence: Rate confidence 0.0-1.0. How certain are you?
3. keyFactor: What single attribute influenced this decision most? (max 50 words)
4. tradeOff: What was the main trade-off you considered? (max 50 words)
5. reasoning: Optional brief explanation (1-2 sentences, max 100 words).

Be concise and realistic.

RESPONSE FORMAT: Return ONLY valid JSON with fields decision, confidence, keyFactor, tradeOff, reasoning. Do not include entityId. No prose, no markdown.`;

    const DecisionWithoutIdSchema = EntityDecisionSchema.omit({ entityId: true });

    try {
      // Lower temperature for more consistent structured output
      const decision = await ctx.ai<Omit<EntityDecision, 'entityId'>>(prompt, {
        schema: DecisionWithoutIdSchema,
        temperature: 0.4
      });
      return { ...decision, entityId: entity.entityId };
    } catch (err) {
      console.warn(`⚠️  Failed entity ${entity.entityId}: ${String(err).slice(0, 120)}`);
      return {
        entityId: entity.entityId,
        decision: scenarioAnalysis.decisionOptions[0] ?? 'unknown',
        confidence: 0,
        keyFactor: 'Error during decision generation',
        tradeOff: 'Unable to evaluate',
        reasoning: 'Failed to generate decision'
      } satisfies EntityDecision;
    }
  },
  {
    inputSchema: z.object({
      entity: EntityProfileSchema,
      scenario: z.string(),
      scenarioAnalysis: ScenarioAnalysisSchema,
      context: z.array(z.string()).optional()
    }),
    outputSchema: EntityDecisionSchema
  }
);

decisionRouter.reasoner(
  'simulateBatchDecisions',
  async (ctx) => {
    const {
      entities,
      scenario,
      scenarioAnalysis,
      context = [],
      parallelBatchSize = 20
    } = ctx.input as {
      entities: EntityProfile[];
      scenario: string;
      scenarioAnalysis: ScenarioAnalysis;
      context?: string[];
      parallelBatchSize?: number;
    };

    const allDecisions: EntityDecision[] = [];
    for (let i = 0; i < entities.length; i += parallelBatchSize) {
      const batch = entities.slice(i, i + parallelBatchSize);
      const tasks = batch.map((entity) =>
        ctx.call('decision_simulateEntityDecision', {
          entity,
          scenario,
          scenarioAnalysis,
          context
        }) as Promise<EntityDecision>
      );

      const batchResults = await Promise.allSettled(tasks);
      batchResults.forEach((res) => {
        if (res.status === 'fulfilled') {
          allDecisions.push(res.value);
        } else {
          console.warn(`⚠️  Exception in batch: ${String(res.reason).slice(0, 120)}`);
        }
      });
    }

    return allDecisions;
  },
  {
    inputSchema: z.object({
      entities: z.array(EntityProfileSchema),
      scenario: z.string(),
      scenarioAnalysis: ScenarioAnalysisSchema,
      context: z.array(z.string()).optional(),
      parallelBatchSize: z.number().optional()
    }),
    outputSchema: z.array(EntityDecisionSchema)
  }
);
