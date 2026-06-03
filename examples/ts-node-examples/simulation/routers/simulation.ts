import { AgentRouter } from '@agentfield/sdk';
import {
  EntityDecisionSchema,
  EntityProfileSchema,
  FactorGraphSchema,
  RunSimulationInputSchema,
  ScenarioAnalysisSchema,
  SimulationResultSchema,
  type EntityDecision,
  type EntityProfile,
  type FactorGraph,
  type RunSimulationInput,
  type ScenarioAnalysis,
  type SimulationResult
} from '../schemas.js';

export const simulationRouter = new AgentRouter({ prefix: 'simulation' });

const scenarioTargets = {
  decompose: 'scenario_decomposeScenario',
  factorGraph: 'scenario_generateFactorGraph'
} as const;

const entityTargets = {
  batch: 'entity_generateEntityBatch'
} as const;

const decisionTargets = {
  batch: 'decision_simulateBatchDecisions'
} as const;

const aggregationTargets = {
  aggregate: 'aggregation_aggregateAndAnalyze'
} as const;

simulationRouter.reasoner<RunSimulationInput, SimulationResult>(
  'runSimulation',
  async (ctx) => {
    const scenario = ctx.input.scenario;
    const populationSize = ctx.input.populationSize;
    const context = ctx.input.context ?? [];
    const parallelBatchSize = ctx.input.parallelBatchSize ?? 20;
    const explorationRatio = ctx.input.explorationRatio ?? 0.1;

    const assertArray = (value: any, label: string) => {
      if (!Array.isArray(value)) {
        throw new Error(`${label} is not an array: ${JSON.stringify(value)}`);
      }
    };

    const logAndThrow = (stage: string, err: unknown) => {
      console.error(`❌ ${stage} failed`, err);
      throw err;
    };

    console.log(`🚀 Starting simulation: ${populationSize} entities`);

    console.log('\n📋 Phase 1: Analyzing scenario...');
    const scenarioAnalysis = (await ctx
      .call(scenarioTargets.decompose, {
        scenario,
        context
      })
      .catch((err) => logAndThrow('Scenario analysis', err))) as ScenarioAnalysis;
    if (!scenarioAnalysis?.decisionOptions) {
      logAndThrow('Scenario analysis output validation', new Error('decisionOptions missing'));
    }
    console.log(`   Entity type: ${scenarioAnalysis.entityType}`);
    console.log(`   Decision type: ${scenarioAnalysis.decisionType}`);
    console.log(`   Options: ${scenarioAnalysis.decisionOptions.join(', ')}`);

    console.log('\n🕸️  Phase 2: Building factor graph...');
    const factorGraph = (await ctx
      .call(scenarioTargets.factorGraph, {
        scenario,
        scenarioAnalysis,
        context
      })
      .catch((err) => logAndThrow('Factor graph', err))) as FactorGraph;
    if (!factorGraph?.attributes) {
      logAndThrow('Factor graph output validation', new Error('attributes missing'));
    }
    console.log(`   Tracking ${Object.keys(factorGraph.attributes).length} attributes`);

    console.log(`\n👥 Phase 3: Generating ${populationSize} entities...`);
    const entitiesPerBatch = 20;
    const allEntities: EntityProfile[] = [];
    const numEntityBatches = Math.ceil(populationSize / entitiesPerBatch);
    for (let batchNum = 0; batchNum < numEntityBatches; batchNum++) {
      const startId = batchNum * entitiesPerBatch;
      const batchSize = Math.min(entitiesPerBatch, populationSize - startId);
      console.log(`   Batch ${batchNum + 1}/${numEntityBatches}: Generating ${batchSize} entities...`);
      const batch = (await ctx
        .call(entityTargets.batch, {
          startId,
          batchSize,
          scenarioAnalysis,
          factorGraph,
          explorationRatio
        })
        .catch((err) => logAndThrow(`Entity batch ${batchNum + 1}`, err))) as EntityProfile[];
      assertArray(batch, `entity batch ${batchNum + 1}`);
      allEntities.push(...batch);
    }
    console.log(`   ✅ Generated ${allEntities.length} entities`);

    console.log('\n🎯 Phase 4: Simulating decisions...');
    const allDecisions = (await ctx
      .call(decisionTargets.batch, {
        entities: allEntities,
        scenario,
        scenarioAnalysis,
        context,
        parallelBatchSize
      })
      .catch((err) => logAndThrow('Decision batch', err))) as EntityDecision[];
    assertArray(allDecisions, 'decision batch');
    console.log(`   ✅ Simulated ${allDecisions.length} decisions`);

    console.log('\n📊 Phase 5: Aggregating results and generating insights...');
    const insights = (await ctx
      .call(aggregationTargets.aggregate, {
        scenario,
        scenarioAnalysis,
        factorGraph,
        entities: allEntities,
        decisions: allDecisions,
        context
      })
      .catch((err) => logAndThrow('Aggregation', err))) as SimulationResult['insights'];

    const insightsForLog =
      insights && typeof insights === 'object'
        ? insights
        : {
            outcomeDistribution: {},
            keyInsight: 'Fallback insight',
            detailedAnalysis: 'Fallback analysis due to invalid model response.',
            segmentPatterns: 'Fallback segment patterns.',
            causalDrivers: 'Fallback causal drivers.'
          };
    console.log(`\n🎯 KEY INSIGHT: ${insightsForLog.keyInsight}`);
    console.log('\n📈 OUTCOME DISTRIBUTION:');
    Object.entries(insightsForLog.outcomeDistribution).forEach(([decision, pct]) => {
      console.log(`   ${decision}: ${(pct * 100).toFixed(1)}%`);
    });

    return {
      scenario,
      context,
      populationSize,
      scenarioAnalysis,
      factorGraph,
      entities: allEntities,
      decisions: allDecisions,
      insights: insightsForLog
    };
  },
  { inputSchema: RunSimulationInputSchema, outputSchema: SimulationResultSchema }
);
