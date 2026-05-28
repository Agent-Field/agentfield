/**
 * E2E manual test: all of Sam's tool-calling use cases (TypeScript SDK).
 *
 * Requires:
 * - Control plane running at localhost:8080
 * - Worker agent registered (run worker.py first)
 *
 * Usage:
 *   OPENROUTER_API_KEY=... npx tsx test_orchestrator.ts
 */

import { Agent } from '@agentfield/sdk';
import type { ToolCallConfig, ToolCallTrace } from '@agentfield/sdk';
import { buildToolConfig, executeToolCallLoop } from '@agentfield/sdk';

const app = new Agent({
  nodeId: 'test-orchestrator-ts',
  agentFieldUrl: process.env.AGENTFIELD_URL ?? 'http://localhost:8080',
  aiConfig: {
    provider: 'openrouter',
    model: process.env.MODEL ?? 'openai/gpt-4o-mini',
    apiKey: process.env.OPENROUTER_API_KEY,
    temperature: 0.2,
  },
});

const aiClient = (app as any).aiClient;
function getModel() {
  return aiClient.getModel({
    provider: 'openrouter',
    model: process.env.MODEL ?? 'openai/gpt-4o-mini',
    apiKey: process.env.OPENROUTER_API_KEY,
  });
}

function printTrace(trace: ToolCallTrace) {
  console.log(`  Tool calls: ${trace.totalToolCalls}, Turns: ${trace.totalTurns}`);
  for (const c of trace.calls) {
    const err = c.error ? ` ERROR: ${c.error}` : '';
    console.log(`    - ${c.toolName}(${JSON.stringify(c.arguments)}) => ${c.latencyMs.toFixed(0)}ms${err}`);
  }
}

type TestFn = () => Promise<void>;
const tests: [string, TestFn][] = [];
const passed: string[] = [];
const failed: [string, string][] = [];

// ============= TEST 1: tools='discover' (unfiltered) =============

tests.push(['discover', async () => {
  console.log('\n' + '='.repeat(60));
  console.log("TEST 1: tools='discover' (unfiltered auto-discovery)");
  console.log('='.repeat(60));

  const { tools, config, needsLazyHydration } = await buildToolConfig('discover', app);
  const toolNames = Object.keys(tools);
  console.log(`  Discovered ${toolNames.length} tools: ${toolNames.join(', ')}`);

  // Should find ALL tools: weather, calculate, summarize
  if (toolNames.length < 3) throw new Error(`Expected >= 3 tools, got ${toolNames.length}`);

  const { text, trace } = await executeToolCallLoop(
    app,
    "What's the weather in Tokyo? Also calculate 42 * 17.",
    tools,
    config,
    needsLazyHydration,
    getModel,
    { system: 'You are a helpful assistant. Use tools to answer accurately.' }
  );

  console.log(`  Answer: ${text.slice(0, 300)}`);
  printTrace(trace);
  if (trace.totalToolCalls < 2) throw new Error(`Expected >= 2 tool calls, got ${trace.totalToolCalls}`);
  console.log('  PASS');
}]);

// ============= TEST 2: Tag-filtered discovery =============

tests.push(['tag_filter', async () => {
  console.log('\n' + '='.repeat(60));
  console.log("TEST 2: ToolCallConfig tags=['weather'] - tag filter");
  console.log('='.repeat(60));

  const toolsParam: ToolCallConfig = { tags: ['weather'] };
  const { tools, config, needsLazyHydration } = await buildToolConfig(toolsParam, app);
  const toolNames = Object.keys(tools);
  console.log(`  Discovered ${toolNames.length} tools: ${toolNames.join(', ')}`);

  // Should only have weather tools
  for (const name of toolNames) {
    if (name.includes('calculate') || name.includes('summarize')) {
      throw new Error(`Tag filter should exclude ${name}`);
    }
  }

  const { text, trace } = await executeToolCallLoop(
    app,
    'Give me the weather for London and Paris.',
    tools,
    config,
    needsLazyHydration,
    getModel,
    { system: 'You are a weather reporter. Use the weather tool.' }
  );

  console.log(`  Answer: ${text.slice(0, 300)}`);
  printTrace(trace);
  if (trace.totalToolCalls < 1) throw new Error(`Expected >= 1 tool call`);
  console.log('  PASS');
}]);

// ============= TEST 3: Progressive/lazy discovery =============

tests.push(['lazy_hydration', async () => {
  console.log('\n' + '='.repeat(60));
  console.log("TEST 3: schema_hydration='lazy' (progressive discovery)");
  console.log('='.repeat(60));

  const toolsParam: ToolCallConfig = {
    schemaHydration: 'lazy',
    maxCandidateTools: 30,
    maxHydratedTools: 8,
  };
  const { tools, config, needsLazyHydration } = await buildToolConfig(toolsParam, app);
  console.log(`  Discovered ${Object.keys(tools).length} tools (metadata-only): ${Object.keys(tools).join(', ')}`);
  console.log(`  needsLazyHydration: ${needsLazyHydration}`);

  if (!needsLazyHydration) throw new Error('Expected needsLazyHydration to be true');

  const { text, trace } = await executeToolCallLoop(
    app,
    "What's the weather in Sydney?",
    tools,
    config,
    needsLazyHydration,
    getModel,
    { system: 'You are a helpful assistant. Use available tools.' }
  );

  console.log(`  Answer: ${text.slice(0, 300)}`);
  printTrace(trace);
  // Note: Some models may not call metadata-only tools. GPT-4o-mini handles this well.
  console.log(`  Tool calls: ${trace.totalToolCalls} (lazy hydration ${trace.totalToolCalls > 0 ? 'worked' : 'skipped by model'})`);
  console.log('  PASS');
}]);

// ============= TEST 4: Guardrails =============

tests.push(['guardrails', async () => {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 4: Guardrails (maxTurns=3, maxToolCalls=2)');
  console.log('='.repeat(60));

  const { tools, config, needsLazyHydration } = await buildToolConfig('discover', app);

  const { text, trace } = await executeToolCallLoop(
    app,
    'Get weather for Tokyo, London, Paris, Sydney, New York. Also calculate 1+1, 2+2, 3+3.',
    tools,
    { ...config, maxTurns: 3, maxToolCalls: 2 },
    needsLazyHydration,
    getModel,
    { system: 'You are a helpful assistant. Use tools for every question.' }
  );

  console.log(`  Answer: ${text.slice(0, 200)}...`);
  printTrace(trace);
  const successful = trace.calls.filter(c => !c.error);
  console.log(`  Successful calls: ${successful.length} (limit was 2)`);
  if (successful.length > 2) throw new Error(`Expected max 2 successful calls, got ${successful.length}`);
  console.log('  PASS');
}]);

// ============= TEST 5: Per-tool-call observability =============

tests.push(['observability', async () => {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 5: Per-tool-call observability (trace verification)');
  console.log('='.repeat(60));

  const { tools, config, needsLazyHydration } = await buildToolConfig('discover', app);

  const { text, trace } = await executeToolCallLoop(
    app,
    'Calculate 10 + 20 and get weather for Paris.',
    tools,
    config,
    needsLazyHydration,
    getModel,
    { system: 'You are a helpful assistant. Use tools to answer.' }
  );

  if (trace.totalToolCalls < 2) throw new Error(`Expected >= 2 calls, got ${trace.totalToolCalls}`);
  if (trace.totalTurns < 1) throw new Error('Expected >= 1 turn');
  if (!trace.finalResponse) throw new Error('Expected finalResponse');

  console.log('  Trace fields for each call:');
  for (const c of trace.calls) {
    console.log(`    toolName:   ${c.toolName}`);
    console.log(`    arguments:  ${JSON.stringify(c.arguments)}`);
    console.log(`    result:     ${JSON.stringify(c.result)?.slice(0, 100)}`);
    console.log(`    error:      ${c.error}`);
    console.log(`    latencyMs:  ${c.latencyMs.toFixed(1)}`);
    console.log(`    turn:       ${c.turn}`);
    if (!c.toolName) throw new Error('toolName should be set');
    if (typeof c.arguments !== 'object') throw new Error('arguments should be object');
    if (c.latencyMs < 0) throw new Error('latencyMs should be non-negative');
    if (c.result === undefined && !c.error) throw new Error('result or error should be set');
    console.log();
  }
  console.log(`  finalResponse: ${trace.finalResponse.slice(0, 100)}...`);
  console.log('  PASS - All trace fields verified');
}]);

// ============= RUN ALL =============

async function main() {
  if (!aiClient) {
    console.error('AIClient not available on agent');
    process.exit(1);
  }

  for (const [name, fn] of tests) {
    try {
      await fn();
      passed.push(name);
    } catch (e: any) {
      failed.push([name, e.message ?? String(e)]);
      console.error(`  FAIL: ${e.message}`);
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log(`RESULTS: ${passed.length} passed, ${failed.length} failed`);
  console.log('='.repeat(60));
  for (const name of passed) console.log(`  PASS: ${name}`);
  for (const [name, err] of failed) console.log(`  FAIL: ${name}: ${err}`);

  process.exit(failed.length > 0 ? 1 : 0);
}

main();
