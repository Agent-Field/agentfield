/**
 * Orchestrator Agent - Demonstrates the tool-calling pipeline.
 *
 * This agent uses `ctx.ai(prompt, { tools: ... })` to automatically:
 * 1. Discover available tools from the control plane
 * 2. Present them to the LLM as callable functions
 * 3. Let the LLM decide which tools to call
 * 4. Dispatch tool calls via agent.call() through the control plane
 * 5. Feed results back to the LLM
 * 6. Repeat until the LLM produces a final answer
 *
 * Requires:
 * - Control plane running at localhost:8080
 * - Worker agent registered (run worker.ts first)
 *
 * Examples show:
 * - tools: "discover" (simple auto-discovery)
 * - tools: ToolCallConfig (with filtering, progressive discovery, guardrails)
 */

import { Agent } from '@agentfield/sdk';
import type { ToolCallConfig } from '@agentfield/sdk';

const app = new Agent({
  nodeId: 'orchestrator-ts',
  agentFieldUrl: process.env.AGENTFIELD_URL ?? 'http://localhost:8080',
  port: 8004,
  aiConfig: {
    provider: 'openrouter',
    model: process.env.MODEL ?? 'openai/gpt-4o-mini',
    apiKey: process.env.OPENROUTER_API_KEY,
    temperature: 0.2,
  },
});

// ============= EXAMPLE 1: Simple discover-all =============

app.reasoner('ask_with_tools', async (ctx) => {
  const question = ctx.input?.question ?? ctx.input;

  const { text, trace } = await ctx.aiWithTools(
    `${question}`,
    {
      tools: 'discover',
      system: 'You are a helpful assistant. Use the available tools to answer the user\'s question accurately.',
    }
  );

  console.log(`  Tool calls made: ${trace.totalToolCalls}`);
  console.log(`  LLM turns: ${trace.totalTurns}`);
  for (const call of trace.calls) {
    console.log(`    - ${call.toolName}(${JSON.stringify(call.arguments)}) => ${call.latencyMs.toFixed(0)}ms`);
  }

  return { answer: text, trace };
}, {
  tags: ['demo'],
  description: 'Answer a question using auto-discovered tools.',
  inputSchema: {
    type: 'object',
    properties: {
      question: { type: 'string', description: 'The question to answer' },
    },
    required: ['question'],
  },
});

// ============= EXAMPLE 2: Filtered discovery with tags =============

app.reasoner('weather_report', async (ctx) => {
  const cities = ctx.input?.cities ?? ctx.input;

  const { text, trace } = await ctx.aiWithTools(
    `What's the weather like in: ${cities}?`,
    {
      tools: { tags: ['weather'] } satisfies ToolCallConfig,
      system: 'You are a weather reporter. Get the weather for each city and provide a brief report.',
    }
  );

  return { report: text, toolCalls: trace.totalToolCalls };
}, {
  tags: ['demo'],
  description: 'Get weather for specific cities using tag-filtered discovery.',
  inputSchema: {
    type: 'object',
    properties: {
      cities: { type: 'string', description: 'Comma-separated list of cities' },
    },
    required: ['cities'],
  },
});

// ============= EXAMPLE 3: Progressive/lazy discovery =============

app.reasoner('smart_query', async (ctx) => {
  const question = ctx.input?.question ?? ctx.input;

  const { text, trace } = await ctx.aiWithTools(
    question,
    {
      tools: {
        schemaHydration: 'lazy',
        maxCandidateTools: 30,
        maxHydratedTools: 8,
      } satisfies ToolCallConfig,
      system: 'You are a helpful assistant with access to tools. Use them when needed.',
    }
  );

  return { answer: text, trace };
}, {
  tags: ['demo'],
  description: 'Answer using progressive discovery (lazy schema hydration).',
});

// ============= EXAMPLE 4: With guardrails =============

app.reasoner('guarded_query', async (ctx) => {
  const question = ctx.input?.question ?? ctx.input;

  const { text, trace } = await ctx.aiWithTools(
    question,
    {
      tools: 'discover',
      system: 'You are a helpful assistant. Be efficient with tool usage.',
      maxTurns: 3,
      maxToolCalls: 5,
    }
  );

  return { answer: text, trace };
}, {
  tags: ['demo'],
  description: 'Answer with strict guardrails on tool usage.',
});

// ============= START =============

console.log('Orchestrator Agent (TypeScript) starting...');
console.log('  Node: orchestrator-ts');
console.log('  Reasoners: ask_with_tools, weather_report, smart_query, guarded_query');

app.run();
