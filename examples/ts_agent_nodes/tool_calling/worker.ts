/**
 * Worker Agent - Provides utility skills for the orchestrator to discover and call.
 *
 * Registers:
 * - get_weather (skill): Returns mock weather data for a city
 * - calculate (skill): Performs basic math operations
 * - summarize (reasoner): Uses AI to summarize text
 *
 * Start this agent first, then run the orchestrator.
 */

import { Agent } from '@agentfield/sdk';

const app = new Agent({
  nodeId: 'utility-worker-ts',
  agentFieldUrl: process.env.AGENTFIELD_URL ?? 'http://localhost:8080',
  port: 8003,
  aiConfig: {
    provider: 'openrouter',
    model: process.env.MODEL ?? 'openai/gpt-4o-mini',
    apiKey: process.env.OPENROUTER_API_KEY,
    temperature: 0.3,
  },
});

// ============= SKILLS (DETERMINISTIC) =============

app.skill('get_weather', async (ctx) => {
  const input = ctx.input as { city: string };
  const weatherData: Record<string, { temp_f: number; conditions: string; humidity: number }> = {
    'new york': { temp_f: 72, conditions: 'Partly cloudy', humidity: 65 },
    'london': { temp_f: 58, conditions: 'Overcast', humidity: 80 },
    'tokyo': { temp_f: 81, conditions: 'Sunny', humidity: 55 },
    'paris': { temp_f: 64, conditions: 'Light rain', humidity: 75 },
    'sydney': { temp_f: 68, conditions: 'Clear', humidity: 50 },
  };

  const key = input.city.toLowerCase().trim();
  const data = weatherData[key] ?? { temp_f: 70, conditions: 'Clear', humidity: 60 };
  return { city: input.city, ...data };
}, {
  tags: ['weather'],
  description: 'Get the current weather for a city. Returns temperature, conditions, and humidity.',
  inputSchema: {
    type: 'object',
    properties: {
      city: { type: 'string', description: 'The city to get weather for' },
    },
    required: ['city'],
  },
});

app.skill('calculate', async (ctx) => {
  const input = ctx.input as { operation: string; a: number; b: number };
  const ops: Record<string, number> = {
    add: input.a + input.b,
    subtract: input.a - input.b,
    multiply: input.a * input.b,
    divide: input.b !== 0 ? input.a / input.b : Infinity,
  };

  const result = ops[input.operation.toLowerCase()];
  if (result === undefined) {
    return { error: `Unknown operation: ${input.operation}. Use: add, subtract, multiply, divide` };
  }
  return { operation: input.operation, a: input.a, b: input.b, result };
}, {
  tags: ['math'],
  description: 'Perform a basic math operation. Supports: add, subtract, multiply, divide.',
  inputSchema: {
    type: 'object',
    properties: {
      operation: { type: 'string', description: 'Math operation: add, subtract, multiply, divide' },
      a: { type: 'number', description: 'First operand' },
      b: { type: 'number', description: 'Second operand' },
    },
    required: ['operation', 'a', 'b'],
  },
});

// ============= REASONER (AI-POWERED) =============

app.reasoner('summarize', async (ctx) => {
  const text = ctx.input?.text ?? ctx.input;
  const result = await ctx.ai(
    `Summarize this in 1-2 sentences:\n\n${text}`,
    { system: 'You are a concise summarizer. Respond with only the summary, no preamble.' }
  );
  return { summary: result };
}, {
  tags: ['text'],
  description: 'Use AI to create a concise summary of the given text.',
  inputSchema: {
    type: 'object',
    properties: {
      text: { type: 'string', description: 'The text to summarize' },
    },
    required: ['text'],
  },
});

// ============= START =============

console.log('Worker Agent (TypeScript) starting...');
console.log('  Node: utility-worker-ts');
console.log('  Skills: get_weather, calculate');
console.log('  Reasoners: summarize');

app.serve();
