import dotenv from 'dotenv';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Agent, type AIConfig } from '@agentfield/sdk';
import {
  aggregationRouter,
  decisionRouter,
  entityRouter,
  scenarioRouter,
  simulationRouter
} from './routers/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: resolve(process.cwd(), '.env') });
if (!process.env.OPENAI_API_KEY && !process.env.OPENROUTER_API_KEY) {
  dotenv.config({ path: resolve(__dirname, '../../../.env') });
}

const aiConfig: AIConfig =
  process.env.OPENROUTER_API_KEY
    ? {
        // OpenRouter is OpenAI-compatible; keep provider as openai
        provider: 'openai',
        model: process.env.OPENROUTER_MODEL ?? 'deepseek/deepseek-v3.1-terminus',
        apiKey: process.env.OPENROUTER_API_KEY,
        baseUrl: process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1'
      }
    : {
        provider: 'openai',
        model: process.env.OPENAI_MODEL ?? 'gpt-4o',
        apiKey: process.env.OPENAI_API_KEY
      };

const agent = new Agent({
  nodeId: 'simulation-engine',
  aiConfig,
  host: 'localhost',
  devMode: true
});

[scenarioRouter, entityRouter, decisionRouter, aggregationRouter, simulationRouter].forEach((router) =>
  agent.includeRouter(router)
);

agent.reasoner<{ message: string }, { echo: string }>('echo', async (ctx) => ({
  echo: ctx.input.message
}));

agent
  .serve()
  .then(() => {
    console.log('Simulation agent serving on port 8001');
  })
  .catch((err) => {
    console.error('Failed to start simulation agent', err);
    process.exit(1);
  });
