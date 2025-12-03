import { Agent } from '@agentfield/sdk';
import dotenv from 'dotenv';

dotenv.config();

const agent = new Agent({
  nodeId: process.env.AGENT_NODE_ID ?? 'ts-serverless-hello',
  version: '1.0.0',
  deploymentType: 'serverless',
  agentFieldUrl: process.env.AGENTFIELD_URL ?? 'http://localhost:8080',
  devMode: true
});

agent.reasoner('hello', async (ctx) => ({
  greeting: `Hello, ${ctx.input.name ?? 'AgentField'}!`,
  runId: ctx.runId,
  executionId: ctx.executionId
}));

agent.reasoner('relay', async (ctx) => {
  const target = (process.env.CHILD_TARGET ?? ctx.input.target) as string | undefined;
  if (!target) {
    return { error: 'target is required' };
  }

  const downstream = await agent.call(target, { message: ctx.input.message ?? 'ping' });
  return { target, downstream };
});

// Exported handler works for AWS Lambda/Cloud Functions and raw HTTP (Vercel/Netlify).
export const handler = agent.handler();

// Optional local runner for smoke-testing without heartbeats.
if (import.meta.url === `file://${process.argv[1]}`) {
  const port = Number(process.env.PORT ?? 8787);
  const { default: express } = await import('express');

  const app = express();
  app.all('*', async (req, res) => {
    await handler(req, res);
  });

  app.listen(port, '0.0.0.0', () => {
    console.log(`Serverless hello handler listening on http://localhost:${port}`);
  });
}
