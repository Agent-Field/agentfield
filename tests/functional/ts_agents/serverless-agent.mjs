import http from 'node:http';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

async function loadSdk() {
  const base = process.env.TS_SDK_PATH;
  const candidates = [
    base && path.join(base, 'dist', 'index.js'),
    '/usr/local/lib/node_modules/@agentfield/sdk/dist/index.js',
    '/usr/lib/node_modules/@agentfield/sdk/dist/index.js'
  ].filter(Boolean);

  for (const candidate of candidates) {
    try {
      return await import(pathToFileURL(candidate).href);
    } catch {
      // try next candidate
    }
  }

  return await import('@agentfield/sdk');
}

const { Agent } = await loadSdk();

const agentFieldUrl = process.env.AGENTFIELD_SERVER ?? 'http://localhost:8080';
const nodeId = process.env.TS_AGENT_ID ?? 'ts-serverless-agent';
const port = Number(process.env.TS_AGENT_PORT ?? 8097);
const host = process.env.TS_AGENT_BIND_HOST ?? '0.0.0.0';

const agent = new Agent({
  nodeId,
  agentFieldUrl,
  deploymentType: 'serverless',
  devMode: true
});

agent.reasoner('hello', async (ctx) => ({
  greeting: `Hello, ${ctx.input.name ?? 'AgentField'}!`,
  runId: ctx.runId,
  executionId: ctx.executionId,
  parentExecutionId: ctx.parentExecutionId
}));

agent.reasoner('relay', async (ctx) => {
  const target = process.env.TS_CHILD_TARGET ?? ctx.input.target;
  if (!target) {
    return { error: 'target is required' };
  }
  const downstream = await agent.call(target, { name: ctx.input.name ?? 'ts-child' });
  return { target, downstream };
});

const handler = agent.handler();

const server = http.createServer(async (req, res) => {
  try {
    await handler(req, res);
  } catch (err) {
    res.statusCode = 500;
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({ error: err?.message ?? 'handler failure' }));
  }
});

server.listen(port, host, () => {
  console.log(`Serverless TS handler listening on http://${host}:${port}`);
});
