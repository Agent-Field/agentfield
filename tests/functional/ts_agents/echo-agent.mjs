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
const nodeId = process.env.TS_AGENT_ID ?? 'ts-functional-agent';
const port = Number(process.env.TS_AGENT_PORT ?? 8099);
const host = process.env.TS_AGENT_BIND_HOST ?? '0.0.0.0';
const publicUrl =
  process.env.TS_AGENT_PUBLIC_URL ?? `http://${process.env.TEST_AGENT_CALLBACK_HOST ?? 'localhost'}:${port}`;

const agent = new Agent({
  nodeId,
  port,
  host,
  publicUrl,
  agentFieldUrl,
  heartbeatIntervalMs: 1000,
  devMode: false
});

agent.reasoner('echo', async (ctx) => ({
  echoed: ctx.input.message,
  runId: ctx.runId,
  workflowId: ctx.workflowId
}));

agent.skill('upper', (ctx) => ({
  value: String(ctx.input.text ?? '').toUpperCase()
}));

const shutdown = async () => {
  await agent.shutdown();
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

await agent.serve();
// Keep process alive; Agent handles requests once started.
await new Promise(() => {});
