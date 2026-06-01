let Agent;

try {
  ({ Agent } = await import('@agentfield/sdk'));
} catch {
  ({ Agent } = await import(new URL('../../../../sdk/typescript/dist/index.js', import.meta.url)));
}

const agentFieldUrl = process.env.AGENTFIELD_SERVER ?? 'http://localhost:8080';
const nodeId = process.env.TS_AGENT_ID ?? 'demo-ts-logs';
const port = Number(process.env.TS_AGENT_PORT ?? 8001);
const host = process.env.TS_AGENT_BIND_HOST ?? '0.0.0.0';
const publicUrl =
  process.env.TS_AGENT_PUBLIC_URL ?? `http://localhost:${port}`;

const agent = new Agent({
  nodeId,
  port,
  host,
  publicUrl,
  agentFieldUrl,
  heartbeatIntervalMs: 2000,
  devMode: true,
  didEnabled: false
});

agent.reasoner('ping', async () => ({ ok: true }));

let tick = 0;
setInterval(() => {
  console.log(`[${nodeId}] demo stdout tick ${tick}`);
  console.error(`[${nodeId}] demo stderr tick ${tick}`);
  tick += 1;
}, 3000);

const shutdown = async () => {
  await agent.shutdown();
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

await agent.serve();
await new Promise(() => {});
