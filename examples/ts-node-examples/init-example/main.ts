import 'dotenv/config';
import { Agent } from '@agentfield/sdk';
import { reasonersRouter } from './reasoners.js';

// Catch any unhandled errors
process.on('uncaughtException', (err) => {
  console.error('[FATAL] Uncaught exception:', err);
});
process.on('unhandledRejection', (reason) => {
  console.error('[FATAL] Unhandled rejection:', reason);
});

async function main() {
  // Log SDK version and startup info
  console.log(`[DEBUG] Starting agent at ${new Date().toISOString()}`);
  console.log(`[DEBUG] AGENTFIELD_URL: ${process.env.AGENTFIELD_URL}`);
  console.log(`[DEBUG] AGENT_CALLBACK_URL: ${process.env.AGENT_CALLBACK_URL}`);

  const agent = new Agent({
    nodeId: process.env.AGENT_ID ?? "init-example",
    agentFieldUrl: process.env.AGENTFIELD_URL ?? 'http://localhost:8080',
    port: Number(process.env.PORT ?? 8005),
    publicUrl: process.env.AGENT_CALLBACK_URL,
    version: '1.0.0',
    devMode: true,
    apiKey: process.env.AGENTFIELD_API_KEY,
    aiConfig: {
      provider: 'openai',
      model: 'gpt-4o',
      apiKey: process.env.OPENAI_API_KEY,
    },
  });

  agent.includeRouter(reasonersRouter);

  await agent.serve();
  console.log(`Agent "${agent.config.nodeId}" listening on http://localhost:${agent.config.port}`);

  // Heartbeat monitoring - log every minute to confirm process is alive
  let heartbeatCount = 0;
  setInterval(() => {
    heartbeatCount++;
    console.log(`[DEBUG] Process alive - minute ${heartbeatCount} - ${new Date().toISOString()}`);
  }, 60000);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error(err);
    process.exit(1);
  });
}
