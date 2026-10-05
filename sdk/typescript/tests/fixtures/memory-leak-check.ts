import { Agent } from '../../src/agent/Agent.js';

if (!global.gc) throw new Error('Leak measurement requires --expose-gc');

function createAndReleaseAgents(): void {
  for (let cycle = 0; cycle < 10; cycle++) {
    const agents: Agent[] = [];
    for (let i = 0; i < 50; i++) {
      const agent = new Agent({ nodeId: `leak-test-${cycle}-${i}`, devMode: true });
      agent.reasoner('test', async () => ({ ok: true }));
      agents.push(agent);
    }
    agents.length = 0;
  }
}

global.gc();
const before = process.memoryUsage().heapUsed;
createAndReleaseAgents();
await new Promise<void>(resolve => setImmediate(resolve));
global.gc();
const leakMB = (process.memoryUsage().heapUsed - before) / 1024 / 1024;
process.stdout.write(JSON.stringify({ gcExposed: true, agentCount: 500, leakMB }));
