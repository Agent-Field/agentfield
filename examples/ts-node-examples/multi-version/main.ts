/**
 * Multi-Version Agent Example
 *
 * Demonstrates multi-version agent support using the composite primary key
 * (id, version). All agents share the same nodeId but register with different
 * versions, creating separate rows in the control plane.
 *
 * The execute endpoint transparently routes across versioned agents using
 * weighted round-robin when no default (unversioned) agent exists.
 *
 * Usage:
 *   # Start control plane first, then:
 *   cd examples/ts-node-examples && npm run dev:multi-version
 */
import 'dotenv/config';
import { Agent } from '@agentfield/sdk';

const CP_URL = process.env.AGENTFIELD_URL ?? 'http://localhost:8080';
const AGENT_ID = 'mv-demo';
const BASE_PORT = 9100;

interface VersionSpec {
  version: string;
  port: number;
}

const versions: VersionSpec[] = [
  { version: '1.0.0', port: BASE_PORT },
  { version: '2.0.0', port: BASE_PORT + 1 },
];

function createAgent(spec: VersionSpec): Agent {
  const agent = new Agent({
    nodeId: AGENT_ID,
    version: spec.version,
    agentFieldUrl: CP_URL,
    port: spec.port,
    devMode: true,
  });

  // Echo reasoner present on every version
  agent.reasoner('echo', async (ctx) => ({
    agent: AGENT_ID,
    version: spec.version,
    echoed: ctx.input.message ?? '',
  }));

  // v2 has an extra capability
  if (spec.version === '2.0.0') {
    agent.reasoner('v2_feature', async (ctx) => ({
      agent: AGENT_ID,
      version: spec.version,
      feature: 'Only available in v2',
      input: ctx.input,
    }));
  }

  return agent;
}

async function validateRegistration() {
  console.log('\n--- Validating multi-version registration ---\n');

  // List all nodes and check that both versions are registered
  const nodesRes = await fetch(`${CP_URL}/api/v1/nodes?show_all=true`);
  if (!nodesRes.ok) {
    console.error(`Failed to list nodes: ${nodesRes.status} ${nodesRes.statusText}`);
    return;
  }
  const nodesData = await nodesRes.json();
  const agentNodes = (nodesData.nodes ?? nodesData.agents ?? []).filter(
    (n: any) => (n.id ?? n.node_id) === AGENT_ID
  );
  console.log(`[Nodes] Found ${agentNodes.length} versions of "${AGENT_ID}":`);
  for (const n of agentNodes) {
    console.log(`  - id=${n.id ?? n.node_id}, version=${n.version}, base_url=${n.base_url}`);
  }

  // Execute against the shared ID — the CP will route via round-robin
  console.log('\n[Execute] Sending requests to mv-demo.echo:');
  for (let i = 0; i < 4; i++) {
    try {
      const res = await fetch(`${CP_URL}/api/v1/execute/${AGENT_ID}.echo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: { message: `request-${i}` } }),
      });
      const routedVersion = res.headers.get('X-Routed-Version') ?? '(default)';
      const data = await res.json();
      console.log(`  Request ${i}: routed to version=${routedVersion}, result=`, data.result);
    } catch (err) {
      console.error(`  Request ${i}: failed`, err);
    }
  }

  console.log('\n--- Validation complete ---\n');
}

async function main() {
  console.log(`Multi-version agent example`);
  console.log(`  Control plane: ${CP_URL}`);
  console.log(`  Agent ID:      ${AGENT_ID}`);
  console.log(`  Versions:      ${versions.map((v) => `${v.version}@:${v.port}`).join(', ')}\n`);

  // Create and start all agents
  const agents = versions.map(createAgent);

  for (const agent of agents) {
    await agent.serve();
    console.log(`  Started ${agent.config.nodeId} v${(agent.config as any).version} on port ${agent.config.port}`);
  }

  // Give the CP a moment to process registrations
  await new Promise((resolve) => setTimeout(resolve, 1500));

  // Validate
  await validateRegistration();

  // Keep running so heartbeats continue
  console.log('All agents running. Press Ctrl+C to stop.\n');
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
