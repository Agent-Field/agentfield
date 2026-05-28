import 'dotenv/config';
import { Agent } from '@agentfield/sdk';
import { reasonersRouter } from './reasoners.js';

/**
 * Verifiable Credentials Example
 *
 * This example demonstrates how to use DID (Decentralized Identifiers) and
 * Verifiable Credentials (VCs) in AgentField to create cryptographically
 * verifiable audit trails for agent executions.
 *
 * Each reasoner in this example generates a VC that:
 * - Records input/output hashes for tamper detection
 * - Links to caller and target DIDs for accountability
 * - Is signed by the control plane's issuer DID
 * - Can be verified independently for compliance/audit
 *
 * Prerequisites:
 * 1. Control plane running with DID enabled (features.did.enabled: true)
 * 2. Keystore directory exists (./data/keys on control plane)
 *
 * Usage:
 *   pnpm dev:vc
 *
 * Then test with:
 *   curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_process \
 *     -H "Content-Type: application/json" \
 *     -d '{"input": {"text": "Hello, Verifiable World!"}}'
 */

async function main() {
  const agent = new Agent({
    nodeId: process.env.AGENT_ID ?? 'vc-demo',
    agentFieldUrl: process.env.AGENTFIELD_URL ?? 'http://localhost:8080',
    port: Number(process.env.PORT ?? 8006),
    version: '1.0.0',
    devMode: true,

    // DID/VC is enabled by default, but we explicitly set it here for clarity
    didEnabled: true,

    // Optional: AI config for the AI-powered reasoner
    aiConfig: {
      provider: 'openai',
      model: 'gpt-4o-mini',
      apiKey: process.env.OPENAI_API_KEY,
    },
  });

  // Include the VC-enabled reasoners
  agent.includeRouter(reasonersRouter);

  await agent.serve();

  console.log(`
╔════════════════════════════════════════════════════════════════════╗
║           Verifiable Credentials Demo Agent Started                ║
╠════════════════════════════════════════════════════════════════════╣
║  Agent ID:     ${agent.config.nodeId.padEnd(50)}║
║  Port:         ${String(agent.config.port).padEnd(50)}║
║  DID Enabled:  ${String(agent.config.didEnabled).padEnd(50)}║
╠════════════════════════════════════════════════════════════════════╣
║  Available Reasoners:                                              ║
║  • vc_process      - Basic processing with VC generation           ║
║  • vc_analyze      - AI analysis with VC audit trail               ║
║  • vc_transform    - Data transformation with VC proof             ║
║  • vc_chain        - Multi-step workflow with chained VCs          ║
╠════════════════════════════════════════════════════════════════════╣
║  Test Commands:                                                    ║
║                                                                    ║
║  # Basic VC generation:                                            ║
║  curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_process \\
║    -H "Content-Type: application/json" \\                          ║
║    -d '{"input": {"text": "Hello World"}}'                         ║
║                                                                    ║
║  # Check workflow in UI for VC badge (green checkmark)             ║
╚════════════════════════════════════════════════════════════════════╝
`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error('Failed to start agent:', err);
    process.exit(1);
  });
}
