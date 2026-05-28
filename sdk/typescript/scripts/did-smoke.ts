import { randomUUID } from 'node:crypto';
import { DidClient } from '../src/did/DidClient.js';

async function main() {
  const baseUrl = process.env.AGENTFIELD_URL ?? 'http://localhost:8080';
  const client = new DidClient(baseUrl);

  const executionId = randomUUID();
  const workflowId = randomUUID();

  console.log(`Using baseUrl=${baseUrl}`);
  console.log(`executionId=${executionId}`);
  console.log(`workflowId=${workflowId}`);

  const credential = await client.generateCredential({
    executionContext: {
      executionId,
      workflowId,
      sessionId: 'session-1',
      callerDid: 'did:example:caller',
      targetDid: 'did:example:target',
      agentNodeDid: 'did:example:agent',
      timestamp: new Date().toISOString()
    },
    inputData: { text: 'hello world' },
    outputData: { ok: true },
    status: 'succeeded',
    durationMs: 123
  });

  console.log('Credential created:', {
    vcId: credential.vcId,
    status: credential.status,
    createdAt: credential.createdAt
  });

  const exportResult = await client.exportAuditTrail({ workflowId });
  console.log('Export result count:', {
    agentDids: exportResult.agentDids.length,
    executionVcs: exportResult.executionVcs.length,
    workflowVcs: exportResult.workflowVcs.length,
    totalCount: exportResult.totalCount
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
