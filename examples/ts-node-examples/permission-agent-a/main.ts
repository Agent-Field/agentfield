/**
 * Permission Agent A (Caller) — TypeScript SDK
 *
 * An agent with tag "analytics" that demonstrates the policy engine:
 *   - call_analytics  -> calls ts-perm-target.analyze_data (ALLOWED by policy)
 *   - call_large_query -> calls ts-perm-target.analyze_data with limit=5000 (DENIED: constraint)
 *   - call_delete     -> calls ts-perm-target.delete_records (DENIED: deny_functions)
 *
 * The "analytics" tag auto-approves (tag_approval_rules), so this agent starts
 * immediately in "active" state.
 *
 * Test flow:
 *  1. Start control plane with authorization enabled
 *  2. Start ts-perm-target -> enters pending_approval
 *  3. Admin approves ts-perm-target's tags
 *  4. Start ts-perm-caller (this agent) -> auto-approved
 *  5. POST /api/v1/execute/ts-perm-caller.call_analytics -> 200 OK
 *  6. POST /api/v1/execute/ts-perm-caller.call_large_query -> 403 constraint
 *  7. POST /api/v1/execute/ts-perm-caller.call_delete -> 403 denied function
 */

import { Agent } from '@agentfield/sdk';

async function main() {
  const agent = new Agent({
    nodeId: 'ts-perm-caller',
    agentFieldUrl: process.env.AGENTFIELD_URL ?? 'http://localhost:8080',
    port: Number(process.env.PORT ?? 8005),
    version: '1.0.0',
    devMode: true,
    didEnabled: true,
    tags: ['analytics'],
  });

  // Simple health check — no cross-agent call, should always work.
  agent.reasoner('ping', async (ctx) => {
    return {
      status: 'ok',
      agent: 'ts-perm-caller',
    };
  }, {
    description: 'Simple health check',
  });

  // Calls ts-perm-target.analyze_data with a small limit.
  // Should succeed: analytics -> data-service, analyze_* is in allow_functions,
  // limit=100 satisfies the <= 1000 constraint.
  agent.reasoner('call_analytics', async (ctx) => {
    const query = ctx.input.query ?? 'default analytics query';

    const result = await agent.call('ts-perm-target.analyze_data', {
      query,
      limit: 100,
    });

    return {
      source: 'ts-perm-caller',
      test: 'allowed_query',
      delegation_result: result,
    };
  }, {
    description: 'Calls ts-perm-target.analyze_data (allowed)',
    tags: ['analytics'],
  });

  // Calls ts-perm-target.analyze_data with limit=5000.
  // Should fail: limit=5000 violates the <= 1000 constraint.
  agent.reasoner('call_large_query', async (ctx) => {
    const query = ctx.input.query ?? 'SELECT * FROM big_table';

    const result = await agent.call('ts-perm-target.analyze_data', {
      query,
      limit: 5000,
    });

    return {
      source: 'ts-perm-caller',
      test: 'constraint_violation',
      delegation_result: result,
    };
  }, {
    description: 'Calls ts-perm-target.analyze_data with large limit (constraint violation)',
    tags: ['analytics'],
  });

  // Calls ts-perm-target.delete_records.
  // Should fail: delete_* is in deny_functions for analytics->data-service.
  agent.reasoner('call_delete', async (ctx) => {
    const table = ctx.input.table ?? 'sensitive_records';

    const result = await agent.call('ts-perm-target.delete_records', {
      table,
    });

    return {
      source: 'ts-perm-caller',
      test: 'deny_function',
      delegation_result: result,
    };
  }, {
    description: 'Calls ts-perm-target.delete_records (denied by policy)',
    tags: ['analytics'],
  });

  await agent.serve();

  console.log(`
Permission Agent A (Caller) — TypeScript SDK
Node: ts-perm-caller
Port: ${agent.config.port}
Server: ${agent.config.agentFieldUrl}
Tags: analytics
Test reasoners: call_analytics (allow), call_large_query (constraint), call_delete (deny)
  `);
}

main().catch((err) => {
  console.error('Failed to start agent:', err);
  process.exit(1);
});
