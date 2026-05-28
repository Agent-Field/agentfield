/**
 * Permission Agent B (Protected Target) — TypeScript SDK
 *
 * A protected agent with tags ["sensitive", "data-service"]. The "sensitive"
 * tag triggers manual approval (tag_approval_rules in config), so this agent
 * starts in "pending_approval" state until an admin approves its tags.
 *
 * Once approved, access policies control which callers can invoke which reasoners:
 *   - analytics callers can call analyze_data and get_schema (allowed by policy)
 *   - analytics callers are denied delete_records (deny_functions in policy)
 *   - constraint violations (e.g. limit > 1000) are rejected
 *
 * Reasoners:
 *   - analyze_data   — simulates data analysis, generates a VC on success
 *   - delete_records — simulates record deletion (denied for analytics callers)
 *   - get_schema     — returns the data schema
 */

import { Agent } from '@agentfield/sdk';

async function main() {
  const agent = new Agent({
    nodeId: 'ts-perm-target',
    agentFieldUrl: process.env.AGENTFIELD_URL ?? 'http://localhost:8080',
    port: Number(process.env.PORT ?? 8006),
    version: '1.0.0',
    devMode: true,
    didEnabled: true,
    tags: ['sensitive', 'data-service'],
  });

  // Reasoner 1: analyze_data — simulates data analysis with VC generation.
  agent.reasoner('analyze_data', async (ctx) => {
    const startTime = Date.now();
    const query = ctx.input.query ?? 'no query provided';
    const limit = ctx.input.limit ?? 100;

    const result = {
      status: 'analyzed',
      agent: 'ts-perm-target',
      query,
      limit,
      insights: [
        { metric: 'total_records', value: 1542 },
        { metric: 'avg_processing_time_ms', value: 23.7 },
        { metric: 'error_rate', value: 0.003 },
      ],
      message: `Analysis complete for query: ${query} (limit=${limit})`,
      vcGenerated: false as boolean,
      vcId: undefined as string | undefined,
    };

    // Generate a Verifiable Credential for this execution
    try {
      const credential = await ctx.did.generateCredential({
        inputData: ctx.input,
        outputData: { insights: result.insights },
        status: 'succeeded',
        durationMs: Date.now() - startTime,
      });

      result.vcGenerated = true;
      result.vcId = credential.vcId;
      console.log(`[VC] Generated credential for analyze_data: ${credential.vcId}`);
    } catch (error) {
      console.error('[VC] Failed to generate credential:', error);
    }

    return result;
  }, {
    description: 'Analyze data. Protected by access policy — analytics callers allowed.',
    tags: ['sensitive', 'data-service'],
  });

  // Reasoner 2: delete_records — denied for analytics callers by policy.
  agent.reasoner('delete_records', async (ctx) => {
    const table = ctx.input.table ?? 'records';

    return {
      status: 'deleted',
      agent: 'ts-perm-target',
      table,
      message: `Records deleted from ${table}`,
    };
  }, {
    description: 'Delete records. Denied for analytics callers by policy.',
    tags: ['data-service'],
  });

  // Reasoner 3: get_schema — returns the data schema.
  agent.reasoner('get_schema', async (ctx) => {
    return {
      status: 'success',
      agent: 'ts-perm-target',
      schema: {
        table: 'records',
        columns: [
          { name: 'id', type: 'integer', primary_key: true },
          { name: 'name', type: 'text' },
          { name: 'created_at', type: 'timestamp' },
        ],
      },
    };
  }, {
    description: 'Get the data schema.',
    tags: ['data-service'],
  });

  await agent.serve();

  console.log(`
Permission Agent B (Protected) — TypeScript SDK
Node: ts-perm-target
Port: ${agent.config.port}
Server: ${agent.config.agentFieldUrl}
Tags: sensitive, data-service
Reasoners: analyze_data, delete_records, get_schema
  `);
}

main().catch((err) => {
  console.error('Failed to start agent:', err);
  process.exit(1);
});
