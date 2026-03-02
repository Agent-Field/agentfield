/**
 * Waiting State Agent - Human-in-the-Loop Approval Example
 *
 * Demonstrates:
 * - Requesting human approval mid-execution (waiting state)
 * - Polling for approval status with exponential backoff
 * - Handling approved / rejected / expired decisions
 * - Using the ApprovalClient for low-level control
 */

import 'dotenv/config';
import { Agent, ApprovalClient } from '@agentfield/sdk';
import crypto from 'node:crypto';

const agentFieldUrl = process.env.AGENTFIELD_URL ?? 'http://localhost:8080';

const agent = new Agent({
  nodeId: process.env.AGENT_ID ?? 'waiting-state-demo',
  agentFieldUrl,
  port: Number(process.env.PORT ?? 8005),
  publicUrl: process.env.AGENT_CALLBACK_URL,
  version: '1.0.0',
  devMode: true,
  apiKey: process.env.AGENTFIELD_API_KEY,
  aiConfig: {
    provider: 'openai',
    model: process.env.SMALL_MODEL ?? 'gpt-4o-mini',
    apiKey: process.env.OPENAI_API_KEY,
  },
});

// Create an ApprovalClient for the low-level approval API
const approvalClient = new ApprovalClient({
  baseURL: agentFieldUrl,
  nodeId: process.env.AGENT_ID ?? 'waiting-state-demo',
  apiKey: process.env.AGENTFIELD_API_KEY,
});

/**
 * Reasoner that generates a plan and pauses for human approval.
 *
 * Flow:
 * 1. AI generates a plan for the given task
 * 2. Execution transitions to "waiting" state via approval request
 * 3. Human reviews and approves/rejects (via webhook)
 * 4. Execution resumes based on the decision
 */
agent.reasoner<
  { task: string },
  { status: string; plan: string; feedback?: string }
>('planWithApproval', async (ctx) => {
  ctx.note('Starting plan generation', ['approval', 'start']);

  // Step 1: Generate a plan using AI
  const plan = await ctx.ai(
    `You are a project planner. Create a concise 3-step plan for: ${ctx.input.task}`,
    { temperature: 0.7 }
  );
  const planText = typeof plan === 'string' ? plan : JSON.stringify(plan);

  ctx.note('Plan generated, requesting approval', ['approval', 'waiting']);

  // Step 2: Request human approval — transitions execution to "waiting"
  const approvalRequestId = `req-${crypto.randomBytes(6).toString('hex')}`;

  const approvalResponse = await approvalClient.requestApproval(
    ctx.executionId,
    {
      projectId: 'waiting-state-demo',
      title: `Plan Review: ${ctx.input.task}`,
      description: planText,
      expiresInHours: 24,
    }
  );

  // Step 3: Wait for approval resolution (polls with exponential backoff)
  const result = await approvalClient.waitForApproval(ctx.executionId, {
    pollIntervalMs: 5_000,
    maxIntervalMs: 30_000,
    timeoutMs: 3_600_000, // 1 hour
  });

  ctx.note(`Approval resolved: ${result.status}`, ['approval', 'resolved']);

  // Step 4: Handle the decision
  const feedback = result.response?.feedback as string | undefined;

  if (result.status === 'approved') {
    return {
      status: 'approved',
      plan: planText,
      feedback,
    };
  }

  return {
    status: result.status,
    plan: planText,
    feedback: feedback ?? `Plan was ${result.status}`,
  };
});

/**
 * Simple reasoner that demonstrates approval status polling
 * without blocking — useful for fire-and-forget approval checks.
 */
agent.reasoner<
  { executionId: string },
  { status: string; response?: Record<string, any> }
>('checkApproval', async (ctx) => {
  const status = await approvalClient.getApprovalStatus(ctx.input.executionId);

  return {
    status: status.status,
    response: status.response,
  };
});


async function main() {
  await agent.serve();
  console.log(`Waiting State Demo Agent listening on http://localhost:${agent.config.port}`);
  console.log(`Control Plane: ${agentFieldUrl}`);
  console.log();
  console.log('Reasoners:');
  console.log('  - planWithApproval: Generates plan, pauses for approval, resumes');
  console.log('  - checkApproval: Polls approval status for a given execution');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
