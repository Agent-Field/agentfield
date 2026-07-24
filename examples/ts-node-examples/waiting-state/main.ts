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
const openAIApiKey = process.env.OPENAI_API_KEY;

const agent = new Agent({
  nodeId: process.env.AGENT_ID ?? 'waiting-state-demo',
  agentFieldUrl,
  port: Number(process.env.PORT ?? 8005),
  publicUrl: process.env.AGENT_CALLBACK_URL,
  version: '1.0.0',
  devMode: true,
  apiKey: process.env.AGENTFIELD_API_KEY,
  ...(openAIApiKey
    ? {
        aiConfig: {
          provider: 'openai' as const,
          model: process.env.SMALL_MODEL ?? 'gpt-4o-mini',
          apiKey: openAIApiKey,
        },
      }
    : {}),
});

// Create an ApprovalClient for the low-level approval API
const approvalClient = new ApprovalClient({
  baseURL: agentFieldUrl,
  nodeId: process.env.AGENT_ID ?? 'waiting-state-demo',
  apiKey: process.env.AGENTFIELD_API_KEY,
});

function staticPlan(task: string): string {
  return [
    `1. Clarify the goal and success criteria for: ${task}`,
    '2. Identify the safest implementation steps and any dependencies.',
    '3. Execute the plan, verify the outcome, and report the result.',
  ].join('\n');
}

function stringifyPlan(plan: unknown): string {
  return typeof plan === 'string' ? plan : JSON.stringify(plan);
}

/**
 * Reasoner that generates a plan and pauses for human approval.
 *
 * Flow:
 * 1. AI generates a plan for the given task, or a static fallback is used
 * 2. Execution transitions to "waiting" state via approval request
 * 3. Human reviews and approves/rejects (via webhook)
 * 4. Execution resumes based on the decision
 */
agent.reasoner<
  { task: string },
  { status: string; plan: string; feedback?: string }
>('planWithApproval', async (ctx) => {
  ctx.note('Starting plan generation', ['approval', 'start']);

  // Step 1: Generate a plan using AI when configured; otherwise keep the
  // approval demo runnable without external provider credentials.
  const planText = openAIApiKey
    ? stringifyPlan(
        await ctx.ai(
          `You are a project planner. Create a concise 3-step plan for: ${ctx.input.task}`,
          { temperature: 0.7 }
        )
      )
    : staticPlan(ctx.input.task);

  ctx.note('Plan generated, requesting approval', ['approval', 'waiting']);

  // Step 2: Request human approval — transitions execution to "waiting"
  const approvalRequestId = `req-${crypto.randomBytes(6).toString('hex')}`;

  const approvalResponse = await approvalClient.requestApproval(
    ctx.executionId,
    {
      approvalRequestId,
      expiresInHours: 24,
    }
  );
  ctx.note(`Approval requested: ${approvalResponse.approvalRequestId}`, ['approval', 'waiting']);

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

/**
 * Reasoner that uses the high-level `ctx.pause()` primitive (parity with the
 * Python SDK's `app.pause()`).
 *
 * Unlike `planWithApproval` above — which drives the low-level ApprovalClient
 * and polls — `ctx.pause()` transitions the execution to WAITING and blocks on
 * a single promise resolved by the agent's `/webhooks/approval` route when the
 * control plane delivers the human decision. Combined with async-execution
 * dispatch (on by default), the reasoner is 202-acked immediately, so the pause
 * can outlive the control plane's synchronous dispatch ceiling.
 */
agent.reasoner<
  { task: string },
  { status: string; plan: string; feedback?: string }
>('planWithPause', async (ctx) => {
  ctx.note('Generating plan before pausing for approval', ['pause', 'start']);

  const planText = openAIApiKey
    ? stringifyPlan(
        await ctx.ai(
          `You are a project planner. Create a concise 3-step plan for: ${ctx.input.task}`,
          { temperature: 0.7 }
        )
      )
    : staticPlan(ctx.input.task);

  // The approval request would normally be created on an external service
  // (e.g. hax-sdk Response Hub) first; here we just mint an id for the demo.
  const approvalRequestId = `req-${crypto.randomBytes(6).toString('hex')}`;

  // Execution transitions to WAITING here and blocks until the control plane
  // POSTs the decision to this agent's /webhooks/approval route (or the pause
  // times out and returns { decision: 'expired' }).
  const result = await ctx.pause({
    approvalRequestId,
    approvalRequestUrl: `https://hub.example.com/review/${approvalRequestId}`,
    expiresInHours: 24,
  });

  ctx.note(`Pause resolved: ${result.decision}`, ['pause', 'resolved']);

  return {
    status: result.approved ? 'approved' : result.decision,
    plan: planText,
    feedback: result.feedback || undefined,
  };
});


async function main() {
  await agent.serve();
  console.log(`Waiting State Demo Agent listening on http://localhost:${agent.config.port}`);
  console.log(`Control Plane: ${agentFieldUrl}`);
  console.log();
  console.log('Reasoners:');
  console.log('  - planWithApproval: Generates plan, requests approval, polls (low-level)');
  console.log('  - planWithPause: Generates plan, pauses via ctx.pause(), resumes (high-level)');
  console.log('  - checkApproval: Polls approval status for a given execution');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
