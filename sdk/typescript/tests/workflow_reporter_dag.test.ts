import { afterEach, describe, expect, it, vi } from 'vitest';
import { WorkflowReporter } from '../src/workflow/WorkflowReporter.js';
import type { AgentFieldClient } from '../src/client/AgentFieldClient.js';

function makeClient(overrides: Partial<AgentFieldClient> = {}): AgentFieldClient {
  return {
    updateExecutionStatus: vi.fn().mockResolvedValue(undefined),
    ...overrides
  } as unknown as AgentFieldClient;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('WorkflowReporter branch behavior', () => {
  it('forwards normalized progress and terminal-style payloads through updateExecutionStatus', async () => {
    const client = makeClient();
    const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

    await reporter.progress(99.6, {
      status: 'succeeded',
      result: { ok: true },
      durationMs: 42
    });

    expect(client.updateExecutionStatus).toHaveBeenCalledWith('exec-1', {
      status: 'succeeded',
      progress: 100,
      result: { ok: true },
      error: undefined,
      durationMs: 42
    });
  });

  it('does not track status transitions or synthesize duration across calls', async () => {
    const client = makeClient();
    const reporter = new WorkflowReporter(client, { executionId: 'exec-2' });

    await reporter.progress(0, { status: 'waiting' });
    await reporter.progress(100, { status: 'succeeded' });

    expect(client.updateExecutionStatus).toHaveBeenNthCalledWith(1, 'exec-2', {
      status: 'waiting',
      progress: 0,
      result: undefined,
      error: undefined,
      durationMs: undefined
    });
    expect(client.updateExecutionStatus).toHaveBeenNthCalledWith(2, 'exec-2', {
      status: 'succeeded',
      progress: 100,
      result: undefined,
      error: undefined,
      durationMs: undefined
    });
  });

  it('uses only executionId for routing and drops ad hoc statusReason values', async () => {
    const client = makeClient();
    const reporter = new WorkflowReporter(client, {
      executionId: 'exec-3',
      runId: 'run-1',
      workflowId: 'wf-1',
      agentNodeId: 'node-1',
      reasonerId: 'planner'
    });

    await reporter.progress(12.4, {
      status: 'failed',
      error: 'boom',
      durationMs: 12,
      statusReason: 'upstream failure'
    } as any);

    expect(client.updateExecutionStatus).toHaveBeenCalledWith('exec-3', {
      status: 'failed',
      progress: 12,
      result: undefined,
      error: 'boom',
      durationMs: 12
    });
  });

  it('propagates client failures instead of swallowing them', async () => {
    const client = makeClient({
      updateExecutionStatus: vi.fn().mockRejectedValue(new Error('control plane unavailable'))
    });
    const reporter = new WorkflowReporter(client, { executionId: 'exec-4' });

    await expect(reporter.progress(50, { status: 'running' })).rejects.toThrow(
      'control plane unavailable'
    );
  });
});
