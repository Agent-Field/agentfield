import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WorkflowReporter } from '../src/workflow/WorkflowReporter.js';
import type { AgentFieldClient } from '../src/client/AgentFieldClient.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeClient(overrides: Partial<AgentFieldClient> = {}): AgentFieldClient {
  return {
    updateExecutionStatus: vi.fn().mockResolvedValue(undefined),
    ...overrides
  } as unknown as AgentFieldClient;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('WorkflowReporter', () => {
  let client: AgentFieldClient;

  beforeEach(() => {
    client = makeClient();
  });

  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------
  describe('constructor', () => {
    it('creates an instance when executionId is provided', () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });
      expect(reporter).toBeInstanceOf(WorkflowReporter);
    });

    it('throws when executionId is missing', () => {
      expect(() => new WorkflowReporter(client, { executionId: '' })).toThrow(
        'WorkflowReporter requires an executionId'
      );
    });
  });

  // -------------------------------------------------------------------------
  // progress()
  // -------------------------------------------------------------------------
  describe('progress()', () => {
    it('calls updateExecutionStatus with the given progress value', async () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-42' });

      await reporter.progress(50);

      expect(client.updateExecutionStatus).toHaveBeenCalledWith('exec-42', expect.objectContaining({ progress: 50 }));
    });

    it('defaults status to "running"', async () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

      await reporter.progress(25);

      expect(client.updateExecutionStatus).toHaveBeenCalledWith(
        'exec-1',
        expect.objectContaining({ status: 'running' })
      );
    });

    it('uses custom status when provided in options', async () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

      await reporter.progress(75, { status: 'processing' });

      expect(client.updateExecutionStatus).toHaveBeenCalledWith(
        'exec-1',
        expect.objectContaining({ status: 'processing' })
      );
    });

    it('includes result when provided', async () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

      await reporter.progress(100, { result: { output: 'done' } });

      expect(client.updateExecutionStatus).toHaveBeenCalledWith(
        'exec-1',
        expect.objectContaining({ result: { output: 'done' } })
      );
    });

    it('includes error when provided', async () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

      await reporter.progress(0, { error: 'something went wrong' });

      expect(client.updateExecutionStatus).toHaveBeenCalledWith(
        'exec-1',
        expect.objectContaining({ error: 'something went wrong' })
      );
    });

    it('includes durationMs when provided', async () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

      await reporter.progress(50, { durationMs: 1234 });

      expect(client.updateExecutionStatus).toHaveBeenCalledWith(
        'exec-1',
        expect.objectContaining({ durationMs: 1234 })
      );
    });

    it('clamps progress above 100 to 100', async () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

      await reporter.progress(150);

      expect(client.updateExecutionStatus).toHaveBeenCalledWith(
        'exec-1',
        expect.objectContaining({ progress: 100 })
      );
    });

    it('clamps progress below 0 to 0', async () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

      await reporter.progress(-10);

      expect(client.updateExecutionStatus).toHaveBeenCalledWith(
        'exec-1',
        expect.objectContaining({ progress: 0 })
      );
    });

    it('rounds fractional progress values', async () => {
      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

      await reporter.progress(33.7);

      expect(client.updateExecutionStatus).toHaveBeenCalledWith(
        'exec-1',
        expect.objectContaining({ progress: 34 })
      );
    });

    it('propagates errors thrown by updateExecutionStatus', async () => {
      (client.updateExecutionStatus as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error('server unavailable')
      );

      const reporter = new WorkflowReporter(client, { executionId: 'exec-1' });

      await expect(reporter.progress(50)).rejects.toThrow('server unavailable');
    });

    it('passes executionId from metadata to updateExecutionStatus', async () => {
      const reporter = new WorkflowReporter(client, {
        executionId: 'exec-special',
        runId: 'run-1',
        workflowId: 'wf-1'
      });

      await reporter.progress(10);

      expect(client.updateExecutionStatus).toHaveBeenCalledWith('exec-special', expect.any(Object));
    });
  });
});
