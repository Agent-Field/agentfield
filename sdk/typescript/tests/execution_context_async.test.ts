import type express from 'express';
import type { Agent } from '../src/agent/Agent.js';
import { describe, expect, it, vi } from 'vitest';
import { ExecutionContext, type ExecutionMetadata } from '../src/context/ExecutionContext.js';

function makeContext(executionId: string, overrides: Partial<ExecutionMetadata> = {}) {
  return new ExecutionContext({
    input: { executionId },
    metadata: {
      executionId,
      ...overrides
    },
    req: {} as express.Request,
    res: {} as express.Response,
    agent: {
      getExecutionLogger: vi.fn()
    } as unknown as Agent
  });
}

describe('ExecutionContext async propagation', () => {
  it('run() exposes the current context through nested awaits', async () => {
    const ctx = makeContext('exec-1', { workflowId: 'wf-1' });

    const result = await ExecutionContext.run(ctx, async () => {
      expect(ExecutionContext.getCurrent()).toBe(ctx);

      await Promise.resolve();
      expect(ExecutionContext.getCurrent()).toBe(ctx);

      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      expect(ExecutionContext.getCurrent()).toBe(ctx);

      return ExecutionContext.getCurrent()?.metadata.workflowId;
    });

    expect(result).toBe('wf-1');
    expect(ExecutionContext.getCurrent()).toBeUndefined();
  });

  it('keeps parallel runs isolated from each other', async () => {
    const ctxA = makeContext('exec-a', { workflowId: 'wf-a', sessionId: 'session-a' });
    const ctxB = makeContext('exec-b', { workflowId: 'wf-b', sessionId: 'session-b' });

    const seen = await Promise.all([
      ExecutionContext.run(ctxA, async () => {
        await Promise.resolve();
        ctxA.metadata.sessionId = 'session-a-updated';
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
        return {
          executionId: ExecutionContext.getCurrent()?.metadata.executionId,
          workflowId: ExecutionContext.getCurrent()?.metadata.workflowId,
          sessionId: ExecutionContext.getCurrent()?.metadata.sessionId
        };
      }),
      ExecutionContext.run(ctxB, async () => {
        await Promise.resolve();
        return {
          executionId: ExecutionContext.getCurrent()?.metadata.executionId,
          workflowId: ExecutionContext.getCurrent()?.metadata.workflowId,
          sessionId: ExecutionContext.getCurrent()?.metadata.sessionId
        };
      })
    ]);

    expect(seen).toEqual([
      {
        executionId: 'exec-a',
        workflowId: 'wf-a',
        sessionId: 'session-a-updated'
      },
      {
        executionId: 'exec-b',
        workflowId: 'wf-b',
        sessionId: 'session-b'
      }
    ]);
    expect(ctxB.metadata.sessionId).toBe('session-b');
    expect(ExecutionContext.getCurrent()).toBeUndefined();
  });

  it('uses the child context inside a nested run and restores the parent afterwards', async () => {
    const parent = makeContext('exec-parent', { workflowId: 'wf-parent' });
    const child = makeContext('exec-child', { workflowId: 'wf-child' });

    const observed = await ExecutionContext.run(parent, async () => {
      const beforeChild = ExecutionContext.getCurrent()?.metadata.executionId;

      const insideChild = await ExecutionContext.run(child, async () => {
        await Promise.resolve();
        return ExecutionContext.getCurrent()?.metadata.executionId;
      });

      const afterChild = ExecutionContext.getCurrent()?.metadata.executionId;

      return { beforeChild, insideChild, afterChild };
    });

    expect(observed).toEqual({
      beforeChild: 'exec-parent',
      insideChild: 'exec-child',
      afterChild: 'exec-parent'
    });
    expect(ExecutionContext.getCurrent()).toBeUndefined();
  });
});
