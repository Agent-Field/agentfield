/**
 * Behavioral invariant tests for ExecutionContext.
 *
 * These tests verify structural properties (scope isolation, propagation
 * completeness, concurrent isolation) that must always hold.
 */
import { describe, it, expect } from 'vitest';
import { ExecutionContext, type ExecutionMetadata } from '../src/context/ExecutionContext.js';

// ── Test helpers ──────────────────────────────────────────────────────────────

function makeCtx(overrides: Partial<ExecutionMetadata> & { input?: any } = {}): ExecutionContext {
  const { input = {}, ...metaOverrides } = overrides;
  return new ExecutionContext({
    input,
    metadata: {
      executionId: `exec-${Math.random().toString(36).slice(2)}`,
      ...metaOverrides
    },
    req: {} as any,
    res: {} as any,
    agent: {} as any
  });
}

// ── Invariants ────────────────────────────────────────────────────────────────

describe('ExecutionContext invariants', () => {
  // ── Nested Scope Isolation ──────────────────────────────────────────────────

  describe('INVARIANT: inner run() scope does not leak to outer scope after completion', () => {
    it('outer context is restored after inner run completes', () => {
      const outer = makeCtx({ executionId: 'outer-exec', sessionId: 'outer-session' });
      const inner = makeCtx({ executionId: 'inner-exec', sessionId: 'inner-session' });

      let outerDuringInner: string | undefined;
      let afterInner: string | undefined;

      ExecutionContext.run(outer, () => {
        ExecutionContext.run(inner, () => {
          // Inside inner: should see inner context
        });
        // After inner completes: should be back to outer
        afterInner = ExecutionContext.getCurrent()?.metadata.executionId;
      });

      // Outer context should be active immediately after inner exits
      expect(afterInner).toBe('outer-exec');
    });

    it('getCurrent() is undefined after all run() calls exit', () => {
      const ctx = makeCtx({ executionId: 'temp-exec' });

      ExecutionContext.run(ctx, () => {
        expect(ExecutionContext.getCurrent()?.metadata.executionId).toBe('temp-exec');
      });

      // After the run block, the store should be empty in the outer scope
      expect(ExecutionContext.getCurrent()).toBeUndefined();
    });

    it('deeply nested scopes unwind correctly', () => {
      const ctx1 = makeCtx({ executionId: 'level-1' });
      const ctx2 = makeCtx({ executionId: 'level-2' });
      const ctx3 = makeCtx({ executionId: 'level-3' });

      const observed: (string | undefined)[] = [];

      ExecutionContext.run(ctx1, () => {
        observed.push(ExecutionContext.getCurrent()?.metadata.executionId);
        ExecutionContext.run(ctx2, () => {
          observed.push(ExecutionContext.getCurrent()?.metadata.executionId);
          ExecutionContext.run(ctx3, () => {
            observed.push(ExecutionContext.getCurrent()?.metadata.executionId);
          });
          // Back at level 2
          observed.push(ExecutionContext.getCurrent()?.metadata.executionId);
        });
        // Back at level 1
        observed.push(ExecutionContext.getCurrent()?.metadata.executionId);
      });

      expect(observed).toEqual(['level-1', 'level-2', 'level-3', 'level-2', 'level-1']);
    });
  });

  // ── Context Propagation Completeness ────────────────────────────────────────

  describe('INVARIANT: all metadata fields from parent context are accessible inside run()', () => {
    it('all optional metadata fields propagate correctly', () => {
      const ctx = new ExecutionContext({
        input: { key: 'val' },
        metadata: {
          executionId: 'exec-full',
          runId: 'run-1',
          sessionId: 'sess-1',
          actorId: 'actor-1',
          workflowId: 'wf-1',
          parentExecutionId: 'parent-exec-1',
          callerDid: 'did:key:caller',
          targetDid: 'did:key:target',
          agentNodeDid: 'did:key:node'
        },
        req: {} as any,
        res: {} as any,
        agent: {} as any
      });

      let captured: ExecutionContext | undefined;
      ExecutionContext.run(ctx, () => {
        captured = ExecutionContext.getCurrent();
      });

      expect(captured).toBeDefined();
      expect(captured!.metadata.executionId).toBe('exec-full');
      expect(captured!.metadata.runId).toBe('run-1');
      expect(captured!.metadata.sessionId).toBe('sess-1');
      expect(captured!.metadata.actorId).toBe('actor-1');
      expect(captured!.metadata.workflowId).toBe('wf-1');
      expect(captured!.metadata.parentExecutionId).toBe('parent-exec-1');
      expect(captured!.metadata.callerDid).toBe('did:key:caller');
      expect(captured!.metadata.targetDid).toBe('did:key:target');
      expect(captured!.metadata.agentNodeDid).toBe('did:key:node');
    });

    it('input is accessible inside the run scope', () => {
      const ctx = makeCtx({ input: { complex: { nested: { value: 42 } } } });

      let capturedInput: any;
      ExecutionContext.run(ctx, () => {
        capturedInput = ExecutionContext.getCurrent()?.input;
      });

      expect(capturedInput).toEqual({ complex: { nested: { value: 42 } } });
    });

    it('req, res, agent references are the same objects inside the scope', () => {
      const req = { id: 'fake-req' } as any;
      const res = { id: 'fake-res' } as any;
      const agent = { id: 'fake-agent' } as any;

      const ctx = new ExecutionContext({
        input: {},
        metadata: { executionId: 'ref-test' },
        req,
        res,
        agent
      });

      let capturedCtx: ExecutionContext | undefined;
      ExecutionContext.run(ctx, () => {
        capturedCtx = ExecutionContext.getCurrent();
      });

      // Reference equality — same objects, not copies
      expect(capturedCtx!.req).toBe(req);
      expect(capturedCtx!.res).toBe(res);
      expect(capturedCtx!.agent).toBe(agent);
    });
  });

  // ── Concurrent Context Isolation ────────────────────────────────────────────

  describe('INVARIANT: concurrent run() calls do not interfere with each other', () => {
    it('two concurrent async run() calls see their own context', async () => {
      const ctxA = makeCtx({ executionId: 'concurrent-A', sessionId: 'sess-A' });
      const ctxB = makeCtx({ executionId: 'concurrent-B', sessionId: 'sess-B' });

      let seenInA: string | undefined;
      let seenInB: string | undefined;

      const taskA = new Promise<void>((resolve) => {
        ExecutionContext.run(ctxA, async () => {
          await new Promise((r) => setTimeout(r, 1)); // yield
          seenInA = ExecutionContext.getCurrent()?.metadata.executionId;
          resolve();
        });
      });

      const taskB = new Promise<void>((resolve) => {
        ExecutionContext.run(ctxB, async () => {
          await new Promise((r) => setTimeout(r, 1)); // yield
          seenInB = ExecutionContext.getCurrent()?.metadata.executionId;
          resolve();
        });
      });

      await Promise.all([taskA, taskB]);

      expect(seenInA).toBe('concurrent-A');
      expect(seenInB).toBe('concurrent-B');
    });

    it('N concurrent executions each see their own distinct context', async () => {
      const COUNT = 10;
      const results: (string | undefined)[] = new Array(COUNT);

      await Promise.all(
        Array.from({ length: COUNT }, (_, i) => {
          const ctx = makeCtx({ executionId: `exec-${i}` });
          return new Promise<void>((resolve) => {
            ExecutionContext.run(ctx, async () => {
              // Introduce async gap to force interleaving
              await new Promise((r) => setTimeout(r, Math.random() * 5));
              results[i] = ExecutionContext.getCurrent()?.metadata.executionId;
              resolve();
            });
          });
        })
      );

      // Every slot must contain its own execution ID
      for (let i = 0; i < COUNT; i++) {
        expect(results[i]).toBe(`exec-${i}`);
      }

      // All IDs must be unique
      const unique = new Set(results);
      expect(unique.size).toBe(COUNT);
    });
  });

  // ── getCurrent / run API contract ────────────────────────────────────────────

  describe('INVARIANT: getCurrent() reflects exact context passed to run()', () => {
    it('getCurrent() returns the exact same object instance passed to run()', () => {
      const ctx = makeCtx();
      let captured: ExecutionContext | undefined;
      ExecutionContext.run(ctx, () => {
        captured = ExecutionContext.getCurrent();
      });
      expect(captured).toBe(ctx); // reference equality
    });

    it('run() return value matches the callback return value', () => {
      const ctx = makeCtx();
      const result = ExecutionContext.run(ctx, () => 'sentinel-value');
      expect(result).toBe('sentinel-value');
    });

    it('run() propagates exceptions from the callback', () => {
      const ctx = makeCtx();
      expect(() =>
        ExecutionContext.run(ctx, () => {
          throw new Error('from-callback');
        })
      ).toThrow('from-callback');
    });
  });
});
