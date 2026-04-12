import { describe, it, expect } from 'vitest';
import { ExecutionContext } from '../src/context/ExecutionContext.js';
import type { ExecutionMetadata } from '../src/context/ExecutionContext.js';

const makeMetadata = (overrides: Partial<ExecutionMetadata> = {}): ExecutionMetadata => ({
  executionId: 'exec-123',
  ...overrides
});

const makeFakeReq = () => ({} as any);
const makeFakeRes = () => ({} as any);
const makeFakeAgent = () => ({} as any);

describe('ExecutionContext', () => {
  describe('construction', () => {
    it('stores input and metadata', () => {
      const meta = makeMetadata({ runId: 'run-1', sessionId: 'sess-1' });
      const ctx = new ExecutionContext({
        input: { msg: 'hello' },
        metadata: meta,
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });

      expect(ctx.input).toEqual({ msg: 'hello' });
      expect(ctx.metadata.executionId).toBe('exec-123');
      expect(ctx.metadata.runId).toBe('run-1');
      expect(ctx.metadata.sessionId).toBe('sess-1');
    });

    it('stores req, res, and agent references', () => {
      const req = makeFakeReq();
      const res = makeFakeRes();
      const agent = makeFakeAgent();

      const ctx = new ExecutionContext({
        input: {},
        metadata: makeMetadata(),
        req,
        res,
        agent
      });

      expect(ctx.req).toBe(req);
      expect(ctx.res).toBe(res);
      expect(ctx.agent).toBe(agent);
    });

    it('accepts empty input', () => {
      const ctx = new ExecutionContext({
        input: {},
        metadata: makeMetadata(),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });
      expect(ctx.input).toEqual({});
    });

    it('accepts null-ish input values', () => {
      const ctx = new ExecutionContext({
        input: null,
        metadata: makeMetadata(),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });
      expect(ctx.input).toBeNull();
    });
  });

  describe('metadata fields', () => {
    it('stores workflowId', () => {
      const ctx = new ExecutionContext({
        input: {},
        metadata: makeMetadata({ workflowId: 'wf-42' }),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });
      expect(ctx.metadata.workflowId).toBe('wf-42');
    });

    it('stores parentExecutionId', () => {
      const ctx = new ExecutionContext({
        input: {},
        metadata: makeMetadata({ parentExecutionId: 'parent-exec-1' }),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });
      expect(ctx.metadata.parentExecutionId).toBe('parent-exec-1');
    });

    it('stores callerDid and targetDid', () => {
      const ctx = new ExecutionContext({
        input: {},
        metadata: makeMetadata({ callerDid: 'did:key:caller', targetDid: 'did:key:target' }),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });
      expect(ctx.metadata.callerDid).toBe('did:key:caller');
      expect(ctx.metadata.targetDid).toBe('did:key:target');
    });

    it('stores agentNodeDid', () => {
      const ctx = new ExecutionContext({
        input: {},
        metadata: makeMetadata({ agentNodeDid: 'did:key:node' }),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });
      expect(ctx.metadata.agentNodeDid).toBe('did:key:node');
    });

    it('allows undefined optional fields', () => {
      const ctx = new ExecutionContext({
        input: {},
        metadata: makeMetadata(),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });
      expect(ctx.metadata.runId).toBeUndefined();
      expect(ctx.metadata.sessionId).toBeUndefined();
      expect(ctx.metadata.workflowId).toBeUndefined();
      expect(ctx.metadata.parentExecutionId).toBeUndefined();
      expect(ctx.metadata.callerDid).toBeUndefined();
    });
  });

  describe('AsyncLocalStorage – getCurrent / run', () => {
    it('getCurrent returns undefined outside a run', () => {
      expect(ExecutionContext.getCurrent()).toBeUndefined();
    });

    it('getCurrent returns the context inside run()', () => {
      const ctx = new ExecutionContext({
        input: { x: 1 },
        metadata: makeMetadata({ sessionId: 'sess-run' }),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });

      let captured: ExecutionContext | undefined;
      ExecutionContext.run(ctx, () => {
        captured = ExecutionContext.getCurrent();
      });

      expect(captured).toBe(ctx);
      expect(captured?.metadata.sessionId).toBe('sess-run');
    });

    it('run() returns the value produced by the callback', () => {
      const ctx = new ExecutionContext({
        input: {},
        metadata: makeMetadata(),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });

      const result = ExecutionContext.run(ctx, () => 42);
      expect(result).toBe(42);
    });

    it('context is isolated – getCurrent is undefined again after run exits', () => {
      const ctx = new ExecutionContext({
        input: {},
        metadata: makeMetadata(),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });

      ExecutionContext.run(ctx, () => {
        /* inside: store is set */
      });

      // After run exits the outer scope has no store
      expect(ExecutionContext.getCurrent()).toBeUndefined();
    });

    it('nested run scopes use inner context', () => {
      const outer = new ExecutionContext({
        input: { level: 'outer' },
        metadata: makeMetadata({ executionId: 'exec-outer' }),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });

      const inner = new ExecutionContext({
        input: { level: 'inner' },
        metadata: makeMetadata({ executionId: 'exec-inner' }),
        req: makeFakeReq(),
        res: makeFakeRes(),
        agent: makeFakeAgent()
      });

      let outerSeen: string | undefined;
      let innerSeen: string | undefined;

      ExecutionContext.run(outer, () => {
        outerSeen = ExecutionContext.getCurrent()?.metadata.executionId;
        ExecutionContext.run(inner, () => {
          innerSeen = ExecutionContext.getCurrent()?.metadata.executionId;
        });
      });

      expect(outerSeen).toBe('exec-outer');
      expect(innerSeen).toBe('exec-inner');
    });
  });
});
