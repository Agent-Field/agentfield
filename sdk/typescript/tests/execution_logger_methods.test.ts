import { describe, expect, it, vi } from 'vitest';
import {
  ExecutionLogger,
  createExecutionLogger,
  isExecutionLogBatchPayload,
  normalizeExecutionLogEntry,
  serializeExecutionLogEntry,
  type ExecutionLogTransportPayload
} from '../src/observability/ExecutionLogger.js';

describe('ExecutionLogger exported API', () => {
  it('normalizes and serializes entries including bigint and circular attributes', () => {
    const circular: { self?: unknown } = {};
    circular.self = circular;

    const normalized = normalizeExecutionLogEntry({
      v: 1,
      ts: '2025-01-01T00:00:00Z',
      executionId: 'exec-1',
      runId: 'run-1',
      workflowId: 'wf-1',
      rootWorkflowId: 'root-1',
      parentExecutionId: 'parent-1',
      sessionId: 'sess-1',
      actorId: 'actor-1',
      agentNodeId: 'node-1',
      reasonerId: 'reasoner-1',
      callerDid: 'did:caller',
      targetDid: 'did:target',
      agentNodeDid: 'did:node',
      level: 'info',
      source: 'sdk.test',
      eventType: 'execution.started',
      message: 'hello',
      systemGenerated: true,
      attributes: {
        size: 12n,
        circular
      }
    });

    expect(normalized).toEqual({
      v: 1,
      ts: '2025-01-01T00:00:00Z',
      execution_id: 'exec-1',
      run_id: 'run-1',
      workflow_id: 'wf-1',
      root_workflow_id: 'root-1',
      parent_execution_id: 'parent-1',
      session_id: 'sess-1',
      actor_id: 'actor-1',
      agent_node_id: 'node-1',
      reasoner_id: 'reasoner-1',
      caller_did: 'did:caller',
      target_did: 'did:target',
      agent_node_did: 'did:node',
      level: 'info',
      source: 'sdk.test',
      event_type: 'execution.started',
      message: 'hello',
      attributes: {
        size: 12n,
        circular
      },
      system_generated: true
    });

    expect(serializeExecutionLogEntry({
      v: 1,
      ts: '2025-01-01T00:00:00Z',
      level: 'info',
      source: 'sdk.test',
      message: 'hello',
      attributes: {
        size: 12n,
        circular
      }
    })).toContain('"size":"12"');
    expect(serializeExecutionLogEntry({
      v: 1,
      ts: '2025-01-01T00:00:00Z',
      level: 'info',
      source: 'sdk.test',
      message: 'hello',
      attributes: {
        circular
      }
    })).toContain('"self":"[Circular]"');
  });

  it('emits logs through stdout and transport only when execution context exists', async () => {
    const writes: string[] = [];
    const transportPayloads: ExecutionLogTransportPayload[] = [];
    const logger = new ExecutionLogger({
      contextProvider: () => ({
        executionId: 'exec-1',
        workflowId: 'wf-1'
      }),
      stdout: {
        write: (chunk: string) => {
          writes.push(chunk);
          return true;
        }
      },
      transport: {
        emit: (payload) => {
          transportPayloads.push(payload);
        }
      },
      source: 'sdk.custom'
    });

    const debugEntry = logger.debug('debug');
    const infoEntry = logger.info('info', { step: 1 });
    const warnEntry = logger.warn('warn');
    const errorEntry = logger.error('error');
    const systemEntry = logger.system('runtime.tick', 'tick', { ok: true });

    expect(debugEntry.level).toBe('debug');
    expect(infoEntry.attributes).toEqual({ step: 1 });
    expect(warnEntry.level).toBe('warn');
    expect(errorEntry.level).toBe('error');
    expect(systemEntry).toMatchObject({
      source: 'sdk.runtime',
      eventType: 'runtime.tick',
      systemGenerated: true
    });
    expect(writes).toHaveLength(5);
    expect(transportPayloads).toHaveLength(5);
    expect(transportPayloads[0]).toMatchObject({
      execution_id: 'exec-1',
      workflow_id: 'wf-1',
      source: 'sdk.custom',
      message: 'debug'
    });

    const noContextPayloads: ExecutionLogTransportPayload[] = [];
    const silentLogger = createExecutionLogger({
      mirrorToStdout: false,
      transport: {
        emit: (payload) => {
          noContextPayloads.push(payload);
        }
      }
    });
    const entry = silentLogger.log('info', 'no transport');

    expect(entry.source).toBe('sdk.logger');
    expect(noContextPayloads).toEqual([]);
  });

  it('ignores transport failures and recognizes batch payloads', async () => {
    const writes: string[] = [];
    const syncFailureLogger = createExecutionLogger({
      contextProvider: () => ({ executionId: 'exec-1' }),
      stdout: {
        write: (chunk: string) => {
          writes.push(chunk);
          return true;
        }
      },
      transport: {
        emit: () => {
          throw new Error('sync failure');
        }
      }
    });

    expect(() => syncFailureLogger.info('still writes')).not.toThrow();
    expect(writes).toHaveLength(1);

    const asyncFailureLogger = createExecutionLogger({
      contextProvider: () => ({ executionId: 'exec-2' }),
      mirrorToStdout: false,
      transport: {
        emit: () => Promise.reject(new Error('async failure'))
      }
    });

    await expect(Promise.resolve(asyncFailureLogger.info('async failure ignored'))).resolves.toMatchObject({
      message: 'async failure ignored'
    });

    expect(isExecutionLogBatchPayload({ entries: [] })).toBe(true);
    expect(isExecutionLogBatchPayload({ message: 'single' } as unknown as ExecutionLogTransportPayload)).toBe(false);
  });
});
