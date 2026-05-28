import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import { AgentFieldClient } from '../src/client/AgentFieldClient.js';
import { ExecutionContext, type ExecutionMetadata } from '../src/context/ExecutionContext.js';
import {
  createExecutionLogger,
  serializeExecutionLogEntry
} from '../src/observability/ExecutionLogger.js';

describe('ExecutionLogger', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('serializes execution context to the backend envelope and mirrors to stdout', () => {
    const writes: string[] = [];
    const payloads: any[] = [];
    const logger = createExecutionLogger({
      contextProvider: () => ({
        executionId: 'exec-1',
        runId: 'run-1',
        workflowId: 'wf-1',
        rootWorkflowId: 'root-1',
        parentExecutionId: 'parent-1',
        sessionId: 'session-1',
        actorId: 'actor-1',
        agentNodeId: 'node-1',
        reasonerId: 'reasoner-1',
        callerDid: 'did:caller',
        targetDid: 'did:target',
        agentNodeDid: 'did:node'
      }),
      stdout: {
        write: (chunk: string) => {
          writes.push(chunk);
          return true;
        }
      },
      transport: {
        emit: (payload) => payloads.push(payload)
      }
    });

    const entry = logger.system('execution.started', 'Execution started', {
      stage: 'bootstrap'
    });

    expect(entry.reasonerId).toBe('reasoner-1');
    expect(serializeExecutionLogEntry(entry)).toContain('"execution_id":"exec-1"');
    expect(writes).toHaveLength(1);
    expect(JSON.parse(writes[0])).toMatchObject({
      execution_id: 'exec-1',
      run_id: 'run-1',
      workflow_id: 'wf-1',
      root_workflow_id: 'root-1',
      parent_execution_id: 'parent-1',
      session_id: 'session-1',
      actor_id: 'actor-1',
      agent_node_id: 'node-1',
      reasoner_id: 'reasoner-1',
      caller_did: 'did:caller',
      target_did: 'did:target',
      agent_node_did: 'did:node',
      event_type: 'execution.started',
      source: 'sdk.runtime',
      system_generated: true,
      attributes: {
        stage: 'bootstrap'
      }
    });
    expect(payloads).toHaveLength(1);
    expect(payloads[0]).toMatchObject({
      execution_id: 'exec-1',
      event_type: 'execution.started'
    });
  });

  it('enriches logs from the current execution context automatically', () => {
    const payloads: any[] = [];
    const logger = createExecutionLogger({
      contextProvider: () => {
        const metadata = ExecutionContext.getCurrent()?.metadata;
        if (!metadata) return undefined;
        return {
          executionId: metadata.executionId,
          runId: metadata.runId,
          workflowId: metadata.workflowId,
          rootWorkflowId: metadata.rootWorkflowId,
          parentExecutionId: metadata.parentExecutionId,
          reasonerId: metadata.reasonerId
        };
      },
      transport: {
        emit: (payload) => payloads.push(payload)
      },
      mirrorToStdout: false
    });

    const metadata: ExecutionMetadata = {
      executionId: 'exec-2',
      runId: 'run-2',
      workflowId: 'wf-2',
      rootWorkflowId: 'root-2',
      parentExecutionId: 'parent-2',
      reasonerId: 'reasoner-2'
    };
    const agent = {
      getExecutionLogger: () => logger
    } as unknown as Agent;

    ExecutionContext.run(
      new ExecutionContext({
        input: { hello: 'world' },
        metadata,
        req: {} as any,
        res: {} as any,
        agent
      }),
      () => {
        logger.info('contextual log', { answer: 42 });
      }
    );

    expect(payloads).toHaveLength(1);
    expect(payloads[0]).toMatchObject({
      execution_id: 'exec-2',
      run_id: 'run-2',
      workflow_id: 'wf-2',
      root_workflow_id: 'root-2',
      parent_execution_id: 'parent-2',
      reasoner_id: 'reasoner-2',
      message: 'contextual log',
      attributes: {
        answer: 42
      }
    });
  });
});

describe('Agent execution logging', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('emits structured runtime logs for a local reasoner execution', async () => {
    const logPayloads: any[] = [];
    vi.spyOn(process.stdout, 'write').mockImplementation(() => true);
    vi.spyOn(AgentFieldClient.prototype, 'publishWorkflowEvent').mockImplementation(() => {});
    vi.spyOn(AgentFieldClient.prototype, 'publishExecutionLogs').mockImplementation((payload) => {
      logPayloads.push(payload);
    });

    const agent = new Agent({ nodeId: 'local', devMode: true });
    agent.reasoner('echo', async (ctx) => {
      ctx.logger.info('inside reasoner', { value: ctx.input.value });
      return { echo: ctx.input.value };
    });

    const metadata: ExecutionMetadata = {
      executionId: 'parent-exec',
      runId: 'parent-run',
      workflowId: 'parent-workflow',
      rootWorkflowId: 'root-workflow',
      reasonerId: 'parent-reasoner'
    };

    const result = await ExecutionContext.run(
      new ExecutionContext({
        input: { value: 'parent' },
        metadata,
        req: {} as any,
        res: {} as any,
        agent
      }),
      async () => agent.call('local.echo', { value: 'hi' })
    );

    expect(result).toEqual({ echo: 'hi' });
    const eventTypes = logPayloads.flatMap((payload) =>
      'entries' in payload ? payload.entries.map((entry: any) => entry.event_type) : [payload.event_type]
    );
    expect(eventTypes).toContain('agent.call.started');
    expect(eventTypes).toContain('execution.started');
    expect(eventTypes).toContain('reasoner.started');
    expect(eventTypes).toContain('reasoner.completed');
    expect(eventTypes).toContain('execution.completed');
    expect(eventTypes).toContain('agent.call.completed');
    expect(logPayloads.some((payload) =>
      'entries' in payload
        ? payload.entries.some((entry: any) => entry.message === 'inside reasoner')
        : payload.message === 'inside reasoner'
    )).toBe(true);
  });
});
