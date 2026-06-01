import { describe, it, expect, beforeEach, vi } from 'vitest';
import axios from 'axios';
import { DidClient } from '../src/did/DidClient.js';
import { DidInterface } from '../src/did/DidInterface.js';

vi.mock('axios', () => {
  const create = vi.fn(() => ({
    post: vi.fn(),
    get: vi.fn()
  }));
  return {
    default: { create },
    create
  };
});

const getCreatedClient = () => {
  const mockCreate = (axios as any).create as ReturnType<typeof vi.fn>;
  const last = mockCreate.mock.results.at(-1);
  return last?.value;
};

describe('DidClient', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('generates an execution credential with serialized payload', async () => {
    const client = new DidClient('http://localhost:8080');
    const http = getCreatedClient();
    const post = vi.fn().mockResolvedValue({
      data: {
        vc_id: 'vc-1',
        execution_id: 'exec-1',
        workflow_id: 'wf-1',
        session_id: 'sess-1',
        issuer_did: 'issuer',
        target_did: 'target',
        caller_did: 'caller',
        vc_document: { proof: {} },
        signature: 'sig',
        input_hash: 'ih',
        output_hash: 'oh',
        status: 'succeeded',
        created_at: '2025-01-01T00:00:00Z'
      }
    });
    http.post = post;

    const result = await client.generateCredential({
      executionContext: {
        executionId: 'exec-1',
        workflowId: 'wf-1',
        sessionId: 'sess-1',
        callerDid: 'caller',
        targetDid: 'target',
        agentNodeDid: 'agent-1',
        timestamp: '2025-01-01T00:00:00Z'
      },
      inputData: { foo: 'bar' },
      outputData: { ok: true },
      status: 'succeeded',
      durationMs: 42
    });

    expect(post).toHaveBeenCalledWith(
      '/api/v1/execution/vc',
      expect.objectContaining({
        execution_context: expect.objectContaining({
          execution_id: 'exec-1',
          workflow_id: 'wf-1',
          session_id: 'sess-1',
          caller_did: 'caller',
          target_did: 'target',
          agent_node_did: 'agent-1',
          timestamp: '2025-01-01T00:00:00Z'
        }),
        status: 'succeeded',
        duration_ms: 42
      }),
      expect.any(Object)
    );
    const [, payload] = post.mock.calls[0] as any[];
    expect(payload.input_data).toBe(Buffer.from(JSON.stringify({ foo: 'bar' }), 'utf-8').toString('base64'));
    expect(payload.output_data).toBe(Buffer.from(JSON.stringify({ ok: true }), 'utf-8').toString('base64'));

    expect(result).toEqual(
      expect.objectContaining({
        vcId: 'vc-1',
        executionId: 'exec-1',
        workflowId: 'wf-1',
        callerDid: 'caller',
        targetDid: 'target'
      })
    );
  });

  it('exports audit trail with mapped keys', async () => {
    const client = new DidClient('http://localhost:8080');
    const http = getCreatedClient();
    http.get = vi.fn().mockResolvedValue({
      data: {
        agent_dids: ['did:a'],
        total_count: 1,
        execution_vcs: [
          {
            vc_id: 'vc-1',
            execution_id: 'exec-1',
            workflow_id: 'wf-1',
            caller_did: 'caller',
            target_did: 'target',
            status: 'succeeded',
            created_at: '2025-01-01T00:00:00Z'
          }
        ],
        workflow_vcs: [
          {
            workflow_id: 'wf-1',
            session_id: 'sess-1',
            component_vcs: ['vc-1'],
            workflow_vc_id: 'wvc-1',
            status: 'complete',
            start_time: '2025-01-01T00:00:00Z',
            end_time: '2025-01-01T00:00:01Z',
            total_steps: 1,
            completed_steps: 1
          }
        ]
      }
    });

    const result = await client.exportAuditTrail({ workflowId: 'wf-1', status: 'succeeded' });

    expect(http.get).toHaveBeenCalledWith(
      '/api/v1/did/export/vcs',
      expect.objectContaining({
        params: expect.objectContaining({ workflow_id: 'wf-1', status: 'succeeded' })
      })
    );
    expect(result.executionVcs[0].vcId).toBe('vc-1');
    expect(result.workflowVcs[0].workflowVcId).toBe('wvc-1');
    expect(result.agentDids).toEqual(['did:a']);
  });
});

describe('DidInterface', () => {
  it('uses execution metadata for defaults', async () => {
    const generateCredential = vi.fn();
    const did = new DidInterface({
      client: { generateCredential } as any,
      metadata: {
        executionId: 'exec-1',
        runId: 'run-1',
        workflowId: 'wf-1',
        sessionId: 'sess-1',
        callerDid: 'caller',
        targetDid: 'target',
        agentNodeDid: 'agent'
      },
      enabled: true,
      defaultInput: { foo: 'bar' }
    });

    await did.generateCredential({ outputData: { ok: true }, status: 'succeeded', durationMs: 10 });

    expect(generateCredential).toHaveBeenCalledWith(
      expect.objectContaining({
        executionContext: expect.objectContaining({
          executionId: 'exec-1',
          workflowId: 'wf-1',
          sessionId: 'sess-1',
          callerDid: 'caller',
          targetDid: 'target',
          agentNodeDid: 'agent'
        }),
        inputData: { foo: 'bar' },
        outputData: { ok: true },
        status: 'succeeded',
        durationMs: 10
      })
    );
  });

  it('throws when disabled', async () => {
    const did = new DidInterface({
      client: {} as any,
      metadata: { executionId: 'exec-1' } as any,
      enabled: false
    });

    await expect(did.generateCredential()).rejects.toThrow(/disabled/);
  });
});
