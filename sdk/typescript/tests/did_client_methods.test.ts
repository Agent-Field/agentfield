import { beforeEach, describe, expect, it, vi } from 'vitest';
import axios from 'axios';
import { DidClient } from '../src/did/DidClient.js';

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

type MockHttpClient = {
  post: ReturnType<typeof vi.fn>;
  get: ReturnType<typeof vi.fn>;
};

const getCreatedClient = (): MockHttpClient => {
  const mockCreate = (axios as unknown as { create: ReturnType<typeof vi.fn> }).create;
  const last = mockCreate.mock.results.at(-1)?.value as MockHttpClient | undefined;
  if (!last) {
    throw new Error('expected axios.create to have been called');
  }
  return last;
};

describe('DidClient exported methods', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('registers an agent and maps the returned identity package', async () => {
    const client = new DidClient('http://localhost:8080/', {
      Authorization: 'Bearer token',
      'X-Enabled': true,
      'X-Count': 2,
      'X-Skip': undefined
    });
    const http = getCreatedClient();
    http.post.mockResolvedValue({
      data: {
        success: true,
        message: 'ok',
        identity_package: {
          agent_did: {
            did: 'did:agent',
            private_key_jwk: 'private',
            public_key_jwk: 'public',
            derivation_path: 'm/0',
            component_type: 'agent'
          },
          reasoner_dids: {
            planner: {
              did: 'did:planner',
              public_key_jwk: 'planner-public',
              derivation_path: 'm/1',
              component_type: 'reasoner',
              function_name: 'plan'
            }
          },
          skill_dids: {
            summarize: {
              did: 'did:skill',
              public_key_jwk: 'skill-public',
              derivation_path: 'm/2',
              component_type: 'skill'
            }
          },
          agentfield_server_id: 'srv-1'
        }
      }
    });

    const result = await client.registerAgent({
      agentNodeId: 'agent-node',
      reasoners: [{ id: 'planner' }],
      skills: [{ id: 'summarize' }]
    });

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/did/register',
      {
        agent_node_id: 'agent-node',
        reasoners: [{ id: 'planner' }],
        skills: [{ id: 'summarize' }]
      },
      {
        headers: {
          Authorization: 'Bearer token',
          'X-Enabled': 'true',
          'X-Count': '2'
        }
      }
    );
    expect(result).toEqual({
      success: true,
      message: 'ok',
      identityPackage: {
        agentDid: {
          did: 'did:agent',
          privateKeyJwk: 'private',
          publicKeyJwk: 'public',
          derivationPath: 'm/0',
          componentType: 'agent',
          functionName: undefined
        },
        reasonerDids: {
          planner: {
            did: 'did:planner',
            privateKeyJwk: undefined,
            publicKeyJwk: 'planner-public',
            derivationPath: 'm/1',
            componentType: 'reasoner',
            functionName: 'plan'
          }
        },
        skillDids: {
          summarize: {
            did: 'did:skill',
            privateKeyJwk: undefined,
            publicKeyJwk: 'skill-public',
            derivationPath: 'm/2',
            componentType: 'skill',
            functionName: undefined
          }
        },
        agentfieldServerId: 'srv-1'
      }
    });
  });

  it('returns a normalized failure when registration is unsuccessful', async () => {
    const client = new DidClient('http://localhost:8080');
    const http = getCreatedClient();
    http.post.mockResolvedValue({ data: { success: false } });

    await expect(
      client.registerAgent({
        agentNodeId: 'agent-node',
        reasoners: [],
        skills: []
      })
    ).resolves.toEqual({
      success: false,
      error: 'DID registration failed'
    });
  });

  it('generates credentials with normalized payload data and merged headers', async () => {
    const client = new DidClient('http://localhost:8080', {
      Authorization: 'Bearer token'
    });
    const http = getCreatedClient();
    http.post.mockResolvedValue({
      data: {
        vc_id: 'vc-1',
        execution_id: 'exec-1',
        workflow_id: 'wf-1',
        session_id: 'sess-1',
        issuer_did: 'issuer',
        target_did: 'target',
        caller_did: 'caller',
        vc_document: { proof: true },
        signature: 'sig',
        input_hash: 'in-hash',
        output_hash: 'out-hash',
        status: 'failed',
        created_at: '2025-01-01T00:00:00Z'
      }
    });

    const outputBytes = new Uint8Array([111, 107]);
    const result = await client.generateCredential({
      executionContext: {
        executionId: 'exec-1',
        workflowId: 'wf-1',
        sessionId: 'sess-1',
        callerDid: 'caller',
        targetDid: 'target',
        agentNodeDid: 'did:agent',
        timestamp: new Date('2025-01-01T00:00:00Z')
      },
      inputData: 'plain-text',
      outputData: outputBytes,
      status: 'failed',
      errorMessage: 'boom',
      headers: {
        'X-Request-ID': 7
      }
    });

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/execution/vc',
      {
        execution_context: {
          execution_id: 'exec-1',
          workflow_id: 'wf-1',
          session_id: 'sess-1',
          caller_did: 'caller',
          target_did: 'target',
          agent_node_did: 'did:agent',
          timestamp: '2025-01-01T00:00:00.000Z'
        },
        input_data: Buffer.from('plain-text', 'utf-8').toString('base64'),
        output_data: Buffer.from(Buffer.from(outputBytes).toString('utf-8'), 'utf-8').toString('base64'),
        status: 'failed',
        error_message: 'boom',
        duration_ms: 0
      },
      {
        headers: {
          Authorization: 'Bearer token',
          'X-Request-ID': '7'
        }
      }
    );
    expect(result).toEqual({
      vcId: 'vc-1',
      executionId: 'exec-1',
      workflowId: 'wf-1',
      sessionId: 'sess-1',
      issuerDid: 'issuer',
      targetDid: 'target',
      callerDid: 'caller',
      vcDocument: { proof: true },
      signature: 'sig',
      inputHash: 'in-hash',
      outputHash: 'out-hash',
      status: 'failed',
      createdAt: '2025-01-01T00:00:00Z'
    });
  });

  it('verifies credentials with the raw verifier payload', async () => {
    const client = new DidClient('http://localhost:8080', {
      Authorization: 'Bearer token'
    });
    const http = getCreatedClient();
    http.post.mockResolvedValue({
      data: {
        valid: true,
        issuer_did: 'did:example:issuer',
        message: 'verified'
      }
    });

    const vcDocument = {
      id: 'urn:agentfield:vc:vc-1',
      proof: { proofValue: 'embedded-signature' }
    };
    const result = await client.verifyCredential(vcDocument);

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/did/verify',
      { vc_document: vcDocument },
      {
        headers: {
          Authorization: 'Bearer token'
        }
      }
    );
    expect(result).toEqual({
      valid: true,
      issuer_did: 'did:example:issuer',
      message: 'verified'
    });
  });

  it('verifies credential without extra headers', async () => {
    const client = new DidClient('http://localhost:8080');
    const http = getCreatedClient();

    http.post.mockResolvedValue({
      data: { valid: true }
    });

    const vcDocument = { id: 'vc-1' };

    const result = await client.verifyCredential(vcDocument);

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/did/verify',
      { vc_document: vcDocument },
      { headers: {} }
    );

    expect(result).toEqual({ valid: true });
  });

  
  it('gets the raw workflow VC chain response', async () => {
    const client = new DidClient('http://localhost:8080');
    const http = getCreatedClient();
    http.get.mockResolvedValue({
      data: {
        workflow_id: 'wf-1',
        component_vcs: [
          {
            vc_id: 'vc-1',
            execution_id: 'exec-1'
          }
        ],
        workflow_vc: {
          workflow_id: 'wf-1',
          workflow_vc_id: 'wvc-1'
        },
        total_steps: 1,
        status: 'succeeded'
      }
    });

    const result = await client.getWorkflowVcChain('wf-1');

    expect(http.get).toHaveBeenCalledWith(
      '/api/v1/did/workflow/wf-1/vc-chain',
      expect.objectContaining({ headers: expect.any(Object) })
    );
    expect(result).toEqual({
      workflow_id: 'wf-1',
      component_vcs: [
        {
          vc_id: 'vc-1',
          execution_id: 'exec-1'
        }
      ],
      workflow_vc: {
        workflow_id: 'wf-1',
        workflow_vc_id: 'wvc-1'
      },
      total_steps: 1,
      status: 'succeeded'
    });
  });

  it('creates a workflow VC from workflow metadata and execution VC ids', async () => {
    const client = new DidClient('http://localhost:8080', {
      Authorization: 'Bearer token'
    });
    const http = getCreatedClient();
    http.post.mockResolvedValue({
      data: {
        workflow_id: 'wf-1',
        session_id: 'sess-1',
        component_vcs: ['vc-1', 'vc-2'],
        workflow_vc_id: 'wvc-1',
        status: 'succeeded',
        start_time: '2025-01-01T00:00:00Z',
        end_time: '2025-01-01T00:00:01Z',
        total_steps: 2,
        completed_steps: 2
      }
    });

    const result = await client.createWorkflowVc('wf-1', 'sess-1', ['vc-1', 'vc-2']);

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/did/workflow/wf-1/vc',
      {
        session_id: 'sess-1',
        execution_vc_ids: ['vc-1', 'vc-2']
      },
      {
        headers: {
          Authorization: 'Bearer token'
        }
      }
    );
    expect(result).toEqual({
      workflowId: 'wf-1',
      sessionId: 'sess-1',
      componentVcs: ['vc-1', 'vc-2'],
      workflowVcId: 'wvc-1',
      status: 'succeeded',
      startTime: '2025-01-01T00:00:00Z',
      endTime: '2025-01-01T00:00:01Z',
      totalSteps: 2,
      completedSteps: 2
    });
  });

  it('exports an audit trail with cleaned filters and fallback workflow VC ids', async () => {
    const client = new DidClient('http://localhost:8080');
    const http = getCreatedClient();
    http.get.mockResolvedValue({
      data: {
        agent_dids: ['did:a'],
        execution_vcs: [
          {
            vc_id: 'vc-1',
            execution_id: 'exec-1',
            workflow_id: 'wf-1',
            session_id: 'sess-1',
            issuer_did: 'issuer',
            target_did: 'target',
            caller_did: 'caller',
            status: 'succeeded',
            created_at: '2025-01-01T00:00:00Z'
          }
        ],
        workflow_vcs: [
          {
            workflow_id: 'wf-1',
            session_id: 'sess-1',
            component_vcs: ['vc-1'],
            status: 'completed',
            start_time: '2025-01-01T00:00:00Z',
            total_steps: 3,
            completed_steps: 2
          }
        ],
        total_count: 1,
        filters_applied: {
          workflow_id: 'wf-1'
        }
      }
    });

    const result = await client.exportAuditTrail({
      workflowId: 'wf-1',
      sessionId: 'sess-1',
      issuerDid: 'issuer',
      status: 'succeeded',
      limit: 10
    });

    expect(http.get).toHaveBeenCalledWith('/api/v1/did/export/vcs', {
      params: {
        workflow_id: 'wf-1',
        session_id: 'sess-1',
        issuer_did: 'issuer',
        status: 'succeeded',
        limit: 10
      },
      headers: {}
    });
    expect(result).toEqual({
      agentDids: ['did:a'],
      executionVcs: [
        {
          vcId: 'vc-1',
          executionId: 'exec-1',
          workflowId: 'wf-1',
          sessionId: 'sess-1',
          issuerDid: 'issuer',
          targetDid: 'target',
          callerDid: 'caller',
          status: 'succeeded',
          createdAt: '2025-01-01T00:00:00Z'
        }
      ],
      workflowVcs: [
        {
          workflowId: 'wf-1',
          sessionId: 'sess-1',
          componentVcs: ['vc-1'],
          workflowVcId: 'wf-1',
          status: 'completed',
          startTime: '2025-01-01T00:00:00Z',
          endTime: undefined,
          totalSteps: 3,
          completedSteps: 2
        }
      ],
      totalCount: 1,
      filtersApplied: {
        workflow_id: 'wf-1'
      }
    });
  });
});
