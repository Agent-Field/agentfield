import express from 'express';
import { randomUUID } from 'node:crypto';
import http from 'node:http';
import axios from 'axios';
import { WebSocketServer } from 'ws';
import { beforeAll, afterAll, describe, expect, it } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import { AgentFieldClient } from '../src/client/AgentFieldClient.js';
import { DidClient } from '../src/did/DidClient.js';

type MemoryEntry = { key: string; value: any; scope: string; scopeId?: string };
type VectorEntry = { key: string; embedding: number[]; scope: string; scopeId?: string };
type DidExecutionVcRequest = { body: any; headers: http.IncomingHttpHeaders };

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(predicate: () => boolean, timeoutMs = 3000, intervalMs = 25) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (predicate()) return;
    await sleep(intervalMs);
  }
  throw new Error('Timed out waiting for condition');
}

function encodeExpectedJson(value: Record<string, unknown>) {
  return Buffer.from(JSON.stringify(value, Object.keys(value).sort()), 'utf-8').toString('base64');
}

async function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = http.createServer();
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (address && typeof address === 'object') {
        const port = address.port;
        server.close(() => resolve(port));
      } else {
        reject(new Error('Failed to acquire test port'));
      }
    });
    server.on('error', reject);
  });
}

function resolveScopeId(scope: string | undefined, headers: http.IncomingHttpHeaders) {
  const pick = (key: string) => {
    const value = headers[key];
    return Array.isArray(value) ? value[0] : value;
  };

  if (scope === 'session') return pick('x-session-id');
  if (scope === 'actor') return pick('x-actor-id');
  if (scope === 'global') return 'global';
  return pick('x-workflow-id') ?? pick('x-run-id');
}

function forwardMetadataHeaders(headers: http.IncomingHttpHeaders) {
  const allowed = [
    'x-run-id',
    'x-workflow-id',
    'x-session-id',
    'x-actor-id',
    'x-execution-id',
    'x-parent-execution-id',
    'x-caller-did',
    'x-target-did',
    'x-agent-node-did',
    'x-agent-node-id'
  ];
  const forwarded: Record<string, string> = {};
  for (const key of allowed) {
    const value = headers[key];
    if (value !== undefined) {
      forwarded[key] = Array.isArray(value) ? value[0] : String(value);
    }
  }
  return forwarded;
}

async function createControlPlaneStub() {
  const app = express();
  app.use(express.json());

  const registrations: any[] = [];
  const heartbeats: Array<{ nodeId: string; status?: string }> = [];
  const workflowEvents: any[] = [];
  const executionStatuses: any[] = [];
  const memory: MemoryEntry[] = [];
  const vectors: VectorEntry[] = [];
  const didRegistrations: any[] = [];
  const executionVcRequests: DidExecutionVcRequest[] = [];
  const executionVcs: any[] = [];
  const workflowVcs: any[] = [];
  const agents = new Map<string, { baseUrl: string; reasoners: any[]; skills: any[] }>();

  const server = http.createServer(app);
  const wss = new WebSocketServer({ server, path: '/api/v1/memory/events/ws' });
  const sockets = new Set<any>();

  wss.on('connection', (socket) => {
    sockets.add(socket);
    socket.on('close', () => sockets.delete(socket));
  });

  const findMemory = (scope: string, scopeId: string | undefined, key: string) =>
    memory.find((entry) => entry.scope === scope && entry.scopeId === scopeId && entry.key === key);

  app.post('/api/v1/nodes/register', (req, res) => {
    const payload = req.body ?? {};
    registrations.push(payload);
    agents.set(payload.id, {
      baseUrl: payload.base_url ?? payload.public_url,
      reasoners: payload.reasoners ?? [],
      skills: payload.skills ?? []
    });
    res.json({ ok: true });
  });

  app.post('/api/v1/nodes/:id/heartbeat', (req, res) => {
    heartbeats.push({ nodeId: req.params.id, status: req.body?.status });
    res.json({ status: req.body?.status ?? 'ready', nodeId: req.params.id });
  });

  app.post('/api/v1/did/register', (req, res) => {
    const payload = req.body ?? {};
    didRegistrations.push(payload);
    const privateSeed = Buffer.alloc(32, 7).toString('base64url');
    const publicKey = Buffer.alloc(32, 9).toString('base64url');
    const reasonerDids = Object.fromEntries(
      (payload.reasoners ?? []).map((reasoner: any, index: number) => [
        reasoner.id,
        {
          did: `did:example:reasoner:${reasoner.id}`,
          public_key_jwk: JSON.stringify({
            kty: 'OKP',
            crv: 'Ed25519',
            x: publicKey
          }),
          derivation_path: `m/${index + 1}`,
          component_type: 'reasoner',
          function_name: reasoner.id
        }
      ])
    );
    const skillDids = Object.fromEntries(
      (payload.skills ?? []).map((skill: any, index: number) => [
        skill.id,
        {
          did: `did:example:skill:${skill.id}`,
          public_key_jwk: JSON.stringify({
            kty: 'OKP',
            crv: 'Ed25519',
            x: publicKey
          }),
          derivation_path: `m/${index + 101}`,
          component_type: 'skill'
        }
      ])
    );

    res.json({
      success: true,
      identity_package: {
        agent_did: {
          did: `did:example:agent:${payload.agent_node_id ?? 'agent'}`,
          private_key_jwk: JSON.stringify({
            kty: 'OKP',
            crv: 'Ed25519',
            d: privateSeed,
            x: publicKey
          }),
          public_key_jwk: JSON.stringify({
            kty: 'OKP',
            crv: 'Ed25519',
            x: publicKey
          }),
          derivation_path: 'm/0',
          component_type: 'agent'
        },
        reasoner_dids: reasonerDids,
        skill_dids: skillDids,
        agentfield_server_id: 'stub-control-plane'
      }
    });
  });

  app.post('/api/v1/workflow/executions/events', (req, res) => {
    workflowEvents.push(req.body);
    res.json({ ok: true });
  });

  app.post('/api/v1/executions/:id/status', (req, res) => {
    executionStatuses.push({ id: req.params.id, body: req.body });
    res.json({ ok: true });
  });

  app.post('/api/v1/execution/vc', (req, res) => {
    const payload = req.body ?? {};
    executionVcRequests.push({ body: payload, headers: req.headers });
    const executionContext = payload.execution_context ?? {};
    const vcId = `vc-${executionVcs.length + 1}`;
    const signature = `sig-${vcId}`;
    const createdAt = new Date().toISOString();

    const vc = {
      vc_id: vcId,
      execution_id: executionContext.execution_id ?? `exec-${randomUUID()}`,
      workflow_id: executionContext.workflow_id ?? '',
      session_id: executionContext.session_id ?? '',
      issuer_did: executionContext.caller_did ?? 'did:example:issuer',
      target_did: executionContext.target_did ?? 'did:example:target',
      caller_did: executionContext.caller_did ?? 'did:example:caller',
      vc_document: {
        id: `urn:agentfield:vc:${vcId}`,
        issuer: executionContext.caller_did ?? 'did:example:issuer',
        proof: {
          type: 'Ed25519Signature2020',
          proofValue: signature
        },
        credentialSubject: {
          execution_id: executionContext.execution_id ?? '',
          workflow_id: executionContext.workflow_id ?? '',
          session_id: executionContext.session_id ?? ''
        }
      },
      signature,
      input_hash: payload.input_data ?? '',
      output_hash: payload.output_data ?? '',
      status: payload.status ?? 'succeeded',
      created_at: createdAt
    };

    executionVcs.push(vc);
    res.json(vc);
  });

  app.post('/api/v1/did/verify', (req, res) => {
    const vcDocument = req.body?.vc_document;
    const suppliedSignature = req.body?.signature ?? vcDocument?.proof?.proofValue;
    const stored = executionVcs.find((vc) => vc.vc_document?.id === vcDocument?.id);
    const valid = Boolean(stored && suppliedSignature === stored.signature);

    res.json({
      valid,
      issuer_did: stored?.issuer_did ?? vcDocument?.issuer ?? '',
      message: valid ? 'verified' : 'invalid signature'
    });
  });

  app.post('/api/v1/did/workflow/:workflowId/vc', (req, res) => {
    const workflowId = req.params.workflowId;
    const sessionId = req.body?.session_id ?? '';
    const executionVcIds = Array.isArray(req.body?.execution_vc_ids) ? req.body.execution_vc_ids : [];
    const components = executionVcIds
      .map((vcId: string) => executionVcs.find((vc) => vc.vc_id === vcId))
      .filter(Boolean);

    if (components.length !== executionVcIds.length) {
      res.status(404).json({ error: 'execution VC not found' });
      return;
    }

    const workflowVc = {
      workflow_id: workflowId,
      session_id: sessionId,
      component_vcs: executionVcIds,
      workflow_vc_id: `wvc-${workflowVcs.length + 1}`,
      status: 'succeeded',
      start_time: components[0]?.created_at ?? new Date().toISOString(),
      end_time: new Date().toISOString(),
      total_steps: executionVcIds.length,
      completed_steps: executionVcIds.length
    };

    workflowVcs.push(workflowVc);
    res.json(workflowVc);
  });

  app.get('/api/v1/did/workflow/:workflowId/vc-chain', (req, res) => {
    const workflowId = req.params.workflowId;
    const workflowVc = [...workflowVcs].reverse().find((vc) => vc.workflow_id === workflowId);

    if (!workflowVc) {
      res.status(404).json({ error: 'workflow VC not found' });
      return;
    }

    const componentVcs = workflowVc.component_vcs
      .map((vcId: string) => executionVcs.find((vc) => vc.vc_id === vcId))
      .filter(Boolean);

    res.json({
      workflow_id: workflowId,
      component_vcs: componentVcs,
      workflow_vc: workflowVc,
      total_steps: workflowVc.total_steps,
      status: workflowVc.status
    });
  });

  app.get('/api/v1/discovery/capabilities', (_req, res) => {
    const capabilities = Array.from(agents.entries()).map(([agentId, info]) => ({
      agent_id: agentId,
      base_url: info.baseUrl,
      version: '',
      health_status: 'running',
      deployment_type: 'long_running',
      last_heartbeat: heartbeats.find((hb) => hb.nodeId === agentId)?.status,
      reasoners: info.reasoners.map((r: any) => ({
        id: r.id,
        invocation_target: `${agentId}.${r.id}`,
        tags: r.tags ?? [],
        input_schema: r.input_schema ?? {},
        output_schema: r.output_schema ?? {}
      })),
      skills: info.skills.map((s: any) => ({
        id: s.id,
        invocation_target: `${agentId}.${s.id}`,
        tags: s.tags ?? [],
        input_schema: s.input_schema ?? {}
      }))
    }));

    const totalReasoners = capabilities.reduce((total, cap) => total + cap.reasoners.length, 0);
    const totalSkills = capabilities.reduce((total, cap) => total + cap.skills.length, 0);

    res.json({
      discovered_at: new Date().toISOString(),
      total_agents: capabilities.length,
      total_reasoners: totalReasoners,
      total_skills: totalSkills,
      pagination: { limit: capabilities.length, offset: 0, has_more: false },
      capabilities
    });
  });

  app.post('/api/v1/execute/:target', async (req, res) => {
    const rawTarget = req.params.target;
    const [agentIdMaybe, nameMaybe] = rawTarget.includes('.') ? rawTarget.split('.', 2) : [undefined, rawTarget];
    const agentId = agentIdMaybe ?? registrations.at(-1)?.id;
    const name = nameMaybe ?? rawTarget;
    const agentInfo = agentId ? agents.get(agentId) : undefined;

    if (!agentInfo) {
      res.status(404).json({ error: 'Agent not registered' });
      return;
    }

    const targetType = agentInfo.skills.some((s) => s.id === name) ? 'skill' : 'reasoner';
    const path = targetType === 'skill' ? `/api/v1/skills/${name}` : `/api/v1/reasoners/${name}`;

    try {
      const response = await axios.post(`${agentInfo.baseUrl}${path}`, req.body?.input ?? {}, {
        headers: forwardMetadataHeaders(req.headers)
      });
      res.json({ result: response.data });
    } catch (err: any) {
      res.status(err?.response?.status ?? 500).json(err?.response?.data ?? { error: 'Forward failed' });
    }
  });

  app.post('/api/v1/memory/set', (req, res) => {
    const scope = req.body?.scope ?? 'workflow';
    const scopeId = resolveScopeId(scope, req.headers);
    const existing = findMemory(scope, scopeId, req.body?.key);
    if (existing) {
      existing.value = req.body?.data;
    } else {
      memory.push({ key: req.body?.key, value: req.body?.data, scope, scopeId });
    }
    res.json({ ok: true });
  });

  app.post('/api/v1/memory/get', (req, res) => {
    const scope = req.body?.scope ?? 'workflow';
    const scopeId = resolveScopeId(scope, req.headers);
    const entry = findMemory(scope, scopeId, req.body?.key);
    if (!entry) {
      res.status(404).json({ error: 'not found' });
      return;
    }
    res.json({ data: entry.value });
  });

  app.post('/api/v1/memory/delete', (req, res) => {
    const scope = req.body?.scope ?? 'workflow';
    const scopeId = resolveScopeId(scope, req.headers);
    const idx = memory.findIndex(
      (entry) => entry.scope === scope && entry.scopeId === scopeId && entry.key === req.body?.key
    );
    if (idx >= 0) memory.splice(idx, 1);
    res.json({ ok: true });
  });

  app.get('/api/v1/memory/list', (req, res) => {
    const scope = String(req.query.scope ?? 'workflow');
    res.json(memory.filter((entry) => entry.scope === scope).map((entry) => ({ key: entry.key })));
  });

  app.post('/api/v1/memory/vector/set', (req, res) => {
    const scope = req.body?.scope ?? 'workflow';
    const scopeId = resolveScopeId(scope, req.headers);
    const existing = vectors.find(
      (entry) => entry.scope === scope && entry.scopeId === scopeId && entry.key === req.body?.key
    );
    if (existing) {
      existing.embedding = req.body?.embedding ?? [];
    } else {
      vectors.push({
        key: req.body?.key,
        embedding: req.body?.embedding ?? [],
        scope,
        scopeId
      });
    }
    res.json({ ok: true });
  });

  app.post('/api/v1/memory/vector/search', (req, res) => {
    const scope = req.body?.scope ?? 'workflow';
    const scopeId = resolveScopeId(scope, req.headers);
    const matches = vectors
      .filter((entry) => entry.scope === scope && entry.scopeId === scopeId)
      .slice(0, req.body?.top_k ?? 10)
      .map((entry) => ({
        key: entry.key,
        scope: entry.scope,
        scopeId: entry.scopeId ?? '',
        score: 1.0
      }));
    res.json(matches);
  });

  app.post('/api/v1/memory/vector/delete', (req, res) => {
    const scope = req.body?.scope ?? 'workflow';
    const scopeId = resolveScopeId(scope, req.headers);
    const idx = vectors.findIndex(
      (entry) => entry.scope === scope && entry.scopeId === scopeId && entry.key === req.body?.key
    );
    if (idx >= 0) vectors.splice(idx, 1);
    res.json({ ok: true });
  });

  const port = await getFreePort();
  await new Promise<void>((resolve) => server.listen(port, '127.0.0.1', () => resolve()));

  return {
    url: `http://127.0.0.1:${port}`,
    registrations,
    didRegistrations,
    heartbeats,
    workflowEvents,
    executionStatuses,
    memory,
    executionVcRequests,
    executionVcs,
    workflowVcs,
    stop: async () => {
      sockets.forEach((socket) => socket.close());
      await new Promise<void>((resolve) => wss.close(() => resolve()));
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  };
}

describe('TypeScript SDK integration', () => {
  let control: Awaited<ReturnType<typeof createControlPlaneStub>>;
  let agent: Agent;
  let client: AgentFieldClient;
  let agentPort: number;

  beforeAll(async () => {
    control = await createControlPlaneStub();
    agentPort = await getFreePort();

    agent = new Agent({
      nodeId: 'ts-e2e-agent',
      port: agentPort,
      host: '127.0.0.1',
      agentFieldUrl: control.url,
      heartbeatIntervalMs: 20,
      devMode: false
    });

    agent.reasoner('echo', async (ctx) => {
      await ctx.memory.set('last_input', ctx.input.message);
      const stored = await ctx.memory.get('last_input');
      return {
        echoed: ctx.input.message,
        stored,
        workflowId: ctx.workflowId,
        runId: ctx.runId
      };
    });

    agent.reasoner('issueVc', async (ctx) => {
      const credential = await ctx.did.generateCredential({
        outputData: { acknowledged: true, step: ctx.input.step },
        status: 'succeeded',
        durationMs: 12
      });

      return {
        executionId: ctx.executionId,
        workflowId: ctx.workflowId,
        sessionId: ctx.sessionId,
        reasonerId: ctx.reasonerId,
        vcId: credential.vcId,
        vcDocument: credential.vcDocument,
        signature: credential.signature,
        inputHash: credential.inputHash,
        outputHash: credential.outputHash
      };
    });

    agent.skill('greet', (ctx) => ({ greeting: `hello ${ctx.input.name}` }));

    await agent.serve();
    client = new AgentFieldClient({ nodeId: 'ts-e2e-client', agentFieldUrl: control.url });
  }, 20000);

  afterAll(async () => {
    if (agent) await agent.shutdown();
    if (control) await control.stop();
  });

  it('registers with the control plane and surfaces capabilities', async () => {
    await waitFor(() => control.registrations.length > 0);

    const registration = control.registrations.at(-1);
    expect(registration.reasoners.map((r: any) => r.id)).toContain('echo');
    expect(control.heartbeats.some((hb) => hb.status === 'starting')).toBe(true);

    const discovery = await client.discoverCapabilities({ agent: 'ts-e2e-agent' });
    const capability = discovery.json?.capabilities.find((cap) => cap.agentId === 'ts-e2e-agent');

    expect(capability?.reasoners.map((r) => r.id)).toContain('echo');
    expect(capability?.skills.map((s) => s.id)).toContain('greet');
  });

  it('executes through the control plane and persists memory', async () => {
    const result = await client.execute<{ echoed: string; stored: string; workflowId: string; runId: string }>(
      'ts-e2e-agent.echo',
      { message: 'integration-hello' },
      { runId: 'run-42', workflowId: 'wf-42' }
    );

    expect(result.echoed).toBe('integration-hello');
    expect(result.stored).toBe('integration-hello');
    expect(result.workflowId).toBe('wf-42');

    const stored = control.memory.find(
      (entry) => entry.key === 'last_input' && entry.scope === 'workflow' && entry.scopeId === 'wf-42'
    );
    expect(stored?.value).toBe('integration-hello');
  });

  it('publishes workflow events for local agent.call executions', async () => {
    const response = await agent.call('ts-e2e-agent.echo', { message: 'local-hop' });
    expect(response.echoed).toBe('local-hop');

    await waitFor(() => control.workflowEvents.length >= 2);
    const lastEvents = control.workflowEvents.slice(-2);

    expect(lastEvents[0].status).toBe('running');
    expect(lastEvents[1].status).toBe('succeeded');
    expect(lastEvents[1].reasoner_id).toBe('echo');
    expect(lastEvents[1].agent_node_id).toBe('ts-e2e-agent');
  });

  it('registers DID identities and issues workflow VCs through ctx.did during live executions', async () => {
    await waitFor(() => control.didRegistrations.length > 0);

    const didRegistration = control.didRegistrations.at(-1);
    expect(didRegistration.agent_node_id).toBe('ts-e2e-agent');
    expect(didRegistration.reasoners.map((reasoner: any) => reasoner.id)).toEqual(
      expect.arrayContaining(['echo', 'issueVc'])
    );
    expect(didRegistration.skills.map((skill: any) => skill.id)).toContain('greet');

    const did = new DidClient(control.url);
    const workflowId = `wf-did-${randomUUID()}`;
    const sessionId = `sess-did-${randomUUID()}`;
    const callerDid = 'did:example:caller:integration';
    const expectedAgentDid = 'did:example:agent:ts-e2e-agent';
    const expectedReasonerDid = 'did:example:reasoner:issueVc';
    const requestCountBefore = control.executionVcRequests.length;

    const vc1 = await client.execute<{
      executionId: string;
      workflowId: string;
      sessionId: string;
      reasonerId: string;
      vcId: string;
      vcDocument: any;
      signature: string;
      inputHash: string;
      outputHash: string;
    }>(
      'ts-e2e-agent.issueVc',
      { step: 1, label: 'first' },
      { runId: 'run-did-1', workflowId, sessionId, callerDid }
    );

    const vc2 = await client.execute<{
      executionId: string;
      workflowId: string;
      sessionId: string;
      reasonerId: string;
      vcId: string;
      vcDocument: any;
      signature: string;
      inputHash: string;
      outputHash: string;
    }>(
      'ts-e2e-agent.issueVc',
      { step: 2, label: 'second' },
      { runId: 'run-did-2', workflowId, sessionId, callerDid }
    );

    await waitFor(() => control.executionVcRequests.length === requestCountBefore + 2);
    const [firstRequest, secondRequest] = control.executionVcRequests.slice(requestCountBefore);

    expect(vc1.vcId).toBeTruthy();
    expect(vc2.vcId).toBeTruthy();
    expect(vc1.workflowId).toBe(workflowId);
    expect(vc2.workflowId).toBe(workflowId);
    expect(vc1.sessionId).toBe(sessionId);
    expect(vc2.sessionId).toBe(sessionId);
    expect(vc1.reasonerId).toBe('issueVc');
    expect(vc2.reasonerId).toBe('issueVc');

    const expectedFirstInputHash = encodeExpectedJson({ label: 'first', step: 1 });
    const expectedSecondInputHash = encodeExpectedJson({ label: 'second', step: 2 });
    const expectedFirstOutputHash = encodeExpectedJson({ acknowledged: true, step: 1 });
    const expectedSecondOutputHash = encodeExpectedJson({ acknowledged: true, step: 2 });

    expect(firstRequest.body.execution_context).toMatchObject({
      execution_id: vc1.executionId,
      workflow_id: workflowId,
      session_id: sessionId,
      caller_did: callerDid,
      target_did: expectedReasonerDid,
      agent_node_did: expectedAgentDid
    });
    expect(secondRequest.body.execution_context).toMatchObject({
      execution_id: vc2.executionId,
      workflow_id: workflowId,
      session_id: sessionId,
      caller_did: callerDid,
      target_did: expectedReasonerDid,
      agent_node_did: expectedAgentDid
    });
    expect(firstRequest.body.input_data).toBe(expectedFirstInputHash);
    expect(secondRequest.body.input_data).toBe(expectedSecondInputHash);
    expect(firstRequest.body.output_data).toBe(expectedFirstOutputHash);
    expect(secondRequest.body.output_data).toBe(expectedSecondOutputHash);
    expect(firstRequest.body.status).toBe('succeeded');
    expect(secondRequest.body.status).toBe('succeeded');
    expect(firstRequest.body.duration_ms).toBe(12);
    expect(secondRequest.body.duration_ms).toBe(12);

    expect(vc1.inputHash).toBe(expectedFirstInputHash);
    expect(vc2.inputHash).toBe(expectedSecondInputHash);
    expect(vc1.outputHash).toBe(expectedFirstOutputHash);
    expect(vc2.outputHash).toBe(expectedSecondOutputHash);
    expect(vc1.vcDocument.issuer).toBe(callerDid);
    expect(vc2.vcDocument.issuer).toBe(callerDid);
    expect(vc1.vcDocument.credentialSubject.execution_id).toBe(vc1.executionId);
    expect(vc2.vcDocument.credentialSubject.execution_id).toBe(vc2.executionId);

    const verification = await did.verifyCredential(vc1.vcDocument);
    expect(verification.valid).toBe(true);
    expect(verification.issuer_did).toBe(callerDid);

    const workflowVc = await did.createWorkflowVc(workflowId, sessionId, [vc1.vcId, vc2.vcId]);
    expect(workflowVc).not.toBeNull();
    expect(workflowVc?.workflowId).toBe(workflowId);
    expect(workflowVc?.sessionId).toBe(sessionId);
    expect(workflowVc?.componentVcs).toEqual([vc1.vcId, vc2.vcId]);

    const chain = await did.getWorkflowVcChain(workflowId);
    expect(chain.workflow_id).toBe(workflowId);
    expect(chain.component_vcs).toHaveLength(2);
    expect(chain.component_vcs.map((vc: any) => vc.vc_id)).toEqual([vc1.vcId, vc2.vcId]);
    expect(chain.component_vcs.map((vc: any) => vc.execution_id)).toEqual([vc1.executionId, vc2.executionId]);
    expect(chain.workflow_vc.workflow_vc_id).toBe(workflowVc?.workflowVcId);
    expect(chain.workflow_vc.session_id).toBe(sessionId);
  });
});
