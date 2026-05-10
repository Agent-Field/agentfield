import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { ApprovalClient } from '../src/approval/ApprovalClient.js';

/**
 * Minimal HTTP server that returns canned JSON responses for approval endpoints.
 */
function createMockServer(
  responses: Array<{ status: number; body: unknown }>,
  onRequest?: (req: http.IncomingMessage, body: string) => void
) {
  let callIndex = 0;
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
    req.on('end', () => {
      onRequest?.(req, Buffer.concat(chunks).toString('utf8'));
      const resp = responses[Math.min(callIndex++, responses.length - 1)];
      res.writeHead(resp.status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(resp.body));
    });
  });
  return server;
}

function serverURL(server: http.Server): string {
  const addr = server.address() as AddressInfo;
  return `http://127.0.0.1:${addr.port}`;
}

describe('ApprovalClient', () => {
  let server: http.Server;

  afterEach(() => {
    return new Promise<void>((resolve) => {
      if (server?.listening) {
        server.closeAllConnections();
        server.close(() => resolve());
      } else {
        resolve();
      }
    });
  });

  // -------------------------------------------------------------------
  // requestApproval
  // -------------------------------------------------------------------

  describe('requestApproval', () => {
    it('posts the control-plane approval tracking contract and returns typed response', async () => {
      let receivedBody: Record<string, unknown> | undefined;

      server = createMockServer([
        {
          status: 200,
          body: {
            approval_request_id: 'req-abc',
            approval_request_url: 'https://hub.example.com/r/req-abc',
          },
        },
      ], (_req, body) => {
        receivedBody = JSON.parse(body) as Record<string, unknown>;
      });
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
        apiKey: 'key-1',
      });

      const result = await client.requestApproval('exec-1', {
        approvalRequestId: 'req-abc',
        approvalRequestUrl: 'https://hub.example.com/r/req-abc',
        callbackUrl: 'https://agent.example.com/webhooks/approval',
        expiresInHours: 24,
      });

      expect(receivedBody).toEqual({
        approval_request_id: 'req-abc',
        approval_request_url: 'https://hub.example.com/r/req-abc',
        callback_url: 'https://agent.example.com/webhooks/approval',
        expires_in_hours: 24,
      });
      expect(receivedBody).not.toHaveProperty('title');
      expect(receivedBody).not.toHaveProperty('project_id');
      expect(receivedBody).not.toHaveProperty('template_type');
      expect(result.approvalRequestId).toBe('req-abc');
      expect(result.approvalRequestUrl).toBe('https://hub.example.com/r/req-abc');
    });

    it('throws on HTTP error', async () => {
      server = createMockServer([
        { status: 404, body: { error: 'not found' } },
      ]);
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      await expect(
        client.requestApproval('exec-1', { approvalRequestId: 'req-fail' })
      ).rejects.toThrow();
    });
  });

  // -------------------------------------------------------------------
  // getApprovalStatus
  // -------------------------------------------------------------------

  describe('getApprovalStatus', () => {
    it('returns typed response for approved status', async () => {
      server = createMockServer([
        {
          status: 200,
          body: {
            status: 'approved',
            response: { decision: 'approved', feedback: 'LGTM' },
            request_url: 'https://hub.example.com/r/req-abc',
            requested_at: '2026-02-25T10:00:00Z',
            responded_at: '2026-02-25T11:00:00Z',
          },
        },
      ]);
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      const result = await client.getApprovalStatus('exec-1');

      expect(result.status).toBe('approved');
      expect(result.response).toEqual({ decision: 'approved', feedback: 'LGTM' });
      expect(result.requestUrl).toBe('https://hub.example.com/r/req-abc');
      expect(result.requestedAt).toBe('2026-02-25T10:00:00Z');
      expect(result.respondedAt).toBe('2026-02-25T11:00:00Z');
    });

    it('returns pending with no response fields', async () => {
      server = createMockServer([
        {
          status: 200,
          body: {
            status: 'pending',
            request_url: 'https://hub.example.com/r/req-abc',
            requested_at: '2026-02-25T10:00:00Z',
          },
        },
      ]);
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      const result = await client.getApprovalStatus('exec-1');

      expect(result.status).toBe('pending');
      expect(result.response).toBeUndefined();
      expect(result.respondedAt).toBeUndefined();
    });

    it('throws on server error', async () => {
      server = createMockServer([
        { status: 500, body: { error: 'internal' } },
      ]);
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      await expect(client.getApprovalStatus('exec-1')).rejects.toThrow();
    });
  });

  // -------------------------------------------------------------------
  // waitForApproval
  // -------------------------------------------------------------------

  describe('waitForApproval', () => {
    it('resolves once status is no longer pending', async () => {
      server = createMockServer([
        { status: 200, body: { status: 'pending' } },
        { status: 200, body: { status: 'approved', response: { decision: 'approved' } } },
      ]);
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      const result = await client.waitForApproval('exec-1', {
        pollIntervalMs: 50,
        maxIntervalMs: 50,
      });

      expect(result.status).toBe('approved');
    });

    it('returns rejected status', async () => {
      server = createMockServer([
        { status: 200, body: { status: 'rejected', response: { feedback: 'needs work' } } },
      ]);
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      const result = await client.waitForApproval('exec-1', {
        pollIntervalMs: 50,
      });

      expect(result.status).toBe('rejected');
    });

    it('throws on timeout', async () => {
      server = createMockServer(
        Array.from({ length: 20 }, () => ({
          status: 200,
          body: { status: 'pending' },
        }))
      );
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      await expect(
        client.waitForApproval('exec-1', {
          pollIntervalMs: 20,
          maxIntervalMs: 20,
          timeoutMs: 100,
        })
      ).rejects.toThrow(/timed out/);
    });

    it('retries on transient errors and eventually resolves', async () => {
      server = createMockServer([
        { status: 500, body: { error: 'transient' } },
        { status: 200, body: { status: 'approved' } },
      ]);
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      const result = await client.waitForApproval('exec-1', {
        pollIntervalMs: 50,
        maxIntervalMs: 50,
      });

      expect(result.status).toBe('approved');
    });

    it('resolves on expired status', async () => {
      server = createMockServer([
        {
          status: 200,
          body: {
            status: 'expired',
            request_url: 'https://hub.example.com/r/req-abc',
            requested_at: '2026-02-25T10:00:00Z',
            responded_at: '2026-02-28T10:00:00Z',
          },
        },
      ]);
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      const result = await client.waitForApproval('exec-1', {
        pollIntervalMs: 50,
      });

      expect(result.status).toBe('expired');
    });
  });

  describe('getApprovalStatus — expired', () => {
    it('returns expired status', async () => {
      server = createMockServer([
        {
          status: 200,
          body: {
            status: 'expired',
            request_url: 'https://hub.example.com/r/req-abc',
            requested_at: '2026-02-25T10:00:00Z',
            responded_at: '2026-02-28T10:00:00Z',
          },
        },
      ]);
      await new Promise<void>((resolve) => server.listen(0, resolve));

      const client = new ApprovalClient({
        baseURL: serverURL(server),
        nodeId: 'test-node',
      });

      const result = await client.getApprovalStatus('exec-1');

      expect(result.status).toBe('expired');
      expect(result.respondedAt).toBe('2026-02-28T10:00:00Z');
    });
  });
});
