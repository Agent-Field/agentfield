import { describe, expect, it, vi } from 'vitest';

import { DidManager } from '../src/did/DidManager.js';

const identityPackage = {
  agentDid: { did: 'did:agent:123' },
  agentfieldServerId: 'server-1',
  reasonerDids: {
    summarize: { did: 'did:reasoner:summarize' },
    classify: { did: 'did:reasoner:classify' }
  },
  skillDids: {
    translate: { did: 'did:skill:translate' }
  }
};

describe('DidManager', () => {
  it('stores a successful registration and resolves agent, reasoner, and skill DIDs', async () => {
    const client = {
      registerAgent: vi.fn().mockResolvedValue({ success: true, identityPackage })
    };
    const manager = new DidManager(client as any, 'agent-node-1');

    expect(manager.enabled).toBe(false);
    expect(manager.getAgentDid()).toBeUndefined();
    expect(manager.getFunctionDid('missing')).toBeUndefined();
    expect(manager.getIdentitySummary()).toEqual({
      enabled: false,
      message: 'No identity package available'
    });

    await expect(
      manager.registerAgent([{ id: 'summarize' }], [{ id: 'translate' }])
    ).resolves.toBe(true);

    expect(client.registerAgent).toHaveBeenCalledWith({
      agentNodeId: 'agent-node-1',
      reasoners: [{ id: 'summarize' }],
      skills: [{ id: 'translate' }]
    });
    expect(manager.enabled).toBe(true);
    expect(manager.getAgentDid()).toBe('did:agent:123');
    expect(manager.getFunctionDid('summarize')).toBe('did:reasoner:summarize');
    expect(manager.getFunctionDid('translate')).toBe('did:skill:translate');
    expect(manager.getFunctionDid('missing')).toBe('did:agent:123');
    expect(manager.getIdentityPackage()).toEqual(identityPackage);
    expect(manager.getIdentitySummary()).toEqual({
      enabled: true,
      agentDid: 'did:agent:123',
      agentfieldServerId: 'server-1',
      reasonerCount: 2,
      skillCount: 1,
      reasonerDids: {
        summarize: 'did:reasoner:summarize',
        classify: 'did:reasoner:classify'
      },
      skillDids: {
        translate: 'did:skill:translate'
      }
    });
  });

  it('returns false and keeps DID support disabled when registration fails', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const client = {
      registerAgent: vi.fn().mockResolvedValue({ success: false, error: 'control plane unavailable' })
    };
    const manager = new DidManager(client as any, 'agent-node-2');

    await expect(manager.registerAgent([], [])).resolves.toBe(false);

    expect(manager.enabled).toBe(false);
    expect(manager.getIdentityPackage()).toBeUndefined();
    expect(warn).toHaveBeenCalledWith(
      '[DID] Registration failed: control plane unavailable'
    );

    warn.mockRestore();
  });
});
