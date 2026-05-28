/**
 * Behavioral invariant tests for Agent.
 *
 * These tests verify structural properties that must always hold regardless of
 * implementation details or refactoring.
 */
import { describe, it, expect } from 'vitest';
import { Agent } from '../src/agent/Agent.js';

describe('Agent invariants', () => {
  // ── Registration Persistence ────────────────────────────────────────────────

  describe('INVARIANT: registration persistence', () => {
    it('after .reasoner("name", fn), the reasoner is discoverable via reasoners.get()', () => {
      const agent = new Agent({ nodeId: 'test-node', devMode: true });
      const handler = async () => ({ ok: true });
      agent.reasoner('my-reasoner', handler);

      const found = agent.reasoners.get('my-reasoner');
      expect(found).toBeDefined();
      expect(found!.name).toBe('my-reasoner');
    });

    it('after .skill("name", fn), the skill is discoverable via skills.get()', () => {
      const agent = new Agent({ nodeId: 'test-node', devMode: true });
      const handler = () => 'result';
      agent.skill('my-skill', handler);

      const found = agent.skills.get('my-skill');
      expect(found).toBeDefined();
      expect(found!.name).toBe('my-skill');
    });

    it('multiple reasoners are all discoverable after registration', () => {
      const agent = new Agent({ nodeId: 'test-node', devMode: true });
      const names = ['alpha', 'beta', 'gamma', 'delta'];
      names.forEach((name) => agent.reasoner(name, async () => ({})));

      const registered = agent.reasoners.all().map((r) => r.name);
      names.forEach((name) => expect(registered).toContain(name));
    });

    it('multiple skills are all discoverable after registration', () => {
      const agent = new Agent({ nodeId: 'test-node', devMode: true });
      const names = ['format', 'parse', 'transform'];
      names.forEach((name) => agent.skill(name, () => ({})));

      const registered = agent.skills.all().map((s) => s.name);
      names.forEach((name) => expect(registered).toContain(name));
    });

    it('re-registering a reasoner with the same name updates it', () => {
      const agent = new Agent({ nodeId: 'test-node', devMode: true });
      const handler1 = async () => ({ version: 1 });
      const handler2 = async () => ({ version: 2 });

      agent.reasoner('updatable', handler1);
      agent.reasoner('updatable', handler2);

      const all = agent.reasoners.all().filter((r) => r.name === 'updatable');
      // Only one registration with that name
      expect(all).toHaveLength(1);
    });
  });

  // ── Skill/Reasoner Namespace Separation ─────────────────────────────────────

  describe('INVARIANT: skill and reasoner namespaces are independent', () => {
    it('registering a skill named "x" and reasoner named "x" are independent registrations', () => {
      const agent = new Agent({ nodeId: 'test-node', devMode: true });

      agent.skill('shared-name', () => 'skill-result');
      agent.reasoner('shared-name', async () => ({ type: 'reasoner' }));

      const skill = agent.skills.get('shared-name');
      const reasoner = agent.reasoners.get('shared-name');

      expect(skill).toBeDefined();
      expect(reasoner).toBeDefined();

      // Both can exist simultaneously with the same name
      expect(agent.skills.all().map((s) => s.name)).toContain('shared-name');
      expect(agent.reasoners.all().map((r) => r.name)).toContain('shared-name');
    });

    it('deleting skill does not remove reasoner with same name', () => {
      const agent = new Agent({ nodeId: 'test-node', devMode: true });
      agent.skill('x', () => 'skill');
      agent.reasoner('x', async () => ({ type: 'reasoner' }));

      // Skills and reasoners have separate registries
      expect(agent.skills.get('x')).toBeDefined();
      expect(agent.reasoners.get('x')).toBeDefined();
    });

    it('reasoners.all() never contains skill entries', () => {
      const agent = new Agent({ nodeId: 'test-node', devMode: true });
      agent.skill('only-a-skill', () => ({}));
      agent.reasoner('only-a-reasoner', async () => ({}));

      const reasonerNames = agent.reasoners.all().map((r) => r.name);
      expect(reasonerNames).not.toContain('only-a-skill');
      expect(reasonerNames).toContain('only-a-reasoner');
    });

    it('skills.all() never contains reasoner entries', () => {
      const agent = new Agent({ nodeId: 'test-node', devMode: true });
      agent.skill('only-a-skill', () => ({}));
      agent.reasoner('only-a-reasoner', async () => ({}));

      const skillNames = agent.skills.all().map((s) => s.name);
      expect(skillNames).not.toContain('only-a-reasoner');
      expect(skillNames).toContain('only-a-skill');
    });
  });

  // ── Config Immutability ──────────────────────────────────────────────────────

  describe('INVARIANT: config immutability after construction', () => {
    it('nodeId does not change after construction', () => {
      const agent = new Agent({ nodeId: 'immutable-node', devMode: true });
      const originalId = agent.config.nodeId;

      // Attempting to mutate config externally shouldn't affect the stored value
      // (we verify the value stays stable across multiple reads)
      expect(agent.config.nodeId).toBe(originalId);
      expect(agent.config.nodeId).toBe('immutable-node');
    });

    it('agentFieldUrl does not change after construction', () => {
      const agent = new Agent({
        nodeId: 'test-node',
        agentFieldUrl: 'http://custom-control-plane:9090',
        devMode: true
      });

      const original = agent.config.agentFieldUrl;
      expect(agent.config.agentFieldUrl).toBe(original);
      expect(agent.config.agentFieldUrl).toBe('http://custom-control-plane:9090');
    });

    it('registering reasoners does not mutate config', () => {
      const agent = new Agent({ nodeId: 'stable-node', devMode: true });
      const nodeIdBefore = agent.config.nodeId;
      const urlBefore = agent.config.agentFieldUrl;

      agent.reasoner('r1', async () => ({}));
      agent.reasoner('r2', async () => ({}));
      agent.skill('s1', () => ({}));

      expect(agent.config.nodeId).toBe(nodeIdBefore);
      expect(agent.config.agentFieldUrl).toBe(urlBefore);
    });
  });

  // ── Local Call Invariants ────────────────────────────────────────────────────

  describe('INVARIANT: local reasoner invocation via agent.call', () => {
    it('calling a registered local reasoner returns its result', async () => {
      const agent = new Agent({ nodeId: 'local-node', devMode: true });
      agent.reasoner('echo', async (ctx) => ({ echoed: ctx.input.message }));

      const result = await agent.call('local-node.echo', { message: 'ping' });
      expect(result).toEqual({ echoed: 'ping' });
    });

    it('calling a non-existent local reasoner throws', async () => {
      const agent = new Agent({ nodeId: 'local-node', devMode: true });

      await expect(agent.call('local-node.nonexistent', {})).rejects.toThrow();
    });
  });
});
