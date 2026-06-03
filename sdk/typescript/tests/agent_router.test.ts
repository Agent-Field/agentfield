import { describe, it, expect, vi } from 'vitest';
import { AgentRouter } from '../src/router/AgentRouter.js';

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('AgentRouter', () => {
  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------
  describe('constructor', () => {
    it('creates an instance with empty reasoners and skills', () => {
      const router = new AgentRouter();
      expect(router.reasoners).toEqual([]);
      expect(router.skills).toEqual([]);
    });

    it('stores prefix from options', () => {
      const router = new AgentRouter({ prefix: 'sim' });
      expect(router.prefix).toBe('sim');
    });

    it('stores tags from options', () => {
      const router = new AgentRouter({ tags: ['alpha', 'beta'] });
      expect(router.tags).toEqual(['alpha', 'beta']);
    });

    it('prefix is undefined when not provided', () => {
      const router = new AgentRouter();
      expect(router.prefix).toBeUndefined();
    });
  });

  // -------------------------------------------------------------------------
  // reasoner()
  // -------------------------------------------------------------------------
  describe('reasoner()', () => {
    it('registers a reasoner with the given name', () => {
      const router = new AgentRouter();
      const handler = vi.fn();

      router.reasoner('analyze', handler);

      expect(router.reasoners).toHaveLength(1);
      expect(router.reasoners[0]?.name).toBe('analyze');
    });

    it('stores the handler reference', () => {
      const router = new AgentRouter();
      const handler = vi.fn();

      router.reasoner('analyze', handler);

      expect(router.reasoners[0]?.handler).toBe(handler);
    });

    it('stores options when provided', () => {
      const router = new AgentRouter();
      const opts = { description: 'An analyzer' };

      router.reasoner('analyze', vi.fn(), opts);

      expect(router.reasoners[0]?.options).toEqual(opts);
    });

    it('prepends prefix to the name', () => {
      const router = new AgentRouter({ prefix: 'sim' });

      router.reasoner('run', vi.fn());

      expect(router.reasoners[0]?.name).toBe('sim_run');
    });

    it('sanitizes prefix with special characters', () => {
      const router = new AgentRouter({ prefix: 'my-module' });

      router.reasoner('action', vi.fn());

      expect(router.reasoners[0]?.name).toBe('my_module_action');
    });

    it('handles prefix with multiple consecutive special characters', () => {
      const router = new AgentRouter({ prefix: 'a--b' });

      router.reasoner('fn', vi.fn());

      expect(router.reasoners[0]?.name).toBe('a_b_fn');
    });

    it('does not modify name when prefix is absent', () => {
      const router = new AgentRouter();

      router.reasoner('plain_name', vi.fn());

      expect(router.reasoners[0]?.name).toBe('plain_name');
    });

    it('returns the router for chaining', () => {
      const router = new AgentRouter();
      const ret = router.reasoner('a', vi.fn());
      expect(ret).toBe(router);
    });

    it('registers multiple reasoners', () => {
      const router = new AgentRouter({ prefix: 'p' });

      router.reasoner('one', vi.fn()).reasoner('two', vi.fn()).reasoner('three', vi.fn());

      expect(router.reasoners.map((r) => r.name)).toEqual(['p_one', 'p_two', 'p_three']);
    });
  });

  // -------------------------------------------------------------------------
  // skill()
  // -------------------------------------------------------------------------
  describe('skill()', () => {
    it('registers a skill with the given name', () => {
      const router = new AgentRouter();
      const handler = vi.fn();

      router.skill('format', handler);

      expect(router.skills).toHaveLength(1);
      expect(router.skills[0]?.name).toBe('format');
    });

    it('stores the handler reference', () => {
      const router = new AgentRouter();
      const handler = vi.fn();

      router.skill('format', handler);

      expect(router.skills[0]?.handler).toBe(handler);
    });

    it('prepends prefix to the skill name', () => {
      const router = new AgentRouter({ prefix: 'utils' });

      router.skill('parse', vi.fn());

      expect(router.skills[0]?.name).toBe('utils_parse');
    });

    it('returns the router for chaining', () => {
      const router = new AgentRouter();
      const ret = router.skill('s', vi.fn());
      expect(ret).toBe(router);
    });

    it('registers multiple skills', () => {
      const router = new AgentRouter({ prefix: 'math' });

      router.skill('add', vi.fn()).skill('sub', vi.fn());

      expect(router.skills.map((s) => s.name)).toEqual(['math_add', 'math_sub']);
    });

    it('stores skill options when provided', () => {
      const router = new AgentRouter();
      const opts = { description: 'A formatter' };

      router.skill('fmt', vi.fn(), opts);

      expect(router.skills[0]?.options).toEqual(opts);
    });
  });

  // -------------------------------------------------------------------------
  // Mixed usage
  // -------------------------------------------------------------------------
  describe('mixed reasoner + skill registration', () => {
    it('maintains separate arrays for reasoners and skills', () => {
      const router = new AgentRouter({ prefix: 'ns' });

      router.reasoner('think', vi.fn()).skill('act', vi.fn());

      expect(router.reasoners).toHaveLength(1);
      expect(router.skills).toHaveLength(1);
      expect(router.reasoners[0]?.name).toBe('ns_think');
      expect(router.skills[0]?.name).toBe('ns_act');
    });
  });

  // -------------------------------------------------------------------------
  // Name sanitization edge cases
  // -------------------------------------------------------------------------
  describe('name sanitization', () => {
    it('strips leading/trailing underscores from prefix', () => {
      const router = new AgentRouter({ prefix: '_leading_' });

      router.reasoner('fn', vi.fn());

      // sanitize('_leading_') → 'leading', so result is 'leading_fn'
      expect(router.reasoners[0]?.name).toBe('leading_fn');
    });

    it('handles numeric prefix', () => {
      const router = new AgentRouter({ prefix: '42' });

      router.skill('go', vi.fn());

      expect(router.skills[0]?.name).toBe('42_go');
    });

    it('handles empty prefix gracefully (treated as no prefix)', () => {
      // empty string → sanitize('') → '' which is falsy → no prefix applied
      const router = new AgentRouter({ prefix: '' });

      router.reasoner('fn', vi.fn());

      expect(router.reasoners[0]?.name).toBe('fn');
    });
  });
});
