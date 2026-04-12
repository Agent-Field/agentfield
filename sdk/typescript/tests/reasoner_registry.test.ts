import { describe, it, expect, vi } from 'vitest';
import { ReasonerRegistry } from '../src/agent/ReasonerRegistry.js';
import { AgentRouter } from '../src/router/AgentRouter.js';

describe('ReasonerRegistry', () => {
  // ---------------------------------------------------------------------------
  // register + get
  // ---------------------------------------------------------------------------

  describe('register', () => {
    it('stores a reasoner and retrieves it by name', () => {
      const registry = new ReasonerRegistry();
      const handler = vi.fn();
      registry.register('analyze', handler);

      const reasoner = registry.get('analyze');
      expect(reasoner).toBeDefined();
      expect(reasoner!.name).toBe('analyze');
      expect(reasoner!.handler).toBe(handler);
    });

    it('stores options alongside the reasoner', () => {
      const registry = new ReasonerRegistry();
      const handler = vi.fn();
      registry.register('classify', handler, {
        description: 'Classify items',
        tags: ['ml'],
        trackWorkflow: true,
      });

      const reasoner = registry.get('classify');
      expect(reasoner!.options?.description).toBe('Classify items');
      expect(reasoner!.options?.tags).toEqual(['ml']);
      expect(reasoner!.options?.trackWorkflow).toBe(true);
    });

    it('returns undefined for an unregistered name', () => {
      const registry = new ReasonerRegistry();
      expect(registry.get('missing')).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------------------
  // duplicate registration
  // ---------------------------------------------------------------------------

  describe('duplicate registration', () => {
    it('overwrites the previous reasoner when the same name is registered again', () => {
      const registry = new ReasonerRegistry();
      const firstHandler = vi.fn();
      const secondHandler = vi.fn();

      registry.register('infer', firstHandler);
      registry.register('infer', secondHandler);

      expect(registry.get('infer')!.handler).toBe(secondHandler);
    });

    it('count stays the same after overwrite', () => {
      const registry = new ReasonerRegistry();
      registry.register('dupe', vi.fn());
      registry.register('dupe', vi.fn());

      expect(registry.all()).toHaveLength(1);
    });
  });

  // ---------------------------------------------------------------------------
  // all
  // ---------------------------------------------------------------------------

  describe('all', () => {
    it('returns an empty array when no reasoners are registered', () => {
      const registry = new ReasonerRegistry();
      expect(registry.all()).toEqual([]);
    });

    it('returns all registered reasoners', () => {
      const registry = new ReasonerRegistry();
      registry.register('reasonerA', vi.fn());
      registry.register('reasonerB', vi.fn());
      registry.register('reasonerC', vi.fn());

      const names = registry.all().map((r) => r.name).sort();
      expect(names).toEqual(['reasonerA', 'reasonerB', 'reasonerC']);
    });

    it('each returned entry has name and handler', () => {
      const registry = new ReasonerRegistry();
      const h = vi.fn();
      registry.register('check', h);

      const [entry] = registry.all();
      expect(entry.name).toBe('check');
      expect(entry.handler).toBe(h);
    });
  });

  // ---------------------------------------------------------------------------
  // includeRouter
  // ---------------------------------------------------------------------------

  describe('includeRouter', () => {
    it('imports reasoners from an AgentRouter', () => {
      const registry = new ReasonerRegistry();
      const router = new AgentRouter();
      router.reasoner('think', vi.fn());
      router.reasoner('decide', vi.fn());

      registry.includeRouter(router);

      expect(registry.get('think')).toBeDefined();
      expect(registry.get('decide')).toBeDefined();
      expect(registry.all()).toHaveLength(2);
    });

    it('applies the router prefix to imported reasoners', () => {
      const registry = new ReasonerRegistry();
      const router = new AgentRouter({ prefix: 'planner' });
      router.reasoner('draft', vi.fn());

      registry.includeRouter(router);

      // AgentRouter builds "planner_draft"
      expect(registry.get('planner_draft')).toBeDefined();
    });

    it('merges router reasoners with existing registry entries', () => {
      const registry = new ReasonerRegistry();
      registry.register('existing', vi.fn());

      const router = new AgentRouter();
      router.reasoner('fresh', vi.fn());
      registry.includeRouter(router);

      expect(registry.all()).toHaveLength(2);
      expect(registry.get('existing')).toBeDefined();
      expect(registry.get('fresh')).toBeDefined();
    });

    it('router reasoner overwrites existing registry entry with the same name', () => {
      const registry = new ReasonerRegistry();
      const oldHandler = vi.fn();
      const newHandler = vi.fn();

      registry.register('conflict', oldHandler);

      const router = new AgentRouter();
      router.reasoner('conflict', newHandler);
      registry.includeRouter(router);

      expect(registry.get('conflict')!.handler).toBe(newHandler);
    });

    it('does not import skills from the router (only reasoners)', () => {
      const registry = new ReasonerRegistry();
      const router = new AgentRouter();
      router.skill('a-skill', vi.fn());
      router.reasoner('a-reasoner', vi.fn());

      registry.includeRouter(router);

      expect(registry.get('a-reasoner')).toBeDefined();
      // 'a-skill' should NOT land in the reasoner registry
      expect(registry.get('a-skill')).toBeUndefined();
      expect(registry.all()).toHaveLength(1);
    });
  });
});
