import { describe, it, expect, vi } from 'vitest';
import { SkillRegistry } from '../src/agent/SkillRegistry.js';
import { AgentRouter } from '../src/router/AgentRouter.js';
import type { SkillContext } from '../src/context/SkillContext.js';

describe('SkillRegistry', () => {
  // ---------------------------------------------------------------------------
  // register + get
  // ---------------------------------------------------------------------------

  describe('register', () => {
    it('stores a skill and retrieves it by name', () => {
      const registry = new SkillRegistry();
      const handler = vi.fn();
      registry.register('greet', handler);

      const skill = registry.get('greet');
      expect(skill).toBeDefined();
      expect(skill!.name).toBe('greet');
      expect(skill!.handler).toBe(handler);
    });

    it('stores options alongside the skill', () => {
      const registry = new SkillRegistry();
      const handler = vi.fn();
      registry.register('search', handler, {
        description: 'Search for things',
        tags: ['utility'],
      });

      const skill = registry.get('search');
      expect(skill!.options?.description).toBe('Search for things');
      expect(skill!.options?.tags).toEqual(['utility']);
    });

    it('returns undefined for an unregistered name', () => {
      const registry = new SkillRegistry();
      expect(registry.get('nonexistent')).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------------------
  // duplicate registration
  // ---------------------------------------------------------------------------

  describe('duplicate registration', () => {
    it('overwrites the previous skill when the same name is registered again', () => {
      const registry = new SkillRegistry();
      const firstHandler = vi.fn();
      const secondHandler = vi.fn();

      registry.register('compute', firstHandler);
      registry.register('compute', secondHandler);

      expect(registry.get('compute')!.handler).toBe(secondHandler);
    });
  });

  // ---------------------------------------------------------------------------
  // all
  // ---------------------------------------------------------------------------

  describe('all', () => {
    it('returns an empty array when no skills are registered', () => {
      const registry = new SkillRegistry();
      expect(registry.all()).toEqual([]);
    });

    it('returns all registered skills', () => {
      const registry = new SkillRegistry();
      registry.register('skillA', vi.fn());
      registry.register('skillB', vi.fn());
      registry.register('skillC', vi.fn());

      const names = registry.all().map((s) => s.name).sort();
      expect(names).toEqual(['skillA', 'skillB', 'skillC']);
    });

    it('returns the correct count after overwrite', () => {
      const registry = new SkillRegistry();
      registry.register('alpha', vi.fn());
      registry.register('alpha', vi.fn()); // overwrite — still 1 unique name
      registry.register('beta', vi.fn());

      expect(registry.all()).toHaveLength(2);
    });
  });

  // ---------------------------------------------------------------------------
  // includeRouter
  // ---------------------------------------------------------------------------

  describe('includeRouter', () => {
    it('imports skills from an AgentRouter', () => {
      const registry = new SkillRegistry();
      const router = new AgentRouter();
      router.skill('ping', vi.fn());
      router.skill('pong', vi.fn());

      registry.includeRouter(router);

      expect(registry.get('ping')).toBeDefined();
      expect(registry.get('pong')).toBeDefined();
      expect(registry.all()).toHaveLength(2);
    });

    it('imports router skills with prefix applied by the router', () => {
      const registry = new SkillRegistry();
      const router = new AgentRouter({ prefix: 'utils' });
      router.skill('format', vi.fn());

      registry.includeRouter(router);

      // AgentRouter applies prefix as "utils_format"
      expect(registry.get('utils_format')).toBeDefined();
    });

    it('merges router skills with existing registry skills', () => {
      const registry = new SkillRegistry();
      registry.register('existing', vi.fn());

      const router = new AgentRouter();
      router.skill('new', vi.fn());
      registry.includeRouter(router);

      expect(registry.all()).toHaveLength(2);
      expect(registry.get('existing')).toBeDefined();
      expect(registry.get('new')).toBeDefined();
    });

    it('router skill overwrites existing registry skill with the same name', () => {
      const registry = new SkillRegistry();
      const originalHandler = vi.fn();
      const routerHandler = vi.fn();

      registry.register('clash', originalHandler);

      const router = new AgentRouter();
      router.skill('clash', routerHandler);
      registry.includeRouter(router);

      expect(registry.get('clash')!.handler).toBe(routerHandler);
    });
  });
});
