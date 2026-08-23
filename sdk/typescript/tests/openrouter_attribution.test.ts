import { afterEach, describe, expect, it } from 'vitest';

import {
  openRouterAttributionEnv,
  openRouterAttributionHeaders,
} from '../src/ai/openrouterAttribution.js';

const CATEGORY_ENV_KEYS = [
  'AGENTFIELD_OPENROUTER_CATEGORIES',
  'OR_CATEGORIES',
] as const;

const originalCategoryEnv = Object.fromEntries(
  CATEGORY_ENV_KEYS.map((key) => [key, process.env[key]])
);

afterEach(() => {
  for (const key of CATEGORY_ENV_KEYS) {
    const previous = originalCategoryEnv[key];
    if (previous === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = previous;
    }
  }
});

describe('openRouterAttributionHeaders', () => {
  it('includes X-OpenRouter-Categories by default', () => {
    delete process.env.AGENTFIELD_OPENROUTER_CATEGORIES;
    delete process.env.OR_CATEGORIES;

    expect(openRouterAttributionHeaders({ env: {} })).toEqual({
      'HTTP-Referer': 'https://agentfield.ai',
      'X-OpenRouter-Title': 'AgentField AI',
      'X-Title': 'AgentField AI',
      'X-OpenRouter-Categories': 'cli-agent,programming-app',
    });
  });

  it('resolves categories from AGENTFIELD_OPENROUTER_CATEGORIES then OR_CATEGORIES', () => {
    expect(
      openRouterAttributionHeaders({
        env: { AGENTFIELD_OPENROUTER_CATEGORIES: 'research,translation' },
      })['X-OpenRouter-Categories']
    ).toBe('research,translation');

    expect(
      openRouterAttributionHeaders({
        env: { OR_CATEGORIES: 'roleplay' },
      })['X-OpenRouter-Categories']
    ).toBe('roleplay');
  });

  it('omits attribution headers when disabled', () => {
    expect(
      openRouterAttributionHeaders({
        env: { AGENTFIELD_OPENROUTER_ATTRIBUTION: 'false' },
      })
    ).toEqual({});
  });
});

describe('openRouterAttributionEnv', () => {
  it('injects category env defaults for subprocesses', () => {
    expect(openRouterAttributionEnv({})).toMatchObject({
      AGENTFIELD_OPENROUTER_CATEGORIES: 'cli-agent,programming-app',
      OR_CATEGORIES: 'cli-agent,programming-app',
    });
  });
});
