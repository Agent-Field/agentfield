import { describe, it, expect } from 'vitest';

import { matchesPattern } from '../src/utils/pattern.js';

describe('matchesPattern', () => {
  it('matches exact values only', () => {
    expect(matchesPattern('users', 'users')).toBe(true);
    expect(matchesPattern('users', 'user')).toBe(false);
  });

  it('supports trailing wildcards', () => {
    expect(matchesPattern('read_*', 'read_users')).toBe(true);
    expect(matchesPattern('read_*', 'write_users')).toBe(false);
  });

  it('supports leading wildcards', () => {
    expect(matchesPattern('*_users', 'read_users')).toBe(true);
    expect(matchesPattern('*_users', 'read_all')).toBe(false);
  });

  it('treats a standalone wildcard as universal', () => {
    expect(matchesPattern('*', 'anything')).toBe(true);
    expect(matchesPattern('*', '')).toBe(true);
  });

  it('escapes regex special characters', () => {
    expect(matchesPattern('a.b+c', 'a.b+c')).toBe(true);
    expect(matchesPattern('a.b+c', 'axb+c')).toBe(false);
  });
});
