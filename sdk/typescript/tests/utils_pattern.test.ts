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

  it('handles empty pattern', () => {
    expect(matchesPattern('', '')).toBe(true);
    expect(matchesPattern('', 'a')).toBe(false);
  });

  it('handles empty value', () => {
    expect(matchesPattern('*', '')).toBe(true);
    expect(matchesPattern('test', '')).toBe(false);
  });

  it('handles multiple wildcards', () => {
    expect(matchesPattern('a*b*c', 'axbyc')).toBe(true);
    expect(matchesPattern('a*b*c', 'abc')).toBe(true);
    expect(matchesPattern('a*b*c', 'axbyz')).toBe(false);
  });

  it('handles dots and plus signs in patterns', () => {
    expect(matchesPattern('file.txt', 'file.txt')).toBe(true);
    expect(matchesPattern('file.txt', 'fileXtxt')).toBe(false);
    expect(matchesPattern('v1.0.0', 'v1.0.0')).toBe(true);
  });

  it('handles wildcard prefix with suffix', () => {
    expect(matchesPattern('*.example.com', 'api.example.com')).toBe(true);
    expect(matchesPattern('*.example.com', 'example.com')).toBe(false);
    expect(matchesPattern('*.example.com', 'api.test.example.com')).toBe(true);
  });
});
