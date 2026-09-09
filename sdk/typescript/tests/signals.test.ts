import { describe, expect, it, vi } from 'vitest';
import { DEFAULT_SHUTDOWN_TIMEOUT_MS, parseShutdownTimeout } from '../src/agent/signals.js';

describe('parseShutdownTimeout', () => {
  it.each([
    [undefined, DEFAULT_SHUTDOWN_TIMEOUT_MS],
    ['30', 30_000],
    ['30s', 30_000],
    ['5m', 300_000]
  ])('parses %s', (value, expected) => expect(parseShutdownTimeout(value)).toBe(expected));

  it('warns and uses the default for invalid values', () => {
    const warn = vi.fn();
    expect(parseShutdownTimeout('later', warn)).toBe(DEFAULT_SHUTDOWN_TIMEOUT_MS);
    expect(warn).toHaveBeenCalledOnce();
  });
});
