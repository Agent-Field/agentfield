import { describe, it, expect } from 'vitest';
import {
  ExecutionStatus,
  CANONICAL_STATUSES,
  TERMINAL_STATUSES,
  ACTIVE_STATUSES,
  normalizeStatus,
  isTerminal,
  isActive,
  type ExecutionStatusValue
} from '../src/status/ExecutionStatus.js';

describe('ExecutionStatus constants', () => {
  it('exposes all canonical status values', () => {
    expect(ExecutionStatus.PENDING).toBe('pending');
    expect(ExecutionStatus.QUEUED).toBe('queued');
    expect(ExecutionStatus.WAITING).toBe('waiting');
    expect(ExecutionStatus.RUNNING).toBe('running');
    expect(ExecutionStatus.SUCCEEDED).toBe('succeeded');
    expect(ExecutionStatus.FAILED).toBe('failed');
    expect(ExecutionStatus.CANCELLED).toBe('cancelled');
    expect(ExecutionStatus.TIMEOUT).toBe('timeout');
    expect(ExecutionStatus.UNKNOWN).toBe('unknown');
  });

  it('CANONICAL_STATUSES contains every ExecutionStatus value', () => {
    for (const value of Object.values(ExecutionStatus)) {
      expect(CANONICAL_STATUSES.has(value)).toBe(true);
    }
    expect(CANONICAL_STATUSES.size).toBe(Object.keys(ExecutionStatus).length);
  });
});

describe('TERMINAL_STATUSES', () => {
  it('includes succeeded, failed, cancelled, timeout', () => {
    expect(TERMINAL_STATUSES.has('succeeded')).toBe(true);
    expect(TERMINAL_STATUSES.has('failed')).toBe(true);
    expect(TERMINAL_STATUSES.has('cancelled')).toBe(true);
    expect(TERMINAL_STATUSES.has('timeout')).toBe(true);
  });

  it('does not include active statuses', () => {
    expect(TERMINAL_STATUSES.has('pending')).toBe(false);
    expect(TERMINAL_STATUSES.has('running')).toBe(false);
    expect(TERMINAL_STATUSES.has('queued')).toBe(false);
    expect(TERMINAL_STATUSES.has('waiting')).toBe(false);
  });
});

describe('ACTIVE_STATUSES', () => {
  it('includes pending, queued, waiting, running', () => {
    expect(ACTIVE_STATUSES.has('pending')).toBe(true);
    expect(ACTIVE_STATUSES.has('queued')).toBe(true);
    expect(ACTIVE_STATUSES.has('waiting')).toBe(true);
    expect(ACTIVE_STATUSES.has('running')).toBe(true);
  });

  it('does not include terminal statuses', () => {
    expect(ACTIVE_STATUSES.has('succeeded')).toBe(false);
    expect(ACTIVE_STATUSES.has('failed')).toBe(false);
    expect(ACTIVE_STATUSES.has('cancelled')).toBe(false);
    expect(ACTIVE_STATUSES.has('timeout')).toBe(false);
  });
});

describe('normalizeStatus()', () => {
  it('returns canonical statuses unchanged', () => {
    for (const value of Object.values(ExecutionStatus)) {
      expect(normalizeStatus(value)).toBe(value);
    }
  });

  it('lowercases and trims input', () => {
    expect(normalizeStatus('  RUNNING  ')).toBe('running');
    expect(normalizeStatus('SUCCEEDED')).toBe('succeeded');
    expect(normalizeStatus('  Failed ')).toBe('failed');
  });

  it('resolves common aliases to canonical statuses', () => {
    const cases: [string, string][] = [
      ['success', 'succeeded'],
      ['successful', 'succeeded'],
      ['completed', 'succeeded'],
      ['complete', 'succeeded'],
      ['done', 'succeeded'],
      ['ok', 'succeeded'],
      ['error', 'failed'],
      ['failure', 'failed'],
      ['errored', 'failed'],
      ['canceled', 'cancelled'],
      ['cancel', 'cancelled'],
      ['timed_out', 'timeout'],
      ['wait', 'queued'],
      ['awaiting_approval', 'waiting'],
      ['awaiting_human', 'waiting'],
      ['approval_pending', 'waiting'],
      ['in_progress', 'running'],
      ['processing', 'running'],
    ];

    for (const [alias, expected] of cases) {
      expect(normalizeStatus(alias)).toBe(expected);
    }
  });

  it('returns "unknown" for null', () => {
    expect(normalizeStatus(null)).toBe('unknown');
  });

  it('returns "unknown" for undefined', () => {
    expect(normalizeStatus(undefined)).toBe('unknown');
  });

  it('returns "unknown" for empty string', () => {
    expect(normalizeStatus('')).toBe('unknown');
  });

  it('returns "unknown" for whitespace-only string', () => {
    expect(normalizeStatus('   ')).toBe('unknown');
  });

  it('returns "unknown" for unrecognized values', () => {
    expect(normalizeStatus('banana')).toBe('unknown');
    expect(normalizeStatus('123')).toBe('unknown');
  });
});

describe('isTerminal()', () => {
  it('returns true for terminal statuses', () => {
    expect(isTerminal('succeeded')).toBe(true);
    expect(isTerminal('failed')).toBe(true);
    expect(isTerminal('cancelled')).toBe(true);
    expect(isTerminal('timeout')).toBe(true);
  });

  it('returns true for terminal aliases', () => {
    expect(isTerminal('success')).toBe(true);
    expect(isTerminal('error')).toBe(true);
    expect(isTerminal('canceled')).toBe(true);
    expect(isTerminal('timed_out')).toBe(true);
  });

  it('returns false for active statuses', () => {
    expect(isTerminal('pending')).toBe(false);
    expect(isTerminal('running')).toBe(false);
    expect(isTerminal('queued')).toBe(false);
    expect(isTerminal('waiting')).toBe(false);
  });

  it('returns false for null/undefined/unknown', () => {
    expect(isTerminal(null)).toBe(false);
    expect(isTerminal(undefined)).toBe(false);
    expect(isTerminal('unknown')).toBe(false);
    expect(isTerminal('garbage')).toBe(false);
  });
});

describe('isActive()', () => {
  it('returns true for active statuses', () => {
    expect(isActive('pending')).toBe(true);
    expect(isActive('queued')).toBe(true);
    expect(isActive('waiting')).toBe(true);
    expect(isActive('running')).toBe(true);
  });

  it('returns true for active aliases', () => {
    expect(isActive('in_progress')).toBe(true);
    expect(isActive('processing')).toBe(true);
    expect(isActive('awaiting_approval')).toBe(true);
    expect(isActive('wait')).toBe(true);
  });

  it('returns false for terminal statuses', () => {
    expect(isActive('succeeded')).toBe(false);
    expect(isActive('failed')).toBe(false);
    expect(isActive('cancelled')).toBe(false);
    expect(isActive('timeout')).toBe(false);
  });

  it('returns false for null/undefined/unknown', () => {
    expect(isActive(null)).toBe(false);
    expect(isActive(undefined)).toBe(false);
    expect(isActive('unknown')).toBe(false);
  });
});
