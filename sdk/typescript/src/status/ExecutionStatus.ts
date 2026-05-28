/**
 * Canonical execution status utilities for the AgentField TypeScript SDK.
 *
 * Mirrors the control plane's status normalization and terminal-state logic
 * so that SDK consumers can classify execution statuses consistently.
 */

/** Canonical execution status values used by the control plane. */
export const ExecutionStatus = {
  PENDING: 'pending',
  QUEUED: 'queued',
  WAITING: 'waiting',
  RUNNING: 'running',
  SUCCEEDED: 'succeeded',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
  TIMEOUT: 'timeout',
  UNKNOWN: 'unknown',
} as const;

export type ExecutionStatusValue = (typeof ExecutionStatus)[keyof typeof ExecutionStatus];

/** All canonical status strings. */
export const CANONICAL_STATUSES: ReadonlySet<string> = new Set<string>(
  Object.values(ExecutionStatus)
);

/** Statuses that represent a completed execution (no further transitions). */
export const TERMINAL_STATUSES: ReadonlySet<string> = new Set<string>([
  ExecutionStatus.SUCCEEDED,
  ExecutionStatus.FAILED,
  ExecutionStatus.CANCELLED,
  ExecutionStatus.TIMEOUT,
]);

/** Statuses that represent an active, pollable execution. */
export const ACTIVE_STATUSES: ReadonlySet<string> = new Set<string>([
  ExecutionStatus.PENDING,
  ExecutionStatus.QUEUED,
  ExecutionStatus.WAITING,
  ExecutionStatus.RUNNING,
]);

/** Human-friendly aliases that map to canonical statuses. */
const STATUS_ALIASES: Record<string, string> = {
  success: 'succeeded',
  successful: 'succeeded',
  completed: 'succeeded',
  complete: 'succeeded',
  done: 'succeeded',
  ok: 'succeeded',
  error: 'failed',
  failure: 'failed',
  errored: 'failed',
  canceled: 'cancelled',
  cancel: 'cancelled',
  timed_out: 'timeout',
  wait: 'queued',
  awaiting_approval: 'waiting',
  awaiting_human: 'waiting',
  approval_pending: 'waiting',
  in_progress: 'running',
  processing: 'running',
};

/**
 * Normalize an arbitrary status string to its canonical form.
 *
 * Returns `"unknown"` for unrecognized or empty values.
 */
export function normalizeStatus(status: string | null | undefined): string {
  if (status == null) return 'unknown';
  const normalized = status.trim().toLowerCase();
  if (!normalized) return 'unknown';
  if (CANONICAL_STATUSES.has(normalized)) return normalized;
  return STATUS_ALIASES[normalized] ?? 'unknown';
}

/** Return `true` if the status represents a terminal (completed) execution. */
export function isTerminal(status: string | null | undefined): boolean {
  return TERMINAL_STATUSES.has(normalizeStatus(status));
}

/** Return `true` if the status represents an active, pollable execution. */
export function isActive(status: string | null | undefined): boolean {
  return ACTIVE_STATUSES.has(normalizeStatus(status));
}
