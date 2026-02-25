export type CanonicalStatus =
  | 'pending'
  | 'queued'
  | 'waiting'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'timeout'
  | 'unknown';

const CANONICAL_STATUS_SET = new Set<CanonicalStatus>([
  'pending',
  'queued',
  'waiting',
  'running',
  'succeeded',
  'failed',
  'cancelled',
  'timeout',
  'unknown',
]);

const STATUS_MAP: Record<string, CanonicalStatus> = {
  pending: 'pending',
  queued: 'queued',
  wait: 'queued', // legacy: short alias preserved for backward compat
  waiting: 'waiting',
  awaiting_approval: 'waiting',
  awaiting_human: 'waiting',
  approval_pending: 'waiting',
  running: 'running',
  processing: 'running',
  in_progress: 'running',
  success: 'succeeded',
  succeeded: 'succeeded',
  completed: 'succeeded',
  verified: 'succeeded',
  done: 'succeeded',
  failed: 'failed',
  failure: 'failed',
  error: 'failed',
  cancelled: 'cancelled',
  canceled: 'cancelled',
  timeout: 'timeout',
  timed_out: 'timeout',
};

export function normalizeExecutionStatus(status?: string | null): CanonicalStatus {
  if (!status) {
    return 'unknown';
  }
  const key = status.trim().toLowerCase();
  const mapped = STATUS_MAP[key];
  if (mapped) {
    return mapped;
  }
  if (CANONICAL_STATUS_SET.has(key as CanonicalStatus)) {
    return key as CanonicalStatus;
  }
  return 'unknown';
}

export function isTerminalStatus(status?: string | null): boolean {
  const normalized = normalizeExecutionStatus(status);
  return normalized === 'succeeded' || normalized === 'failed' || normalized === 'cancelled' || normalized === 'timeout';
}

export function isSuccessStatus(status?: string | null): boolean {
  return normalizeExecutionStatus(status) === 'succeeded';
}

export function isFailureStatus(status?: string | null): boolean {
  return normalizeExecutionStatus(status) === 'failed';
}

export function isCancelledStatus(status?: string | null): boolean {
  return normalizeExecutionStatus(status) === 'cancelled';
}

export function isTimeoutStatus(status?: string | null): boolean {
  return normalizeExecutionStatus(status) === 'timeout';
}

export function isRunningStatus(status?: string | null): boolean {
  return normalizeExecutionStatus(status) === 'running';
}

export function isWaitingStatus(status?: string | null): boolean {
  return normalizeExecutionStatus(status) === 'waiting';
}

export function isQueuedStatus(status?: string | null): boolean {
  const normalized = normalizeExecutionStatus(status);
  return normalized === 'queued' || normalized === 'pending';
}

export function getStatusLabel(status?: string | null): string {
  switch (normalizeExecutionStatus(status)) {
    case 'succeeded':
      return 'Succeeded';
    case 'failed':
      return 'Failed';
    case 'cancelled':
      return 'Cancelled';
    case 'timeout':
      return 'Timed Out';
    case 'running':
      return 'Running';
    case 'waiting':
      return 'Waiting';
    case 'queued':
      return 'Queued';
    case 'pending':
      return 'Pending';
    default:
      return 'Unknown';
  }
}

import { statusTone, type StatusTone as ThemeStatusTone } from "../lib/theme";

export interface StatusTheme {
  status: CanonicalStatus;
  badgeVariant: 'default' | 'secondary' | 'destructive' | 'outline';
  textClass: string;
  iconClass: string;
  dotClass: string;
  indicatorClass: string;
  pillClass: string;
  borderClass: string;
  bgClass: string;
  hexColor: string;
  iconHex: string;
  glowColor: string;
}

const STATUS_HEX: Record<CanonicalStatus, { base: string; light: string }> = {
  pending: { base: '#f59e0b', light: '#fbbf24' },
  queued: { base: '#f59e0b', light: '#fbbf24' },
  waiting: { base: '#d97706', light: '#f59e0b' },
  running: { base: '#2563eb', light: '#60a5fa' },
  succeeded: { base: '#16a34a', light: '#22c55e' },
  failed: { base: '#ef4444', light: '#f87171' },
  cancelled: { base: '#6b7280', light: '#9ca3af' },
  timeout: { base: '#8b5cf6', light: '#a78bfa' },
  unknown: { base: '#737373', light: '#9ca3af' },
};

const STATUS_TONE_MAP: Record<CanonicalStatus, ThemeStatusTone> = {
  pending: 'warning',
  queued: 'warning',
  waiting: 'warning',
  running: 'info',
  succeeded: 'success',
  failed: 'error',
  cancelled: 'neutral',
  timeout: 'info',
  unknown: 'neutral',
};

const BADGE_VARIANT: Record<CanonicalStatus, StatusTheme['badgeVariant']> = {
  pending: 'secondary',
  queued: 'secondary',
  waiting: 'secondary',
  running: 'secondary',
  succeeded: 'default',
  failed: 'destructive',
  cancelled: 'outline',
  timeout: 'outline',
  unknown: 'outline',
};

function createStatusTheme(status: CanonicalStatus): StatusTheme {
  const toneKey = STATUS_TONE_MAP[status];
  const tone = statusTone[toneKey];
  const { base, light } = STATUS_HEX[status];

  return {
    status,
    badgeVariant: BADGE_VARIANT[status],
    textClass: tone.fg,
    iconClass: tone.accent,
    dotClass: tone.dot,
    indicatorClass: tone.solidBg,
    pillClass: [tone.bg, tone.fg, tone.border].join(' '),
    borderClass: tone.border,
    bgClass: tone.bg,
    hexColor: base,
    iconHex: light,
    glowColor: `color-mix(in srgb, ${base} 40%, transparent)`,
  };
}

const STATUS_THEME: Record<CanonicalStatus, StatusTheme> = {
  pending: createStatusTheme('pending'),
  queued: createStatusTheme('queued'),
  waiting: createStatusTheme('waiting'),
  running: createStatusTheme('running'),
  succeeded: createStatusTheme('succeeded'),
  failed: createStatusTheme('failed'),
  cancelled: createStatusTheme('cancelled'),
  timeout: createStatusTheme('timeout'),
  unknown: createStatusTheme('unknown'),
};

const DEFAULT_THEME = STATUS_THEME.unknown;

export function getStatusTheme(status?: string | null): StatusTheme {
  const normalized = normalizeExecutionStatus(status);
  return STATUS_THEME[normalized] ?? DEFAULT_THEME;
}
