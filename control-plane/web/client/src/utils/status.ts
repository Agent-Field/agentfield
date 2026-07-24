export type CanonicalStatus =
  | 'pending'
  | 'queued'
  | 'waiting'
  | 'running'
  | 'paused'
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
  'paused',
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
  paused: 'paused',
  pause: 'paused',
  hold: 'paused',
  on_hold: 'paused',
  suspended: 'paused',
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

export function isPausedStatus(status?: string | null): boolean {
  return normalizeExecutionStatus(status) === 'paused';
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
    case 'paused':
      return 'Paused';
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

import {
  Ban,
  CheckCircle2,
  Circle,
  Clock,
  HelpCircle,
  Hourglass,
  Loader2,
  PauseCircle,
  TimerOff,
  XCircle,
  type LucideIcon,
} from "lucide-react";
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
  glowColor: string;
  /** Icon component for this status — used by StatusPill, Badge, and anywhere
   * else that renders a glyph representation. */
  icon: LucideIcon;
  /** Motion behaviour. "live" = actively progressing (halo ping on the dot,
   * slow spin on the icon). Anything else = static. Only running is "live"
   * today but keeping the enum leaves room for "blocked" spinner, etc. */
  motion: "none" | "live";
}

const STATUS_TONE_MAP: Record<CanonicalStatus, ThemeStatusTone> = {
  pending: 'warning',
  queued: 'warning',
  waiting: 'warning',
  running: 'info',
  paused: 'warning',
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
  paused: 'secondary',
  succeeded: 'default',
  failed: 'destructive',
  cancelled: 'outline',
  timeout: 'outline',
  unknown: 'outline',
};

const STATUS_CSS_VAR: Record<ThemeStatusTone, string> = {
  success: "--status-success",
  warning: "--status-warning",
  error: "--status-error",
  info: "--status-info",
  neutral: "--muted-foreground",
};

const CANONICAL_TO_CSS_VAR: Record<CanonicalStatus, string> = {
  pending: STATUS_CSS_VAR[STATUS_TONE_MAP.pending],
  queued: STATUS_CSS_VAR[STATUS_TONE_MAP.queued],
  waiting: STATUS_CSS_VAR[STATUS_TONE_MAP.waiting],
  running: STATUS_CSS_VAR[STATUS_TONE_MAP.running],
  paused: STATUS_CSS_VAR[STATUS_TONE_MAP.paused],
  succeeded: STATUS_CSS_VAR[STATUS_TONE_MAP.succeeded],
  failed: STATUS_CSS_VAR[STATUS_TONE_MAP.failed],
  cancelled: STATUS_CSS_VAR[STATUS_TONE_MAP.cancelled],
  timeout: STATUS_CSS_VAR[STATUS_TONE_MAP.timeout],
  unknown: STATUS_CSS_VAR[STATUS_TONE_MAP.unknown],
};

function readCssThemeColor(varName: string): string {
  if (typeof document === "undefined") {
    return "";
  }

  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(varName)
    .trim();

  return raw ? `hsl(${raw})` : "";
}

/** Returns an hsl() string for the current theme. SSR-safe (returns empty string). */
export function getStatusHexColor(status: CanonicalStatus): string {
  return readCssThemeColor(CANONICAL_TO_CSS_VAR[status]);
}

export function getStatusGlowColor(status: CanonicalStatus): string {
  const color = getStatusHexColor(status);
  return color ? `color-mix(in srgb, ${color} 40%, transparent)` : "";
}

/** Icon glyph per canonical status. Single source of truth consumed by
 * StatusPill, StatusDot, Badge, the DAG, and anything else that wants to
 * show a status. */
const STATUS_ICON: Record<CanonicalStatus, LucideIcon> = {
  pending: Clock,
  queued: Clock,
  waiting: Hourglass,
  running: Loader2,
  paused: PauseCircle,
  succeeded: CheckCircle2,
  failed: XCircle,
  cancelled: Ban,
  timeout: TimerOff,
  unknown: HelpCircle,
};

/** Which statuses render with motion. Keeping this a small enum rather
 * than a freeform CSS string so consumers can interpret "live" in their
 * own idiomatic way (halo ping for a dot, slow spin for an icon, etc). */
const STATUS_MOTION: Record<CanonicalStatus, StatusTheme["motion"]> = {
  pending: "none",
  queued: "none",
  waiting: "none",
  running: "live",
  paused: "none",
  succeeded: "none",
  failed: "none",
  cancelled: "none",
  timeout: "none",
  unknown: "none",
};

function createStatusTheme(status: CanonicalStatus): StatusTheme {
  const toneKey = STATUS_TONE_MAP[status];
  const tone = statusTone[toneKey];
  const hexColor = getStatusHexColor(status);

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
    hexColor,
    glowColor: getStatusGlowColor(status),
    icon: STATUS_ICON[status] ?? Circle,
    motion: STATUS_MOTION[status] ?? "none",
  };
}

export function getStatusTheme(status?: string | null): StatusTheme {
  const normalized = normalizeExecutionStatus(status);
  return createStatusTheme(normalized);
}
