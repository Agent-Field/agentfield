import type { LifecycleStatus as AgentfieldLifecycleStatus } from "@/types/agentfield";
import { statusTone, type StatusTone } from "@/lib/theme";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  HelpCircle,
  Loader2,
  Square,
  WifiOff,
  XCircle,
  type LucideIcon,
} from "lucide-react";

export type LifecycleStatus = AgentfieldLifecycleStatus;

type LifecycleBadgeVariant =
  | "success"
  | "running"
  | "pending"
  | "degraded"
  | "failed"
  | "unknown";

type LifecycleMotion = "none" | "pulse" | "live";

export interface LifecycleTheme {
  status: LifecycleStatus;
  label: string;
  tone: StatusTone;
  icon: LucideIcon;
  indicatorClass: string;
  textClass: string;
  iconClass: string;
  pillClass: string;
  borderClass: string;
  bgClass: string;
  dotClass: string;
  hexColor: string;
  iconHex: string;
  glowColor: string;
  badgeVariant: LifecycleBadgeVariant;
  motion: LifecycleMotion;
  online: boolean;
}

const LIFECYCLE_STATUS_SET = new Set<LifecycleStatus>([
  "starting",
  "ready",
  "running",
  "degraded",
  "stopped",
  "offline",
  "error",
  "unknown",
]);

const LIFECYCLE_ALIASES: Record<string, LifecycleStatus> = {
  up: "ready",
  down: "offline",
  unhealthy: "degraded",
};

const LIFECYCLE_LABELS: Record<LifecycleStatus, string> = {
  starting: "Starting",
  ready: "Ready",
  running: "Running",
  degraded: "Degraded",
  stopped: "Stopped",
  offline: "Offline",
  error: "Error",
  unknown: "Unknown",
};

const LIFECYCLE_TONES: Record<LifecycleStatus, StatusTone> = {
  starting: "info",
  ready: "success",
  running: "success",
  degraded: "warning",
  stopped: "neutral",
  offline: "error",
  error: "error",
  unknown: "neutral",
};

const LIFECYCLE_ICONS: Record<LifecycleStatus, LucideIcon> = {
  starting: Clock,
  ready: CheckCircle2,
  running: Loader2,
  degraded: AlertTriangle,
  stopped: Square,
  offline: WifiOff,
  error: XCircle,
  unknown: HelpCircle,
};

const LIFECYCLE_BADGE_VARIANTS: Record<LifecycleStatus, LifecycleBadgeVariant> = {
  starting: "pending",
  ready: "success",
  running: "success",
  degraded: "degraded",
  stopped: "unknown",
  offline: "failed",
  error: "failed",
  unknown: "unknown",
};

const LIFECYCLE_MOTION: Record<LifecycleStatus, LifecycleMotion> = {
  starting: "pulse",
  ready: "none",
  running: "live",
  degraded: "pulse",
  stopped: "none",
  offline: "none",
  error: "none",
  unknown: "none",
};

const LIFECYCLE_ONLINE: Record<LifecycleStatus, boolean> = {
  starting: true,
  ready: true,
  running: true,
  degraded: true,
  stopped: false,
  offline: false,
  error: false,
  unknown: false,
};

const TONE_COLOR_VARS: Record<StatusTone, string> = {
  success: "hsl(var(--status-success))",
  warning: "hsl(var(--status-warning))",
  error: "hsl(var(--status-error))",
  info: "hsl(var(--status-info))",
  neutral: "hsl(var(--muted-foreground))",
};

function createLifecycleTheme(status: LifecycleStatus): LifecycleTheme {
  const toneKey = LIFECYCLE_TONES[status];
  const tone = statusTone[toneKey];
  const colorVar = TONE_COLOR_VARS[toneKey];

  return {
    status,
    label: LIFECYCLE_LABELS[status],
    tone: toneKey,
    icon: LIFECYCLE_ICONS[status],
    indicatorClass: tone.solidBg,
    textClass: tone.fg,
    iconClass: tone.accent,
    pillClass: [tone.bg, tone.fg, tone.border].join(" "),
    borderClass: tone.border,
    bgClass: tone.bg,
    dotClass: tone.dot,
    hexColor: colorVar,
    iconHex: colorVar,
    glowColor: `color-mix(in srgb, ${colorVar} 40%, transparent)`,
    badgeVariant: LIFECYCLE_BADGE_VARIANTS[status],
    motion: LIFECYCLE_MOTION[status],
    online: LIFECYCLE_ONLINE[status],
  };
}

export const LIFECYCLE_THEME: Record<LifecycleStatus, LifecycleTheme> = {
  starting: createLifecycleTheme("starting"),
  ready: createLifecycleTheme("ready"),
  running: createLifecycleTheme("running"),
  degraded: createLifecycleTheme("degraded"),
  stopped: createLifecycleTheme("stopped"),
  offline: createLifecycleTheme("offline"),
  error: createLifecycleTheme("error"),
  unknown: createLifecycleTheme("unknown"),
};

const DEFAULT_THEME = LIFECYCLE_THEME.unknown;

export function normalizeLifecycleStatus(
  status?: LifecycleStatus | string | null,
): LifecycleStatus {
  if (!status) {
    return "unknown";
  }

  const key = status.trim().toLowerCase();
  if (!key) {
    return "unknown";
  }

  const aliased = LIFECYCLE_ALIASES[key];
  if (aliased) {
    return aliased;
  }

  if (LIFECYCLE_STATUS_SET.has(key as LifecycleStatus)) {
    return key as LifecycleStatus;
  }

  return "unknown";
}

export function getLifecycleTheme(
  status?: LifecycleStatus | string | null,
): LifecycleTheme {
  const normalized = normalizeLifecycleStatus(status);
  return LIFECYCLE_THEME[normalized] ?? DEFAULT_THEME;
}

export function getLifecycleLabel(status?: LifecycleStatus | string | null): string {
  return getLifecycleTheme(status).label;
}

export function isLifecycleOnline(status?: LifecycleStatus | string | null): boolean {
  return getLifecycleTheme(status).online;
}
