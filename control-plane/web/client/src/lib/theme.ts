const STATUS_TONES = {
  success: {
    accent: "text-status-success",
    fg: "text-status-success",
    mutedFg: "text-status-success",
    bg: "bg-status-success/10",
    solidBg: "bg-status-success",
    solidFg: "text-primary-foreground",
    border: "border border-status-success/30",
    dot: "status-dot status-dot-success",
  },
  warning: {
    accent: "text-status-warning",
    fg: "text-status-warning",
    mutedFg: "text-status-warning",
    bg: "bg-status-warning/10",
    solidBg: "bg-status-warning",
    solidFg: "text-primary-foreground",
    border: "border border-status-warning/30",
    dot: "status-dot status-dot-pending",
  },
  error: {
    accent: "text-status-error",
    fg: "text-status-error",
    mutedFg: "text-status-error",
    bg: "bg-status-error/10",
    solidBg: "bg-status-error",
    solidFg: "text-primary-foreground",
    border: "border border-status-error/30",
    dot: "status-dot status-dot-failed",
  },
  info: {
    accent: "text-status-info",
    fg: "text-status-info",
    mutedFg: "text-status-info",
    bg: "bg-status-info/10",
    solidBg: "bg-status-info",
    solidFg: "text-primary-foreground",
    border: "border border-status-info/30",
    dot: "status-dot status-dot-running",
  },
  neutral: {
    accent: "text-muted-foreground",
    fg: "text-muted-foreground",
    mutedFg: "text-muted-foreground",
    bg: "bg-muted",
    solidBg: "bg-muted",
    solidFg: "text-foreground",
    border: "border border-border",
    dot: "status-dot bg-gray-400",
  },
} as const;

export type StatusTone = keyof typeof STATUS_TONES;

export function getStatusTone(tone: StatusTone) {
  return STATUS_TONES[tone];
}

export function getStatusBadgeClasses(tone: StatusTone) {
  const status = getStatusTone(tone);
  return [
    "inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium",
    "shadow-sm transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
    status.bg,
    status.fg,
    status.border,
  ].join(" ");
}

export const statusTone = STATUS_TONES;
