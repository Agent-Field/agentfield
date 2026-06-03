import { formatTruncatedFormattedJson } from "@/components/ui/json-syntax-highlight";

const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

export const PREVIEW_JSON_MAX = 10_000;

export function shortRunIdDisplay(runId: string, tail = 4): string {
  const normalizedTail = Math.max(2, tail);
  if (runId.length <= normalizedTail + 2) {
    return runId;
  }

  return `…${runId.slice(-normalizedTail)}`;
}

export function formatAbsoluteStarted(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }

  return date.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatRelativeStarted(
  startedMs: number,
  nowMs: number,
  liveGranular: boolean,
): string {
  const diff = Math.max(0, nowMs - startedMs);
  const seconds = Math.floor(diff / 1000);

  if (liveGranular) {
    if (seconds < 8) return "just now";
    if (seconds < 3600) {
      if (seconds < 60) return `${seconds}s ago`;
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds}s ago`;
    }

    if (seconds < 86400) {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return minutes > 0 ? `${hours}h ${minutes}m ago` : `${hours}h ago`;
    }
  } else if (seconds < 10) {
    return "just now";
  }

  if (seconds < 60) return rtf.format(-seconds, "second");
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return rtf.format(-minutes, "minute");
  const hours = Math.floor(seconds / 3600);
  if (hours < 24) return rtf.format(-hours, "hour");
  const days = Math.floor(seconds / 86400);
  if (days < 7) return rtf.format(-days, "day");
  const weeks = Math.floor(days / 7);
  if (weeks < 8) return rtf.format(-weeks, "week");
  const months = Math.floor(days / 30);
  if (months < 12) return rtf.format(-months, "month");
  const years = Math.floor(days / 365);
  return rtf.format(-Math.max(1, years), "year");
}

export function formatDuration(ms: number | undefined, terminal?: boolean): string {
  if (!terminal && ms == null) return "—";
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;

  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    const remainingSeconds = Math.round(seconds % 60);
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const remainingMinutes = minutes % 60;
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
  }

  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;
  return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`;
}

export function hasMeaningfulPayload(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value as object).length > 0;
  return true;
}

export function formatPreviewJson(value: unknown): string {
  return formatTruncatedFormattedJson(value, PREVIEW_JSON_MAX);
}

export function getPaginationPages(
  current: number,
  total: number,
): Array<number | "ellipsis"> {
  if (total < 1) return [];
  if (total <= 7) {
    return Array.from({ length: total }, (_, index) => index + 1);
  }

  const pages = new Set([1, total, current, current - 1, current + 1]);
  const sortedPages = [...pages]
    .filter((page) => page >= 1 && page <= total)
    .sort((a, b) => a - b);

  const output: Array<number | "ellipsis"> = [];
  let previous = 0;
  for (const page of sortedPages) {
    if (page - previous > 1) {
      output.push("ellipsis");
    }
    output.push(page);
    previous = page;
  }

  return output;
}
