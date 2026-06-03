import { cn } from "@/lib/utils";

/**
 * JSON syntax colors — driven by CSS variables in `index.css` (`--code-json-*`)
 * so light/dark themes stay consistent across Run preview, step detail, modals, etc.
 */

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

const TRUNC_SUFFIX_RE = /\n\n… truncated \([\s\S]*\)$/;

function splitTruncationSuffix(body: string): { core: string; suffix: string } {
  const m = body.match(TRUNC_SUFFIX_RE);
  if (!m || m.index === undefined) return { core: body, suffix: "" };
  return {
    core: body.slice(0, m.index),
    suffix: body.slice(m.index),
  };
}

/**
 * Token-highlight formatted JSON (e.g. from JSON.stringify(_, null, 2)).
 * Falls back safely on non-JSON fragments by emitting plain text for unknown chars.
 */
export function highlightJsonText(source: string): string {
  let i = 0;
  const n = source.length;
  const parts: string[] = [];

  const span = (cls: string, raw: string) => {
    parts.push(`<span class="${cls}">${escapeHtml(raw)}</span>`);
  };
  const plain = (raw: string) => {
    parts.push(escapeHtml(raw));
  };

  while (i < n) {
    const c = source[i];

    if (/\s/.test(c)) {
      let ws = "";
      while (i < n && /\s/.test(source[i])) ws += source[i++];
      plain(ws);
      continue;
    }

    if ("{}[],:".includes(c)) {
      span("json-hl-punct", c);
      i++;
      continue;
    }

    if (c === '"') {
      let str = "";
      str += source[i++];
      while (i < n) {
        const ch = source[i];
        if (ch === "\\") {
          str += source[i++];
          if (i < n) str += source[i++];
          continue;
        }
        if (ch === '"') {
          str += source[i++];
          break;
        }
        str += source[i++];
      }
      let j = i;
      while (j < n && /\s/.test(source[j])) j++;
      const isKey = source[j] === ":";
      span(isKey ? "json-hl-key" : "json-hl-string", str);
      continue;
    }

    if (source.startsWith("true", i)) {
      span("json-hl-boolean", "true");
      i += 4;
      continue;
    }
    if (source.startsWith("false", i)) {
      span("json-hl-boolean", "false");
      i += 5;
      continue;
    }
    if (source.startsWith("null", i)) {
      span("json-hl-null", "null");
      i += 4;
      continue;
    }

    if (/[-0-9]/.test(c)) {
      let num = "";
      if (source[i] === "-") num += source[i++];
      while (i < n && /[0-9]/.test(source[i])) num += source[i++];
      if (i < n && source[i] === ".") {
        num += source[i++];
        while (i < n && /[0-9]/.test(source[i])) num += source[i++];
      }
      if (i < n && (source[i] === "e" || source[i] === "E")) {
        num += source[i++];
        if (i < n && (source[i] === "+" || source[i] === "-")) num += source[i++];
        while (i < n && /[0-9]/.test(source[i])) num += source[i++];
      }
      span("json-hl-number", num);
      continue;
    }

    plain(c);
    i++;
  }

  return parts.join("");
}

export function jsonStringifyFormatted(data: unknown): string {
  try {
    return JSON.stringify(data, null, 2);
  } catch {
    return String(data);
  }
}

/** Pretty-print JSON with optional truncation suffix (same pattern as run hover preview). */
export function formatTruncatedFormattedJson(
  value: unknown,
  maxChars: number,
  emptyPlaceholder = "—",
): string {
  if (value === null || value === undefined) return emptyPlaceholder;
  if (typeof value === "string" && value.trim() === "") return emptyPlaceholder;
  try {
    const raw = JSON.stringify(value, null, 2);
    if (raw.length <= maxChars) return raw;
    return `${raw.slice(0, maxChars)}\n\n… truncated (${raw.length.toLocaleString()} chars total)`;
  } catch {
    const s = String(value);
    if (s.length <= maxChars) return s;
    return `${s.slice(0, maxChars)}\n\n… truncated (${s.length.toLocaleString()} chars total)`;
  }
}

type JsonHighlightedPreProps = {
  /** Raw display text (already formatted); use "—" for empty placeholder */
  text?: string;
  /** Serialize with JSON.stringify(null, 2) and highlight */
  data?: unknown;
  className?: string;
};

export function JsonHighlightedPre({ text, data, className }: JsonHighlightedPreProps) {
  const raw =
    text !== undefined
      ? text
      : data !== undefined
        ? jsonStringifyFormatted(data)
        : "—";

  if (raw === "—") {
    return (
      <pre
        className={cn(
          "json-hl-root m-0 whitespace-pre-wrap break-all [overflow-wrap:anywhere]",
          className,
        )}
      >
        —
      </pre>
    );
  }

  const { core, suffix } = splitTruncationSuffix(raw);
  const html = highlightJsonText(core) + (suffix ? escapeHtml(suffix) : "");

  return (
    <pre
      className={cn(
        "json-hl-root m-0 whitespace-pre-wrap break-all [overflow-wrap:anywhere]",
        className,
      )}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
