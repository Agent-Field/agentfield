"use client";

import { CopyButton } from "@/components/ui/copy-button";
import { cn } from "@/lib/utils";

/** Last `visible` characters, prefixed with an ellipsis — good for long run/workflow ids. */
export function truncateIdTail(id: string, visible: number): string {
  const v = Math.max(1, visible);
  if (id.length <= v) return id;
  return `\u2026${id.slice(-v)}`;
}

/** Middle ellipsis, similar length to legacy header truncation. */
export function truncateIdMiddle(id: string, maxLen = 20): string {
  if (id.length <= maxLen) return id;
  const keep = Math.floor((maxLen - 1) / 2);
  return `${id.slice(0, keep + 2)}\u2026${id.slice(-keep)}`;
}

export interface CopyIdentifierChipProps {
  /** Optional short field label (e.g. Run, Flow). Rendered as a micro caption. */
  label?: string;
  value: string | undefined;
  tooltip: string;
  copiedTooltip?: string;
  noValueMessage?: string;
  noValueTitle?: string;
  /** Used when `formatDisplay` is omitted. */
  idTailVisible?: number;
  formatDisplay?: (value: string) => string;
  className?: string;
}

/**
 * Inline identifier + copy control for page headers and toolbars.
 * One visual system: calm border, xs mono id, compact copy affordance.
 */
export function CopyIdentifierChip({
  label,
  value,
  tooltip,
  copiedTooltip = "Copied!",
  noValueMessage,
  noValueTitle,
  idTailVisible = 6,
  formatDisplay,
  className,
}: CopyIdentifierChipProps) {
  if (!value) {
    if (!noValueMessage) return null;
    return (
      <span
        className="text-micro text-muted-foreground/70"
        title={noValueTitle}
      >
        {noValueMessage}
      </span>
    );
  }

  const idDisplay =
    formatDisplay?.(value) ?? truncateIdTail(value, idTailVisible);

  const titleText = label ? `${label}: ${value}` : value;

  return (
    <div
      className={cn(
        "inline-flex h-7 max-w-full items-center gap-1.5 rounded-md border border-border/50 bg-muted/30 pl-2 pr-0.5",
        !label && "pl-1.5",
        className
      )}
      title={titleText}
    >
      {label ? (
        <span className="shrink-0 select-none text-micro font-semibold uppercase tracking-wider text-muted-foreground/75">
          {label}
        </span>
      ) : null}
      <span
        className="min-w-0 truncate font-mono text-xs font-medium tabular-nums leading-none text-foreground/90"
        title={value}
      >
        {idDisplay}
      </span>
      <CopyButton
        value={value}
        tooltip={tooltip}
        copiedTooltip={copiedTooltip}
        variant="ghost"
        size="icon"
        title={value}
        className="h-6 w-6 shrink-0 rounded-sm text-muted-foreground hover:bg-muted/60 hover:text-foreground [&_svg]:size-3.5"
      />
    </div>
  );
}
