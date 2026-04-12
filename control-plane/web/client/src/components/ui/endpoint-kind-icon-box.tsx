import { cn } from "@/lib/utils";
import { ReasonerIcon, SkillIcon } from "@/components/ui/icon-bridge";

/** Thinking endpoints vs deterministic skills — see `ReasonerIcon` / `SkillIcon` in icon-bridge. */
export type EndpointKind = "reasoner" | "skill";

const box =
  "flex shrink-0 items-center justify-center rounded-md border border-border bg-background text-muted-foreground";

/**
 * Tile for reasoner (thinking) vs skill (deterministic) rows — Agents list, tables, etc.
 */
export function EndpointKindIconBox({
  kind,
  className,
  iconClassName,
  compact,
}: {
  kind: EndpointKind;
  className?: string;
  iconClassName?: string;
  /** Slightly smaller tile for dense popovers. */
  compact?: boolean;
}) {
  const Icon = kind === "skill" ? SkillIcon : ReasonerIcon;
  return (
    <span
      className={cn(
        box,
        compact ? "size-7" : "size-8",
        className
      )}
    >
      <Icon
        className={cn(
          "shrink-0",
          compact ? "size-3.5" : "size-4",
          iconClassName
        )}
        aria-hidden
      />
    </span>
  );
}
