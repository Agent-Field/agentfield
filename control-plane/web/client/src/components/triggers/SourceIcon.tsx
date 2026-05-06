import type { ComponentType, SVGProps } from "react";
import {
  Clock,
  GithubLogo,
  Key,
  Lock,
  Webhook,
} from "@/components/ui/icon-bridge";
import { cn } from "@/lib/utils";

type IconLike = ComponentType<{ className?: string } & SVGProps<SVGSVGElement>>;

function StripeGlyph({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden
    >
      <path d="M13.479 9.883c-1.626-.604-2.512-1.067-2.512-1.803 0-.622.511-.977 1.42-.977 1.658 0 3.36.643 4.527 1.221l.668-4.105C16.65 4.766 14.84 4.211 12.62 4.211c-1.846 0-3.39.475-4.491 1.366-1.151.93-1.738 2.275-1.738 3.91 0 2.969 1.811 4.241 4.764 5.314 1.901.694 2.535 1.182 2.535 1.93 0 .724-.617 1.142-1.728 1.142-1.398 0-3.701-.689-5.193-1.563L6.07 19.49c1.288.732 3.69 1.475 6.182 1.475 1.951 0 3.589-.464 4.685-1.337 1.231-.978 1.872-2.413 1.872-4.262 0-3.034-1.844-4.305-5.33-5.483z" />
    </svg>
  );
}

function SlackGlyph({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden
    >
      <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zm1.271 0a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zm0 1.271a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zm10.122 2.521a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zm-1.268 0a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zm-2.523 10.122a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zm0-1.268a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z" />
    </svg>
  );
}

const SOURCE_ICON_MAP: Record<string, IconLike> = {
  stripe: StripeGlyph as IconLike,
  github: GithubLogo as IconLike,
  slack: SlackGlyph as IconLike,
  cron: Clock as IconLike,
  generic_hmac: Lock as IconLike,
  generic_bearer: Key as IconLike,
};

export function getSourceIcon(sourceName: string): IconLike {
  const key = sourceName.toLowerCase();
  return SOURCE_ICON_MAP[key] ?? (Webhook as IconLike);
}

interface SourceIconProps {
  source: string;
  className?: string;
  iconClassName?: string;
  /** Tile size; default size-7. Pass `compact` for size-6. */
  size?: "compact" | "default" | "lg";
}

/**
 * Bordered icon tile for a Source plugin — same composition as
 * EndpointKindIconBox so it sits next to other tiles consistently.
 */
export function SourceIcon({
  source,
  className,
  iconClassName,
  size = "default",
}: SourceIconProps) {
  const Glyph = getSourceIcon(source);
  const tile =
    size === "compact"
      ? "size-6"
      : size === "lg"
        ? "size-10"
        : "size-7";
  const icon =
    size === "compact"
      ? "size-3.5"
      : size === "lg"
        ? "size-5"
        : "size-4";
  return (
    <span
      className={cn(
        "flex shrink-0 items-center justify-center rounded-md border border-border bg-background text-muted-foreground",
        tile,
        className,
      )}
    >
      <Glyph className={cn("shrink-0", icon, iconClassName)} />
    </span>
  );
}
