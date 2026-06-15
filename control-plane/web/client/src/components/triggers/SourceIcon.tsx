import type { ComponentType, SVGProps } from "react";
import {
  SiDatabricks,
  SiGithub,
  SiLinear,
  SiSentry,
  SiSlack,
  SiSnowflake,
  SiStripe,
} from "react-icons/si";
import {
  Clock,
  Key,
  Lock,
  Webhook,
} from "@/components/ui/icon-bridge";
import { cn } from "@/lib/utils";

type IconLike = ComponentType<{ className?: string } & SVGProps<SVGSVGElement>>;

const SOURCE_ICON_MAP: Record<string, IconLike> = {
  stripe: SiStripe as IconLike,
  github: SiGithub as IconLike,
  slack: SiSlack as IconLike,
  snowflake: SiSnowflake as IconLike,
  databricks: SiDatabricks as IconLike,
  linear: SiLinear as IconLike,
  sentry: SiSentry as IconLike,
  cron: Clock as IconLike,
  generic_hmac: Lock as IconLike,
  generic_bearer: Key as IconLike,
};

function getSourceIcon(sourceName: string): IconLike {
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
