import { cn } from "@/lib/utils";
import {
  getLifecycleLabel,
  getLifecycleTheme,
  normalizeLifecycleStatus,
} from "@/utils/lifecycle-status";
import {
  getStatusLabel,
  getStatusTheme,
  normalizeExecutionStatus,
} from "@/utils/status";

type StatusSize = "sm" | "md" | "lg";

const DOT_SIZE: Record<StatusSize, string> = {
  sm: "size-1.5",
  md: "size-2",
  lg: "size-2.5",
};

const ICON_SIZE: Record<StatusSize, string> = {
  sm: "size-3",
  md: "size-3.5",
  lg: "size-4",
};

const TEXT_SIZE: Record<StatusSize, string> = {
  sm: "text-[11px]",
  md: "text-xs",
  lg: "text-sm",
};

const PILL_PADDING: Record<StatusSize, string> = {
  sm: "px-2 py-0.5",
  md: "px-2.5 py-1",
  lg: "px-3 py-1.5",
};

interface StatusDotProps {
  status: string;
  size?: StatusSize;
  label?: boolean;
  className?: string;
}

export function StatusDot({
  status,
  size = "sm",
  label = true,
  className,
}: StatusDotProps) {
  const normalized = normalizeExecutionStatus(status);
  const theme = getStatusTheme(normalized);
  const sizeClass = DOT_SIZE[size];
  const isLive = theme.motion === "live";

  return (
    <span
      className={cn("inline-flex items-center gap-1.5", className)}
      data-status={normalized}
      role={label ? undefined : "img"}
      aria-label={label ? undefined : getStatusLabel(normalized)}
    >
      <span
        className={cn(
          "relative inline-flex shrink-0 items-center justify-center",
          sizeClass,
        )}
      >
        {isLive ? (
          <span
            aria-hidden
            className={cn(
              "absolute inline-flex size-full rounded-full opacity-60",
              theme.indicatorClass,
              "motion-safe:animate-ping",
            )}
          />
        ) : null}
        <span
          className={cn(
            "relative inline-flex rounded-full",
            sizeClass,
            theme.indicatorClass,
          )}
        />
      </span>
      {label ? (
        <span className={cn("leading-none", TEXT_SIZE[size], "text-foreground/90")}>
          {getStatusLabel(normalized).toLowerCase()}
        </span>
      ) : null}
    </span>
  );
}

interface StatusIconProps {
  status: string;
  size?: StatusSize;
  className?: string;
}

export function StatusIcon({
  status,
  size = "sm",
  className,
}: StatusIconProps) {
  const normalized = normalizeExecutionStatus(status);
  const theme = getStatusTheme(normalized);
  const Icon = theme.icon;
  const isLive = theme.motion === "live";

  return (
    <Icon
      aria-hidden
      data-status={normalized}
      className={cn(
        ICON_SIZE[size],
        theme.iconClass,
        isLive && "motion-safe:animate-spin",
        className,
      )}
      style={isLive ? { animationDuration: "2.5s" } : undefined}
    />
  );
}

interface StatusPillProps {
  status: string;
  size?: StatusSize;
  className?: string;
  showLabel?: boolean;
  showIcon?: boolean;
}

export function StatusPill({
  status,
  size = "md",
  className,
  showLabel = true,
  showIcon = true,
}: StatusPillProps) {
  const normalized = normalizeExecutionStatus(status);
  const theme = getStatusTheme(normalized);

  return (
    <span
      data-status={normalized}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-medium",
        PILL_PADDING[size],
        TEXT_SIZE[size],
        theme.bgClass,
        theme.borderClass,
        theme.textClass,
        className,
      )}
    >
      {showIcon ? <StatusIcon status={normalized} size={size} /> : null}
      {showLabel ? (
        <span className="capitalize leading-none">
          {getStatusLabel(normalized)}
        </span>
      ) : null}
    </span>
  );
}

interface LifecycleDotProps {
  status: string | null | undefined;
  size?: StatusSize;
  label?: boolean;
  className?: string;
}

export function LifecycleDot({
  status,
  size = "sm",
  label = true,
  className,
}: LifecycleDotProps) {
  const normalized = normalizeLifecycleStatus(status);
  const theme = getLifecycleTheme(normalized);
  const sizeClass = DOT_SIZE[size];
  const isLive = theme.motion === "live";
  const isPulse = theme.motion === "pulse";

  return (
    <span
      className={cn("inline-flex items-center gap-1.5", className)}
      data-status={normalized}
      role={label ? undefined : "img"}
      aria-label={label ? undefined : getLifecycleLabel(normalized)}
    >
      <span
        className={cn(
          "relative inline-flex shrink-0 items-center justify-center",
          sizeClass,
        )}
      >
        {isLive ? (
          <span
            aria-hidden
            className={cn(
              "absolute inline-flex size-full rounded-full opacity-60",
              theme.indicatorClass,
              "motion-safe:animate-ping",
            )}
          />
        ) : null}
        <span
          className={cn(
            "relative inline-flex rounded-full",
            sizeClass,
            theme.indicatorClass,
            isPulse && "motion-safe:animate-pulse",
          )}
        />
      </span>
      {label ? (
        <span className={cn("leading-none", TEXT_SIZE[size], theme.textClass)}>
          {getLifecycleLabel(normalized).toLowerCase()}
        </span>
      ) : null}
    </span>
  );
}

interface LifecycleIconProps {
  status: string | null | undefined;
  size?: StatusSize;
  className?: string;
}

export function LifecycleIcon({
  status,
  size = "sm",
  className,
}: LifecycleIconProps) {
  const normalized = normalizeLifecycleStatus(status);
  const theme = getLifecycleTheme(normalized);
  const Icon = theme.icon;
  const isLive = theme.motion === "live";
  const isPulse = theme.motion === "pulse";

  return (
    <Icon
      aria-hidden
      data-status={normalized}
      className={cn(
        ICON_SIZE[size],
        theme.iconClass,
        isLive && "motion-safe:animate-spin",
        isPulse && "motion-safe:animate-pulse",
        className,
      )}
      style={isLive ? { animationDuration: "2.5s" } : undefined}
    />
  );
}

interface LifecyclePillProps {
  status: string | null | undefined;
  size?: StatusSize;
  className?: string;
  showLabel?: boolean;
  showIcon?: boolean;
}

export function LifecyclePill({
  status,
  size = "md",
  className,
  showLabel = true,
  showIcon = true,
}: LifecyclePillProps) {
  const normalized = normalizeLifecycleStatus(status);
  const theme = getLifecycleTheme(normalized);

  return (
    <span
      data-status={normalized}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-medium",
        PILL_PADDING[size],
        TEXT_SIZE[size],
        theme.bgClass,
        theme.borderClass,
        theme.textClass,
        className,
      )}
    >
      {showIcon ? <LifecycleIcon status={normalized} size={size} /> : null}
      {showLabel ? (
        <span className="capitalize leading-none">
          {getLifecycleLabel(normalized)}
        </span>
      ) : null}
    </span>
  );
}
