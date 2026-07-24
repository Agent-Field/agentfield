import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { statusTone } from "@/lib/theme";
import { cn } from "@/lib/utils";
import { AlertTriangle, RefreshCw } from "@/components/ui/icon-bridge";
import type { IconComponent } from "@/components/ui/icon-bridge";

interface ErrorStateProps {
  title?: string;
  description?: string;
  error?: Error | string;
  onRetry?: () => void;
  onDismiss?: () => void;
  retrying?: boolean;
  variant?: "card" | "inline" | "banner";
  severity?: "error" | "warning" | "info";
  icon?: IconComponent;
  className?: string;
}

const severityConfig = {
  error: {
    card: cn(statusTone.error.bg, statusTone.error.border),
    inline: cn(statusTone.error.bg, statusTone.error.border),
    banner: cn(statusTone.error.bg, statusTone.error.border),
    icon: statusTone.error.accent,
    title: statusTone.error.fg,
    text: cn(statusTone.error.fg, "opacity-80"),
  },
  warning: {
    card: cn(statusTone.warning.bg, statusTone.warning.border),
    inline: cn(statusTone.warning.bg, statusTone.warning.border),
    banner: cn(statusTone.warning.bg, statusTone.warning.border),
    icon: statusTone.warning.accent,
    title: statusTone.warning.fg,
    text: cn(statusTone.warning.fg, "opacity-80"),
  },
  info: {
    card: cn(statusTone.info.bg, statusTone.info.border),
    inline: cn(statusTone.info.bg, statusTone.info.border),
    banner: cn(statusTone.info.bg, statusTone.info.border),
    icon: statusTone.info.accent,
    title: statusTone.info.fg,
    text: cn(statusTone.info.fg, "opacity-80"),
  },
};

export function ErrorState({
  title = "Something went wrong",
  description,
  error,
  onRetry,
  onDismiss,
  retrying = false,
  variant = "card",
  severity = "error",
  icon: CustomIcon,
  className
}: ErrorStateProps) {
  const Icon = CustomIcon || AlertTriangle;
  const config = severityConfig[severity];
  const errorMessage = typeof error === 'string' ? error : error?.message;

  if (variant === "banner") {
    return (
      <Card className={cn("border", config.banner, className)}>
        <CardContent className="flex items-center justify-between gap-4 py-4">
          <div className="flex items-center gap-3 text-sm">
            <Icon className={cn("h-5 w-5", config.icon)} />
            <div>
              <span className={cn("font-medium", config.title)}>{title}</span>
              {(description || errorMessage) && (
                <p className={cn("text-xs mt-0.5", config.text)}>
                  {description || errorMessage}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {onRetry && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onRetry}
                disabled={retrying}
                className="text-xs"
              >
                <RefreshCw className={cn("h-3 w-3 mr-1.5", retrying && "animate-spin")} />
                {retrying ? "Retrying" : "Retry"}
              </Button>
            )}
            {onDismiss && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onDismiss}
                className="text-xs"
              >
                Dismiss
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (variant === "inline") {
    return (
      <div className={cn(
        "flex items-center gap-3 p-3 rounded-lg border text-sm",
        config.inline,
        className
      )}>
        <Icon className={cn("h-4 w-4 shrink-0", config.icon)} />
        <div className="flex-1 min-w-0">
          <p className={cn("font-medium", config.title)}>{title}</p>
          {(description || errorMessage) && (
            <p className={cn("text-xs mt-0.5 line-clamp-2", config.text)}>
              {description || errorMessage}
            </p>
          )}
        </div>
        {onRetry && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRetry}
            disabled={retrying}
            className="shrink-0"
          >
            <RefreshCw className={cn("h-3 w-3", retrying && "animate-spin")} />
          </Button>
        )}
      </div>
    );
  }

  // Card variant (default)
  return (
    <Card className={cn("border-dashed", config.card, className)}>
      <CardHeader>
        <CardTitle className={cn("flex items-center gap-2 text-base font-semibold", config.title)}>
          <Icon className={cn("h-5 w-5", config.icon)} />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {(description || errorMessage) && (
          <p className={cn("text-sm", config.text)}>
            {description || errorMessage}
          </p>
        )}
        {(onRetry || onDismiss) && (
          <div className="flex gap-3">
            {onRetry && (
              <Button onClick={onRetry} disabled={retrying} variant="outline" size="sm">
                <RefreshCw className={cn("mr-2 h-4 w-4", retrying && "animate-spin")} />
                {retrying ? "Retrying..." : "Try again"}
              </Button>
            )}
            {onDismiss && (
              <Button variant="ghost" onClick={onDismiss} size="sm">
                Dismiss
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
