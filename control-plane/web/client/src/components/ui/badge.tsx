import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "../../lib/utils"
import { getStatusBadgeClasses, statusTone, type StatusTone } from "../../lib/theme"
import { getStatusTheme, type CanonicalStatus } from "../../utils/status"
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  AlertTriangle,
  HelpCircle,
  type LucideIcon,
} from "lucide-react"

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-md border border-transparent px-2 py-0.5 text-xs font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "bg-muted text-foreground border border-border hover:bg-muted shadow-sm",
        secondary:
          "bg-muted text-muted-foreground border border-border/50 hover:bg-muted shadow-sm",
        destructive:
          cn(
            statusTone.error.bg,
            statusTone.error.fg,
            statusTone.error.border,
            "shadow-sm"
          ),
        outline:
          "text-foreground border border-border bg-transparent hover:bg-accent shadow-sm",
        metadata:
          "rounded-md bg-muted/40 text-muted-foreground border border-border/60 px-1.5 py-0.5 text-micro font-medium font-mono",
        count:
          "rounded-full bg-muted text-foreground border border-border px-2 py-0.5 text-micro font-semibold uppercase tracking-wide",
        pill:
          "rounded-full bg-muted/30 text-foreground border border-border/40 px-2.5 py-0.5 text-micro-plus",

        // Tooltip variant – -style chip optimized for dark tooltip backgrounds
        tooltip:
          "bg-white/15 text-primary-foreground border border-white/20 rounded-md",

        // Status variants with standardized colors and icons
        success:
          cn(getStatusBadgeClasses("success" satisfies StatusTone), "font-mono tracking-tight"),
        failed:
          cn(getStatusBadgeClasses("error" satisfies StatusTone), "font-mono tracking-tight"),
        running:
          cn(getStatusBadgeClasses("info" satisfies StatusTone), "font-mono tracking-tight"),
        pending:
          cn(getStatusBadgeClasses("warning" satisfies StatusTone), "font-mono tracking-tight"),

        // Additional status variants for degraded states
        degraded:
          cn(getStatusBadgeClasses("warning" satisfies StatusTone), "font-mono tracking-tight"),
        unknown:
          cn(getStatusBadgeClasses("neutral" satisfies StatusTone), "font-mono tracking-tight"),
      },
      size: {
        sm: "px-1.5 py-0 text-micro",
        md: "px-2 py-0.5 text-xs",
        lg: "px-3 py-1 text-sm",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  }
)

type BadgeVariant = NonNullable<VariantProps<typeof badgeVariants>["variant"]>;

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {
  icon?: React.ReactNode;
  showIcon?: boolean;
}

const STATUS_VARIANT_META: Partial<Record<BadgeVariant, {
  icon: LucideIcon
  tone: StatusTone
  canonical: CanonicalStatus
}>> = {
  success: { icon: CheckCircle2, tone: "success", canonical: "succeeded" },
  failed: { icon: XCircle, tone: "error", canonical: "failed" },
  running: { icon: Loader2, tone: "info", canonical: "running" },
  pending: { icon: Clock, tone: "warning", canonical: "pending" },
  degraded: { icon: AlertTriangle, tone: "warning", canonical: "paused" },
  unknown: { icon: HelpCircle, tone: "neutral", canonical: "unknown" },
  destructive: { icon: XCircle, tone: "error", canonical: "failed" },
}

function Badge({ className, variant, size, icon, showIcon = true, children, ...props }: BadgeProps) {
  const statusMeta = variant ? STATUS_VARIANT_META[variant] : undefined
  const StatusIconComponent = statusMeta?.icon
  const shouldSpinIcon = statusMeta
    ? getStatusTheme(statusMeta.canonical).motion === "live"
    : false;

  return (
    <div className={cn(badgeVariants({ variant, size }), className)} {...props}>
      {showIcon && (icon || (StatusIconComponent && (
        <StatusIconComponent
          size={12}
          className={cn(
            "flex-shrink-0",
            statusMeta ? statusTone[statusMeta.tone].accent : undefined,
            shouldSpinIcon && "motion-safe:animate-spin"
          )}
          style={
            shouldSpinIcon ? { animationDuration: "2.5s" } : undefined
          }
        />
      )))}
      {children}
    </div>
  )
}

// Convenience status badge components
function StatusBadge({
  status,
  children,
  className,
  ...props
}: Omit<BadgeProps, 'variant'> & {
  status: 'success' | 'failed' | 'running' | 'pending' | 'degraded' | 'unknown'
}) {
  return (
    <Badge variant={status} className={className} {...props}>
      {children || status.toUpperCase()}
    </Badge>
  );
}

export { Badge, StatusBadge, badgeVariants }
