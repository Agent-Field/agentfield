import React from 'react';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';

interface KPICardProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  primaryValue: string | number;
  secondaryInfo?: string;
  onClick?: () => void;
  status?: 'success' | 'warning' | 'error';
  loading?: boolean;
}

export const KPICard = React.memo(function KPICard({
  icon: Icon,
  label,
  primaryValue,
  secondaryInfo,
  onClick,
  status,
  loading = false
}: KPICardProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'success':
        return 'var(--status-success)';
      case 'warning':
        return 'var(--status-warning)';
      case 'error':
        return 'var(--status-error)';
      default:
        return 'hsl(var(--muted-foreground))';
    }
  };

  if (loading) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
        <div className="space-y-4">
          <div className="flex items-center space-x-3">
            <Skeleton className="h-5 w-5 rounded" />
            <Skeleton className="h-4 w-20" />
          </div>
          <Skeleton className="h-8 w-16" />
          <Skeleton className="h-3 w-24" />
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        // Base styling inspired by Linear.app
        "rounded-xl border bg-card shadow-sm transition-all duration-200",
        "p-6", // 24px padding
        "border-border/60", // Softer border
        // Hover effects - more subtle like Linear
        "hover:border-border hover:shadow-lg hover:bg-card/80",
        // Click behavior with subtle scaling
        onClick && "cursor-pointer hover:scale-[1.01] active:scale-[0.99]",
        // Better focus states
        onClick && "focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/40"
      )}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      aria-label={onClick ? `View ${label} details` : undefined}
    >
      <div className="space-y-3">
        {/* Icon and Label Row */}
        <div className="flex items-center space-x-3">
          <div
            className="h-5 w-5 flex items-center justify-center"
            style={{ color: getStatusColor() }}
          >
            <Icon className="h-5 w-5" />
          </div>
          <span className="text-sm text-muted-foreground font-medium text-muted-foreground tracking-tight">
            {label}
          </span>
        </div>

        {/* Primary Value */}
        <div className="text-2xl font-semibold tracking-tight leading-tight tracking-tight">
          {primaryValue === '—' ? (
            <span className="text-muted-foreground">—</span>
          ) : (
            primaryValue
          )}
        </div>

        {/* Secondary Info */}
        {secondaryInfo && (
          <div className="text-sm text-muted-foreground text-muted-foreground/80 leading-relaxed">
            {secondaryInfo}
          </div>
        )}
      </div>
    </div>
  );
});
