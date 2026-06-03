import type { ReasonerStatus } from '../../types/reasoners';

interface ReasonerStatusDotProps {
  status: ReasonerStatus;
  showText?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export function ReasonerStatusDot({ status, showText = true, size = 'md' }: ReasonerStatusDotProps) {
  const getStatusConfig = (status: ReasonerStatus) => {
    switch (status) {
      case 'online':
        return {
          label: 'Online',
          dot: 'bg-status-success',
          text: 'text-status-success',
          chip: 'bg-status-success/10 border-status-success/30'
        };
      case 'degraded':
        return {
          label: 'Limited',
          dot: 'bg-status-warning',
          text: 'text-status-warning',
          chip: 'bg-status-warning/10 border-status-warning/30'
        };
      case 'offline':
        return {
          label: 'Offline',
          dot: 'bg-muted',
          text: 'text-muted-foreground',
          chip: 'bg-muted border-border'
        };
      case 'unknown':
      default:
        return {
          label: 'Unknown',
          dot: 'bg-muted',
          text: 'text-muted-foreground',
          chip: 'bg-muted border-border'
        };
    }
  };

  const getSizeConfig = (size: 'sm' | 'md' | 'lg') => {
    switch (size) {
      case 'sm':
        return { dot: 'w-2 h-2', text: 'text-xs', gap: 'gap-1.5' };
      case 'lg':
        return { dot: 'w-3 h-3', text: 'text-sm', gap: 'gap-2' };
      case 'md':
      default:
        return { dot: 'w-2.5 h-2.5', text: 'text-xs', gap: 'gap-2' };
    }
  };

  const statusConfig = getStatusConfig(status);
  const sizeConfig = getSizeConfig(size);

  if (!showText) {
    return (
      <div className={`${sizeConfig.dot} ${statusConfig.dot} rounded-full flex-shrink-0`} />
    );
  }

  return (
    <div className={`inline-flex items-center ${sizeConfig.gap} px-2 py-1 rounded-full border ${statusConfig.chip}`}>
      <div className={`${sizeConfig.dot} ${statusConfig.dot} rounded-full flex-shrink-0`} />
      <span className={`${sizeConfig.text} font-medium ${statusConfig.text}`}>
        {statusConfig.label}
      </span>
    </div>
  );
}
