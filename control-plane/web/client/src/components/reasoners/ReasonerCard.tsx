import type { KeyboardEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import type { ReasonerCardProps } from '../../types/reasoners';
import { ReasonerStatusDot } from './ReasonerStatusDot';
import { CompositeDIDStatus } from '../did/DIDStatusBadge';
import { useDIDStatus } from '../../hooks/useDIDInfo';
import { ReasonerIcon, Layers, Timer, Tag, Flash, CheckCircle, BarChart3, Identification } from '@/components/ui/icon-bridge';
import { Card } from '@/components/ui/card';

export function ReasonerCard({ reasoner, onClick }: ReasonerCardProps) {
  const navigate = useNavigate();

  // Get DID status for the reasoner
  const { status: didStatus } = useDIDStatus(reasoner.reasoner_id);

  const handleClick = () => {
    if (onClick) {
      onClick(reasoner);
    } else {
      // Navigate to reasoner detail page
      // reasoner_id already contains the full format: "node_id.reasoner_name"
      navigate(`/reasoners/${encodeURIComponent(reasoner.reasoner_id)}`);
    }
  };

  const getStatusFromNodeStatus = (nodeStatus: string) => {
    switch (nodeStatus) {
      case 'active':
        return 'online';
      case 'inactive':
        return 'offline';
      default:
        return 'unknown';
    }
  };

  const status = getStatusFromNodeStatus(reasoner.node_status);
  const isOffline = status === 'offline';

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleClick();
    }
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;

    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;

    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <Card
      variant="default"
      interactive={true}
      role="button"
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      aria-label={`View reasoner ${reasoner.name}`}
      className={cn(
        "group flex h-full flex-col gap-4 p-4 transition-transform duration-200 cursor-pointer",
        "hover:-translate-y-[1px]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        isOffline && "opacity-75"
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 flex-1 items-start gap-3">
          <div className="mt-0.5 flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
            <ReasonerIcon className="h-4 w-4" aria-hidden />
          </div>
          <div className="min-w-0 space-y-1">
            <h3
              className="line-clamp-2 text-sm font-medium leading-tight text-foreground transition-colors group-hover:text-foreground"
              title={reasoner.name}
            >
              {reasoner.name}
            </h3>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
              <span className="truncate">{reasoner.node_id}</span>
              <span className="text-muted-foreground/50">•</span>
              <span className="whitespace-nowrap">
                Updated {formatTimeAgo(reasoner.last_updated)}
              </span>
            </div>
          </div>
        </div>
        <div className="mt-0.5 flex flex-shrink-0 items-center gap-2">
          <ReasonerStatusDot status={status} size="sm" />
          {didStatus && didStatus.has_did && (
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <Identification className="h-3 w-3" />
              <CompositeDIDStatus
                status={didStatus.did_status}
                reasonerCount={didStatus.reasoner_count}
                skillCount={didStatus.skill_count}
                compact={true}
                className="text-xs"
              />
            </div>
          )}
        </div>
      </div>

      <p className="min-h-[2.5rem] text-xs leading-relaxed text-muted-foreground line-clamp-3">
        {reasoner.description}
      </p>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <Layers
            className="h-3 w-3 flex-shrink-0"
          />
          <span className="whitespace-nowrap">
            {reasoner.memory_config?.cache_results ? "Cached" : "No cache"}
          </span>
        </div>
        {reasoner.memory_config?.memory_retention && (
          <div className="flex items-center gap-1.5">
            <Timer className="h-3 w-3 flex-shrink-0" />
            <span className="whitespace-nowrap">
              {reasoner.memory_config.memory_retention}
            </span>
          </div>
        )}
        <div className="flex items-center gap-1.5">
          <Tag className="h-3 w-3 flex-shrink-0" />
          <span className="whitespace-nowrap">v{reasoner.node_version}</span>
        </div>
      </div>

      {(reasoner.avg_response_time_ms ||
        reasoner.success_rate ||
        reasoner.total_runs) && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted-foreground">
          {reasoner.avg_response_time_ms && (
            <div className="flex items-center gap-1">
              <Flash className="h-3 w-3 flex-shrink-0" />
              <span className="whitespace-nowrap">
                {reasoner.avg_response_time_ms}ms avg
              </span>
            </div>
          )}
          {reasoner.success_rate && (
            <div className="flex items-center gap-1">
              <CheckCircle
                className="h-3 w-3 flex-shrink-0 text-status-success"
              />
              <span className="whitespace-nowrap">
                {(reasoner.success_rate * 100).toFixed(1)}% success
              </span>
            </div>
          )}
          {reasoner.total_runs && (
            <div className="flex items-center gap-1">
              <BarChart3 className="h-3 w-3 flex-shrink-0" />
              <span className="whitespace-nowrap">
                {reasoner.total_runs.toLocaleString()} runs
              </span>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
