import type { ReactNode } from "react";
import {
  Wifi, 
  WifiOff, 
  Grid, 
  Terminal,
  Renew,
  Search,
  CloudOffline
} from "@/components/ui/icon-bridge";
import { Button } from "../ui/button";
import { cn } from "../../lib/utils";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "../ui/empty";

interface EmptyReasonersStateProps {
  type: 'no-reasoners' | 'no-online' | 'no-offline' | 'no-search-results';
  searchTerm?: string;
  onRefresh?: () => void;
  onClearFilters?: () => void;
  onShowAll?: () => void;
  loading?: boolean;
  className?: string;
}

interface EmptyStateAction {
  label: string;
  action?: () => void;
  icon?: ReactNode;
}

interface EmptyStateTip {
  title: string;
  body: string;
  icon: ReactNode;
}

interface EmptyStateConfig {
  icon: ReactNode;
  title: string;
  description: string;
  primaryAction?: EmptyStateAction | null;
  secondaryAction?: EmptyStateAction | null;
  tip?: EmptyStateTip;
}

export function EmptyReasonersState({ 
  type, 
  searchTerm,
  onRefresh,
  onClearFilters,
  onShowAll,
  loading = false,
  className 
}: EmptyReasonersStateProps) {
  const getStateConfig = (): EmptyStateConfig => {
    switch (type) {
      case 'no-reasoners':
        return {
          icon: <Grid className="h-10 w-10" />,
          title: "No Reasoners Available",
          description: "There are no reasoners registered in the system yet. Connect some agent nodes to see reasoners here.",
          primaryAction: { label: "Refresh", action: onRefresh, icon: <Renew className={cn("h-4 w-4", loading && "animate-spin")} /> },
          secondaryAction: null,
          tip: {
            icon: <Terminal className="h-5 w-5 text-muted-foreground" />,
            title: "Getting started",
            body: "Launch an agent node to register reasoners with Haxen. They will appear here as soon as they are online.",
          },
        };
      
      case 'no-online':
        return {
          icon: <Wifi className="h-10 w-10" />,
          title: "No Online Reasoners",
          description: "All reasoners are currently offline. Check your agent node connections or try viewing all reasoners.",
          primaryAction: { label: "Show All Reasoners", action: onShowAll, icon: <Grid className="h-4 w-4" /> },
          secondaryAction: { label: "Refresh", action: onRefresh, icon: <Renew className={cn("h-4 w-4", loading && "animate-spin")} /> },
          tip: {
            icon: <CloudOffline className="h-5 w-5 text-muted-foreground" />,
            title: "Connection check",
            body: "Verify that your agent nodes are connected and healthy. Offline reasoners usually indicate network or configuration issues.",
          },
        };
      
      case 'no-offline':
        return {
          icon: <WifiOff className="h-10 w-10" />,
          title: "No Offline Reasoners",
          description: "Great! All your reasoners are currently online and ready to use.",
          primaryAction: { label: "Show Online Reasoners", action: onShowAll, icon: <Wifi className="h-4 w-4" /> },
          secondaryAction: { label: "Refresh", action: onRefresh, icon: <Renew className={cn("h-4 w-4", loading && "animate-spin")} /> }
        };
      
      case 'no-search-results':
        return {
          icon: <Search className="h-10 w-10" />,
          title: "No Results Found",
          description: searchTerm 
            ? `No reasoners match "${searchTerm}". Try a different search term or clear your filters.`
            : "No reasoners match your current filters. Try adjusting your search criteria.",
          primaryAction: { label: "Clear Filters", action: onClearFilters, icon: <Grid className="h-4 w-4" /> },
          secondaryAction: { label: "Refresh", action: onRefresh, icon: <Renew className={cn("h-4 w-4", loading && "animate-spin")} /> }
        };
      
      default:
        return {
          icon: <CloudOffline className="h-10 w-10" />,
          title: "Something went wrong",
          description: "Unable to load reasoners. Please try refreshing the page.",
          primaryAction: { label: "Refresh", action: onRefresh, icon: <Renew className={cn("h-4 w-4", loading && "animate-spin")} /> },
          secondaryAction: null
        };
    }
  };

  const config = getStateConfig();

  return (
    <Empty className={cn("min-h-[360px]", className)}>
      <EmptyHeader>
        <EmptyMedia variant="icon">{config.icon}</EmptyMedia>
        <EmptyTitle>{config.title}</EmptyTitle>
        <EmptyDescription>{config.description}</EmptyDescription>
      </EmptyHeader>

      {(config.primaryAction || config.secondaryAction) && (
        <EmptyContent className="gap-2 sm:gap-3">
          {config.primaryAction ? (
            <Button
              onClick={config.primaryAction.action}
              disabled={loading}
              className="inline-flex min-w-[140px] items-center gap-2"
            >
              {config.primaryAction.icon}
              {config.primaryAction.label}
            </Button>
          ) : null}
          {config.secondaryAction ? (
            <Button
              variant="outline"
              onClick={config.secondaryAction.action}
              disabled={loading}
              className="inline-flex min-w-[140px] items-center gap-2"
            >
              {config.secondaryAction.icon}
              {config.secondaryAction.label}
            </Button>
          ) : null}
        </EmptyContent>
      )}

      {config.tip && <Tip title={config.tip.title} icon={config.tip.icon} body={config.tip.body} />}
    </Empty>
  );
}

function Tip({
  title,
  body,
  icon,
}: {
  title: string;
  body: string;
  icon: ReactNode;
}) {
  return (
    <div className="mt-4 w-full max-w-md rounded-lg border border-border/40 bg-muted/15 p-4 text-left">
      <div className="flex items-start gap-3 text-body">
        <span className="mt-1 flex h-9 w-9 items-center justify-center rounded-full bg-muted/40 text-muted-foreground">
          {icon}
        </span>
        <div className="space-y-1">
          <p className="text-body font-medium text-text-primary">{title}</p>
          <p className="text-body-small text-muted-foreground leading-relaxed">
            {body}
          </p>
        </div>
      </div>
    </div>
  );
}
