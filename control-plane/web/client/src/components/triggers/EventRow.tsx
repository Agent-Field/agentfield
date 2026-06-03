"use client";

import * as React from "react";
import { ChevronDown, ChevronRight } from "@/components/ui/icon-bridge";
import { Badge } from "@/components/ui/badge";
import { CopyIdentifierChip } from "@/components/ui/copy-identifier-chip";
import { TimestampDisplay } from "@/components/ui/data-formatters";
import { cn } from "@/lib/utils";
import { EventDetailPanel } from "./EventDetailPanel";

export interface InboundEvent {
  id: string;
  trigger_id: string;
  source_name: string;
  event_type: string;
  raw_payload: Record<string, unknown>;
  normalized_payload?: Record<string, unknown>;
  idempotency_key: string;
  vc_id?: string;
  status: "received" | "dispatched" | "failed" | "replayed";
  error_message?: string;
  received_at: string;
  processed_at?: string;
}

interface EventRowProps {
  event: InboundEvent;
  triggerID: string;
  defaultExpanded?: boolean;
  onReplayed?: () => void;
}

const statusBadgeVariant: Record<InboundEvent["status"], string> = {
  received: "default",
  dispatched: "default",
  failed: "destructive",
  replayed: "secondary",
};

export function EventRow({
  event,
  triggerID,
  defaultExpanded = false,
  onReplayed,
}: EventRowProps) {
  const [expanded, setExpanded] = React.useState(defaultExpanded);

  const handleToggle = () => {
    setExpanded((prev) => !prev);
  };

  const relativeTime = event.received_at ? (
    <TimestampDisplay timestamp={event.received_at} format="relative" />
  ) : (
    "—"
  );

  return (
    <div className="space-y-0">
      {/* Collapsed row */}
      <div
        role="button"
        tabIndex={0}
        onClick={handleToggle}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            handleToggle();
          }
        }}
        className={cn(
          "w-full flex items-center gap-3 px-4 py-3 border border-border rounded-md transition-colors",
          "hover:bg-muted/50 active:bg-muted/60",
          expanded && "rounded-b-none border-b-0"
        )}
        aria-expanded={expanded}
      >
        {/* Chevron toggle */}
        <div className="flex-shrink-0 text-muted-foreground">
          {expanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </div>

        {/* Source + Event Type */}
        <div className="flex-shrink-0 flex flex-col gap-0.5 min-w-0">
          <span className="text-xs text-muted-foreground font-mono uppercase tracking-wide">
            {event.source_name}
          </span>
          <span className="text-sm font-medium truncate">
            {event.event_type}
          </span>
        </div>

        {/* Status badge */}
        <div className="flex-shrink-0">
          <Badge variant={statusBadgeVariant[event.status] as unknown as "default"}>
            {event.status}
          </Badge>
        </div>

        {/* Idempotency key chip */}
        <div className="flex-shrink-0">
          <CopyIdentifierChip
            label="KEY"
            value={event.idempotency_key}
            tooltip="Copy idempotency key"
            idTailVisible={6}
          />
        </div>

        {/* Relative time */}
        <div className="flex-shrink-0 text-sm text-muted-foreground">
          {relativeTime}
        </div>

        {/* Spacer */}
        <div className="flex-grow" />
      </div>

      {/* Expanded detail panel */}
      {expanded && (
        <div
          className={cn(
            "px-4 py-4 border border-border border-t-0 rounded-b-md",
            "bg-muted/20"
          )}
        >
          <EventDetailPanel
            event={event}
            triggerID={triggerID}
            onReplayed={onReplayed}
          />
        </div>
      )}
    </div>
  );
}
