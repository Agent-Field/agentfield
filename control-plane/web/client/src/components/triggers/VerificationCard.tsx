"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CopyIdentifierChip } from "@/components/ui/copy-identifier-chip";
import { TimestampDisplay } from "@/components/ui/data-formatters";
import type { InboundEvent } from "./EventRow";

interface VerificationCardProps {
  event: InboundEvent;
}

/**
 * Audit-friendly verification evidence card.
 * 
 * TODO: Once the SDK exposes trigger event VC's credentialSubject with
 * signature algorithm and body hash, wire those fields here.
 * For now, we render status, timestamps, and infer verification pass/fail
 * from the event status.
 */
export function VerificationCard({ event }: VerificationCardProps) {
  const statusVariant =
    event.status === "failed" ? "destructive" : 
    event.status === "replayed" ? "secondary" : 
    "default";

  const algorithmDisplay = "pending-sdk-integration";
  const bodyHashDisplay = "pending-sdk-integration";

  return (
    <Card variant="outline">
      <CardHeader>
        <CardTitle className="text-base">Verification</CardTitle>
        <CardDescription>Cryptographic audit evidence</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Status badge */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Status</span>
          <Badge variant={statusVariant as unknown as "default"}>
            {event.status === "failed" ? "Failed" : 
             event.status === "replayed" ? "Replayed" : 
             "Verified"}
          </Badge>
        </div>

        {/* Algorithm (placeholder pending SDK) */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Algorithm</span>
          <span className="text-sm font-mono text-muted-foreground/70">
            {algorithmDisplay}
          </span>
        </div>

        {/* Body hash (placeholder pending SDK) */}
        <div className="flex items-start gap-2">
          <span className="text-sm text-muted-foreground">Body Hash</span>
          <CopyIdentifierChip
            value={bodyHashDisplay}
            tooltip="Copy body hash"
            noValueMessage="—"
            idTailVisible={8}
          />
        </div>

        {/* Timestamps */}
        <div className="space-y-2 border-t border-border/50 pt-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Received</span>
            <TimestampDisplay
              timestamp={event.received_at}
              format="absolute"
              className="font-mono text-foreground"
            />
          </div>
          {event.processed_at && (
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Processed</span>
              <TimestampDisplay
                timestamp={event.processed_at}
                format="absolute"
                className="font-mono text-foreground"
              />
            </div>
          )}
        </div>

        {/* Error message if failed */}
        {event.status === "failed" && event.error_message && (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 p-2">
            <p className="text-xs text-destructive font-mono">
              {event.error_message}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
