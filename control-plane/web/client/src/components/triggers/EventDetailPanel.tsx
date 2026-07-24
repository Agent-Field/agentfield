"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { VerificationCard } from "./VerificationCard";
import { PayloadViewer } from "./PayloadViewer";
import { VCChainCard } from "./VCChainCard";
import type { InboundEvent } from "./EventRow";

interface EventDetailPanelProps {
  event: InboundEvent;
  triggerID: string;
  onReplayed?: () => void;
}

export function EventDetailPanel({
  event,
  triggerID,
  onReplayed,
}: EventDetailPanelProps) {
  const [isReplaying, setIsReplaying] = React.useState(false);

  const handleReplay = async () => {
    setIsReplaying(true);
    try {
      const response = await fetch(
        `/api/v1/triggers/${triggerID}/events/${event.id}/replay`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        }
      );
      if (response.ok) {
        onReplayed?.();
      }
    } catch (error) {
      // Error handled by parent component
      console.error("Replay failed:", error);
    } finally {
      setIsReplaying(false);
    }
  };

  const handleCopyAsFixture = () => {
    const fixture = {
      body: event.raw_payload,
      headers: {},
    };
    navigator.clipboard.writeText(JSON.stringify(fixture, null, 2));
  };

  return (
    <div className="space-y-4">
      {/* Three cards stacked vertically */}
      <VerificationCard event={event} />

      <Separator className="my-2" />

      <PayloadViewer event={event} />

      <Separator className="my-2" />

      <VCChainCard event={event} triggerID={triggerID} />

      {/* Footer action row */}
      <Separator className="my-2" />

      <div className="flex gap-2 justify-end">
        <Button
          variant="outline"
          size="sm"
          onClick={handleCopyAsFixture}
          className="gap-1.5"
        >
          Copy as fixture
        </Button>
        <Button
          variant="default"
          size="sm"
          onClick={handleReplay}
          disabled={isReplaying}
          className="gap-1.5"
        >
          {isReplaying ? "Replaying..." : "Replay"}
        </Button>
      </div>
    </div>
  );
}
