"use client";

import { useNavigate } from "react-router-dom";
import { ChevronDown, ArrowDown } from "@/components/ui/icon-bridge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CopyIdentifierChip } from "@/components/ui/copy-identifier-chip";
import { Empty, EmptyHeader, EmptyTitle, EmptyDescription } from "@/components/ui/empty";
import type { InboundEvent } from "./EventRow";

interface VCChainCardProps {
  event: InboundEvent;
  triggerID: string;
}

export function VCChainCard({ event }: VCChainCardProps) {
  const navigate = useNavigate();

  if (!event.vc_id) {
    return (
      <Card variant="outline">
        <CardHeader>
          <CardTitle className="text-base">Verifiable Credential Chain</CardTitle>
          <CardDescription>Cryptographic audit trail</CardDescription>
        </CardHeader>
        <CardContent>
          <Empty className="min-h-32">
            <EmptyHeader>
              <EmptyTitle>DID Not Enabled</EmptyTitle>
              <EmptyDescription>
                No VC chain for this event. Enable DID in trigger configuration.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        </CardContent>
      </Card>
    );
  }

  const handleViewProvenance = () => {
    navigate(`/verify?vc=${event.vc_id}`);
  };

  return (
    <Card variant="outline">
      <CardHeader>
        <CardTitle className="text-base">Verifiable Credential Chain</CardTitle>
        <CardDescription>Cryptographic audit trail</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Chain visualization */}
          <div className="flex flex-col items-start gap-3">
            {/* Trigger event VC */}
            <div className="w-full flex flex-col gap-1.5">
              <span className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wide">
                Trigger Event VC
              </span>
              <CopyIdentifierChip
                value={event.vc_id}
                tooltip="Copy VC ID"
                idTailVisible={8}
              />
            </div>

            {/* Chevron down */}
            <div className="flex justify-center w-full">
              <div className="text-muted-foreground/60">
                <ArrowDown className="h-4 w-4" />
              </div>
            </div>

            {/* Execution VC (placeholder) */}
            <div className="w-full flex flex-col gap-1.5">
              <span className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wide">
                Execution VC
              </span>
              <span className="text-xs text-muted-foreground/70">
                (Resolved via run lookup, optional)
              </span>
            </div>
          </div>

          {/* View provenance button */}
          <div className="pt-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleViewProvenance}
              className="gap-1.5 text-primary hover:text-primary"
            >
              View provenance
              <ChevronDown className="h-4 w-4 -rotate-90" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
