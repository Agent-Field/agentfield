"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { UnifiedJsonViewer } from "@/components/ui/UnifiedJsonViewer";
import { Empty, EmptyHeader, EmptyTitle, EmptyDescription } from "@/components/ui/empty";
import type { InboundEvent } from "./EventRow";

interface PayloadViewerProps {
  event: InboundEvent;
}

export function PayloadViewer({ event }: PayloadViewerProps) {
  const normalized = event.normalized_payload || {};
  const raw = event.raw_payload || {};
  
  const isSame = JSON.stringify(normalized) === JSON.stringify(raw);

  const headers: Record<string, string> = {
    "X-Source-Name": event.source_name,
    "X-Event-Type": event.event_type,
    "X-Trigger-ID": event.trigger_id,
    "X-Event-ID": event.id,
    ...(event.vc_id && { "X-Parent-VC-ID": event.vc_id }),
  };

  return (
    <Card variant="outline">
      <CardHeader>
        <CardTitle className="text-base">Payload</CardTitle>
        <CardDescription>Raw, normalized, and dispatch headers</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="raw" className="w-full">
          <TabsList variant="underline" className="w-full justify-start">
            <TabsTrigger value="raw">Raw</TabsTrigger>
            <TabsTrigger value="normalized">Normalized</TabsTrigger>
            <TabsTrigger value="headers">Headers</TabsTrigger>
          </TabsList>

          {/* Raw tab */}
          <TabsContent value="raw" className="mt-4">
            <UnifiedJsonViewer
              data={raw}
              showCopyButton={true}
              searchable={true}
              maxHeight="400px"
            />
          </TabsContent>

          {/* Normalized tab */}
          <TabsContent value="normalized" className="mt-4">
            {isSame ? (
              <Empty className="min-h-48">
                <EmptyHeader>
                  <EmptyTitle>Same as Raw</EmptyTitle>
                  <EmptyDescription>
                    No transformation applied
                  </EmptyDescription>
                </EmptyHeader>
              </Empty>
            ) : (
              <UnifiedJsonViewer
                data={normalized}
                showCopyButton={true}
                searchable={true}
                maxHeight="400px"
              />
            )}
          </TabsContent>

          {/* Headers tab */}
          <TabsContent value="headers" className="mt-4">
            <div className="space-y-2">
              {Object.entries(headers).map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-start justify-between gap-3 p-2 rounded-md border border-border/50 bg-muted/20"
                >
                  <span className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wide">
                    {key}
                  </span>
                  <span className="text-xs font-mono text-foreground break-all text-right max-w-xs">
                    {value}
                  </span>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
