import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CopyButton } from "@/components/ui/copy-button";
import { CopyIdentifierChip } from "@/components/ui/copy-identifier-chip";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Switch } from "@/components/ui/switch";
import {
  AnimatedTabs,
  AnimatedTabsContent,
  AnimatedTabsList,
  AnimatedTabsTrigger,
} from "@/components/ui/animated-tabs";
import {
  Activity,
  ArrowLeftRight,
  ArrowUpRight,
  Lock,
  PauseFilled,
  SlidersHorizontal,
  Trash,
} from "@/components/ui/icon-bridge";
import { SourceIcon } from "@/components/triggers/SourceIcon";
import {
  EventRow,
  type InboundEvent as EventRowInboundEvent,
} from "@/components/triggers/EventRow";

interface Trigger {
  id: string;
  source_name: string;
  config: Record<string, unknown> | string | null;
  secret_env_var: string;
  target_node_id: string;
  target_reasoner: string;
  event_types?: string[] | null;
  managed_by: "code" | "ui";
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

interface TriggerSheetProps {
  open: boolean;
  trigger: Trigger | null;
  serverUrl: string;
  publicUrl: string;
  busy?: boolean;
  onOpenChange: (open: boolean) => void;
  onEnabledChange: (enabled: boolean) => void;
  onDelete: () => void;
}

type InboundEvent = EventRowInboundEvent;

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatConfig(config: Trigger["config"]): string {
  if (typeof config === "string") return config;
  return JSON.stringify(config ?? {}, null, 2);
}

export function TriggerSheet({
  open,
  trigger,
  serverUrl,
  publicUrl,
  busy = false,
  onOpenChange,
  onEnabledChange,
  onDelete,
}: TriggerSheetProps) {
  const navigate = useNavigate();
  const [events, setEvents] = useState<InboundEvent[]>([]);
  const [loadingEvents, setLoadingEvents] = useState(false);
  const [eventError, setEventError] = useState<string | null>(null);

  const triggerId = trigger?.id;
  const isCodeManaged = trigger?.managed_by === "code";

  const eventTypeLabel = useMemo(() => {
    if (!trigger) return "";
    if (!trigger.event_types?.length) return "All events";
    return trigger.event_types.join(", ");
  }, [trigger]);

  const refreshEvents = useCallback(async () => {
    if (!triggerId) return;
    try {
      setLoadingEvents(true);
      const res = await fetchJson<{ events: InboundEvent[] }>(
        `${serverUrl}/api/v1/triggers/${triggerId}/events`,
      );
      setEvents(res.events || []);
      setEventError(null);
    } catch (e) {
      setEventError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingEvents(false);
    }
  }, [serverUrl, triggerId]);

  useEffect(() => {
    if (open && triggerId) void refreshEvents();
  }, [open, refreshEvents, triggerId]);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="flex w-full flex-col gap-0 p-0 sm:max-w-2xl">
        {trigger ? (
          <>
            {/* Header */}
            <div className="border-b border-border bg-background px-6 py-5">
              <SheetHeader className="space-y-3">
                <div className="flex items-start justify-between gap-4 pr-8">
                  <div className="flex min-w-0 items-start gap-3">
                    <SourceIcon source={trigger.source_name} size="lg" />
                    <div className="min-w-0 space-y-1">
                      <SheetTitle className="flex flex-wrap items-center gap-2 text-base font-semibold">
                        <span className="truncate lowercase">
                          {trigger.source_name}
                        </span>
                        <CopyIdentifierChip
                          value={trigger.id}
                          tooltip="Copy trigger ID"
                          idTailVisible={6}
                        />
                      </SheetTitle>
                      <SheetDescription className="truncate font-mono text-xs">
                        <span className="text-muted-foreground">
                          {trigger.target_node_id}
                        </span>
                        <span className="text-muted-foreground/60">.</span>
                        <span className="font-medium text-foreground">
                          {trigger.target_reasoner}
                        </span>
                      </SheetDescription>
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Switch
                      checked={trigger.enabled}
                      disabled={busy}
                      onCheckedChange={onEnabledChange}
                      aria-label={trigger.enabled ? "Disable trigger" : "Enable trigger"}
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      disabled={isCodeManaged || busy}
                      onClick={onDelete}
                      title={
                        isCodeManaged
                          ? "Code-managed triggers must be deleted from agent code"
                          : "Delete trigger"
                      }
                    >
                      <Trash className="size-4" />
                    </Button>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2 pt-1">
                  <Badge
                    variant={trigger.enabled ? "secondary" : "outline"}
                    size="sm"
                    showIcon={false}
                  >
                    {trigger.enabled ? "Enabled" : "Paused"}
                  </Badge>
                  <Badge variant="outline" size="sm" showIcon={false}>
                    {isCodeManaged ? "Code-managed" : "UI-managed"}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {eventTypeLabel}
                  </span>
                  <span className="text-xs text-muted-foreground">·</span>
                  <span className="text-xs text-muted-foreground">
                    Updated {formatDate(trigger.updated_at)}
                  </span>
                </div>
              </SheetHeader>
            </div>

            {!trigger.enabled ? (
              <div className="border-b border-border bg-muted px-6 py-2">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <PauseFilled className="size-3.5" />
                  This trigger is paused. Inbound events are accepted and
                  recorded, but not dispatched.
                </div>
              </div>
            ) : null}

            {/* Tabs — flush full width */}
            <AnimatedTabs
              defaultValue="events"
              className="flex min-h-0 flex-1 flex-col"
            >
              <div className="border-b border-border bg-background px-6 py-3">
                <AnimatedTabsList className="h-10 w-full justify-start gap-1 rounded-md bg-muted/40 p-1">
                  <AnimatedTabsTrigger
                    value="events"
                    className="gap-1.5 px-3 text-xs"
                  >
                    <Activity className="size-3.5" />
                    Events
                  </AnimatedTabsTrigger>
                  <AnimatedTabsTrigger
                    value="configuration"
                    className="gap-1.5 px-3 text-xs"
                  >
                    <SlidersHorizontal className="size-3.5" />
                    Configuration
                  </AnimatedTabsTrigger>
                  <AnimatedTabsTrigger
                    value="secrets"
                    className="gap-1.5 px-3 text-xs"
                  >
                    <Lock className="size-3.5" />
                    Secrets
                  </AnimatedTabsTrigger>
                  <AnimatedTabsTrigger
                    value="dispatch"
                    className="gap-1.5 px-3 text-xs"
                  >
                    <ArrowLeftRight className="size-3.5" />
                    Dispatches
                  </AnimatedTabsTrigger>
                </AnimatedTabsList>
              </div>

              <ScrollArea className="min-h-0 flex-1">
                <div className="space-y-6 p-6">
                  <AnimatedTabsContent value="events" className="mt-0 space-y-5">
                    <DetailSection title="Public ingest URL">
                      <div className="flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2">
                        <span className="min-w-0 flex-1 truncate font-mono text-xs">
                          {publicUrl}
                        </span>
                        <CopyButton
                          value={publicUrl}
                          size="icon-sm"
                          tooltip="Copy public URL"
                        />
                      </div>
                    </DetailSection>

                    <Separator />

                    <div className="flex items-center justify-between gap-3">
                      <div className="space-y-0.5">
                        <h3 className="text-sm font-medium">Recent events</h3>
                        <p className="text-xs text-muted-foreground">
                          Stored events received for this trigger.
                        </p>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => void refreshEvents()}
                        disabled={loadingEvents}
                      >
                        Refresh
                      </Button>
                    </div>
                    {eventError ? (
                      <Alert variant="destructive">
                        <AlertDescription>{eventError}</AlertDescription>
                      </Alert>
                    ) : null}
                    {loadingEvents ? (
                      <div className="text-sm text-muted-foreground">
                        Loading…
                      </div>
                    ) : events.length === 0 ? (
                      <div className="rounded-md border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
                        No events received yet. Events appear here after the first inbound delivery.
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {events.map((event) => (
                          <EventRow
                            key={event.id}
                            event={event}
                            triggerID={triggerId ?? ""}
                            onReplayed={() => void refreshEvents()}
                          />
                        ))}
                      </div>
                    )}
                  </AnimatedTabsContent>

                  <AnimatedTabsContent
                    value="configuration"
                    className="mt-0 space-y-5"
                  >
                    <DetailSection title="Target">
                      <RowList
                        rows={[
                          ["Node", trigger.target_node_id],
                          ["Reasoner", trigger.target_reasoner],
                          ["Event types", eventTypeLabel],
                        ]}
                      />
                    </DetailSection>
                    <Separator />
                    <DetailSection title="Lifecycle">
                      <RowList
                        rows={[
                          ["Created", formatDate(trigger.created_at)],
                          ["Updated", formatDate(trigger.updated_at)],
                        ]}
                      />
                    </DetailSection>
                    <Separator />
                    <DetailSection title="Config JSON">
                      <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs text-foreground">
                        {formatConfig(trigger.config)}
                      </pre>
                    </DetailSection>
                  </AnimatedTabsContent>

                  <AnimatedTabsContent
                    value="secrets"
                    className="mt-0 space-y-5"
                  >
                    <DetailSection title="Signature secret">
                      <RowList
                        rows={[
                          ["Env var", trigger.secret_env_var || "Not configured"],
                          ["Source", trigger.source_name],
                        ]}
                      />
                    </DetailSection>
                    <div className="rounded-md border border-border bg-muted/30 p-4 text-xs text-muted-foreground">
                      Secret values stay on the control plane and are referenced
                      by env var name only. Rotate by updating the env var on
                      the host — no UI restart required.
                    </div>
                  </AnimatedTabsContent>

                  <AnimatedTabsContent
                    value="dispatch"
                    className="mt-0 space-y-5"
                  >
                    <DetailSection title="Dispatch target">
                      <dl className="grid gap-3 sm:grid-cols-2">
                        <div className="grid gap-1">
                          <dt className="text-xs text-muted-foreground">
                            Target
                          </dt>
                          <dd>
                            <button
                              type="button"
                              onClick={() => {
                                navigate(
                                  `/runs?search=${encodeURIComponent(trigger.target_reasoner)}`,
                                );
                              }}
                              className="group inline-flex max-w-full items-center gap-1.5 rounded-md border border-border/60 bg-muted/30 px-2 py-1 font-mono text-xs transition-colors hover:border-border hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                              title={`View runs for ${trigger.target_reasoner}`}
                            >
                              <span className="min-w-0 truncate">
                                <span className="text-muted-foreground">
                                  {trigger.target_node_id}
                                </span>
                                <span className="text-muted-foreground/60">.</span>
                                <span className="font-medium text-foreground">
                                  {trigger.target_reasoner}
                                </span>
                              </span>
                              <ArrowUpRight
                                className="size-3 shrink-0 text-muted-foreground transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
                                aria-hidden
                              />
                            </button>
                          </dd>
                        </div>
                        <div className="grid gap-1">
                          <dt className="text-xs text-muted-foreground">
                            Last update
                          </dt>
                          <dd className="font-mono text-xs text-foreground">
                            {formatDate(trigger.updated_at)}
                          </dd>
                        </div>
                      </dl>
                    </DetailSection>
                    <div className="rounded-md border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
                      Click the target above to jump to runs originating from
                      this reasoner. Per-trigger dispatch log rows appear here
                      once the API surfaces them.
                    </div>
                  </AnimatedTabsContent>
                </div>
              </ScrollArea>
            </AnimatedTabs>
          </>
        ) : null}
      </SheetContent>
    </Sheet>
  );
}

function DetailSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="space-y-2.5">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </h3>
      {children}
    </section>
  );
}

function RowList({ rows }: { rows: Array<[string, string]> }) {
  return (
    <dl className="grid gap-3 sm:grid-cols-2">
      {rows.map(([label, value]) => (
        <div key={label} className="grid gap-1">
          <dt className="text-xs text-muted-foreground">{label}</dt>
          <dd className="break-all font-mono text-xs text-foreground">
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}
