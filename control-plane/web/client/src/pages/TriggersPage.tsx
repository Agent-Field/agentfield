import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { CopyIdentifierChip } from "@/components/ui/copy-identifier-chip";
import { Sparkline } from "@/components/ui/Sparkline";
import { formatCompactRelativeTime } from "@/utils/dateFormat";
import {
  ArrowRight,
  Copy,
  MoreHorizontal,
  Plug,
  Plus,
  Search,
  Trash,
  Webhook,
} from "@/components/ui/icon-bridge";
import { PageHeader } from "@/components/PageHeader";
import { NewTriggerDialog } from "@/components/triggers/NewTriggerDialog";
import { SourceIcon } from "@/components/triggers/SourceIcon";
import { TriggerSheet } from "@/components/triggers/TriggerSheet";
import { cn } from "@/lib/utils";

export interface SourceCatalogEntry {
  name: string;
  kind: "http" | "loop" | string;
  secret_required: boolean;
  config_schema: Record<string, unknown>;
}

export interface Trigger {
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
  // Per-trigger 24h activity (populated by the backend on the list response).
  // Optional so older API responses degrade gracefully — UI shows "—".
  event_count_24h?: number;
  dispatch_success_24h?: number;
  dispatch_failed_24h?: number;
  last_event_at?: string | null;
  /** 24-element histogram, index 0 = oldest hour, index 23 = current hour. */
  dispatch_buckets_24h?: number[];
}

const serverUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(
    "/api/ui/v1",
    "",
  ) || window.location.origin;

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

function publicIngestUrl(triggerId: string): string {
  return `${serverUrl}/sources/${triggerId}`;
}

function shortTriggerId(id: string): string {
  if (id.length <= 8) return id;
  // Pull the last 6 chars — same shape as RunsPage's run_id chip.
  return `…${id.slice(-6)}`;
}

function summarizeEventTypes(types: string[] | null | undefined): {
  primary: string;
  overflow: number;
} {
  if (!types || types.length === 0) return { primary: "All events", overflow: 0 };
  return { primary: types[0], overflow: Math.max(0, types.length - 1) };
}

function ownerLabel(managed: Trigger["managed_by"]): string {
  return managed === "code" ? "Code" : "UI";
}

/**
 * "Last 24h" cell — failure-first, single-line.
 *
 * Operator's daily question is "did anything fail?" so we lead with the
 * failed count when > 0 (status-error tone) and stay quiet otherwise:
 *   - never fired: em-dash
 *   - clean: "2m ago" + tiny muted sparkline
 *   - failures: "N failed · 2m ago" + tiny muted sparkline
 *
 * The sparkline never carries status tone — it stays muted-foreground via
 * currentColor regardless of failures. The single status signal is the
 * "N failed" text.
 */
function TriggerActivityCell({ trigger }: { trigger: Trigger }) {
  const total = trigger.event_count_24h ?? 0;
  const failed = trigger.dispatch_failed_24h ?? 0;
  const buckets = trigger.dispatch_buckets_24h ?? [];
  const lastEventAt = trigger.last_event_at ?? null;

  if (total === 0 && !lastEventAt) {
    return (
      <span
        className="text-xs text-muted-foreground/70"
        aria-label="No activity in the last 24 hours"
      >
        —
      </span>
    );
  }

  const lastFired = lastEventAt
    ? formatCompactRelativeTime(lastEventAt)
    : null;

  return (
    <div
      className="flex min-w-0 items-center gap-2"
      title={`${total} events · ${failed} failed in the last 24h`}
    >
      {failed > 0 ? (
        <>
          <span className="font-mono text-xs tabular-nums text-status-error">
            {failed} failed
          </span>
          {lastFired ? (
            <>
              <span className="text-muted-foreground/60">·</span>
              <span className="text-xs text-muted-foreground">{lastFired}</span>
            </>
          ) : null}
        </>
      ) : (
        lastFired ? (
          <span className="text-xs text-muted-foreground">{lastFired}</span>
        ) : null
      )}
      <Sparkline
        data={buckets}
        width={48}
        height={14}
        showArea
        className="text-muted-foreground/60"
      />
    </div>
  );
}

interface TriggerRowProps {
  trigger: Trigger;
  selected: boolean;
  busy: boolean;
  onOpen: (trigger: Trigger) => void;
  onToggleEnabled: (trigger: Trigger, enabled: boolean) => void;
  onCopyUrl: (trigger: Trigger) => void;
  onDelete: (trigger: Trigger) => void;
}

function TriggerRow({
  trigger,
  selected,
  busy,
  onOpen,
  onToggleEnabled,
  onCopyUrl,
  onDelete,
}: TriggerRowProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const events = summarizeEventTypes(trigger.event_types);
  const isCodeManaged = trigger.managed_by === "code";

  return (
    <TableRow
      className="group/trigger-row cursor-pointer"
      data-state={selected ? "selected" : undefined}
      tabIndex={0}
      onClick={() => onOpen(trigger)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen(trigger);
        }
      }}
    >
      {/* Source */}
      <TableCell className="w-40">
        <div className="flex min-w-0 items-center gap-2.5">
          <SourceIcon source={trigger.source_name} size="compact" />
          <span className="truncate text-xs font-medium lowercase">
            {trigger.source_name}
          </span>
        </div>
      </TableCell>

      {/* Target — agent.reasoner muted/bold + trigger short-id chip */}
      <TableCell className="min-w-0 max-w-[min(36rem,60vw)]">
        <div className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-1">
          <span className="inline-block min-w-0 max-w-[min(100%,22rem)] truncate font-mono text-xs">
            <span className="text-muted-foreground">
              {trigger.target_node_id}
            </span>
            <span className="text-muted-foreground/60">.</span>
            <span className="font-medium text-foreground">
              {trigger.target_reasoner}
            </span>
          </span>
          <div onClick={(event) => event.stopPropagation()}>
            <CopyIdentifierChip
              value={trigger.id}
              tooltip="Copy trigger ID"
              idTailVisible={6}
            />
          </div>
        </div>
      </TableCell>

      {/* Events */}
      <TableCell className="w-44">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="truncate font-mono text-xs text-foreground/90">
            {events.primary}
          </span>
          {events.overflow > 0 ? (
            <Badge
              variant="secondary"
              size="sm"
              showIcon={false}
              className="shrink-0"
            >
              +{events.overflow}
            </Badge>
          ) : null}
        </div>
      </TableCell>

      {/* Last 24h */}
      <TableCell className="w-40">
        <TriggerActivityCell trigger={trigger} />
      </TableCell>

      {/* Owner */}
      <TableCell className="w-20">
        <Badge variant="outline" size="sm" showIcon={false}>
          {ownerLabel(trigger.managed_by)}
        </Badge>
      </TableCell>

      {/* Enabled */}
      <TableCell
        className="w-20"
        onClick={(event) => event.stopPropagation()}
      >
        <Switch
          checked={trigger.enabled}
          disabled={busy}
          onCheckedChange={(enabled) => onToggleEnabled(trigger, enabled)}
        />
      </TableCell>

      {/* Kebab */}
      <TableCell
        className="w-10 px-2 text-right"
        onClick={(event) => event.stopPropagation()}
      >
        <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "size-8 shrink-0 text-muted-foreground/70 transition-colors",
                "group-hover/trigger-row:text-foreground",
                "hover:bg-muted hover:text-foreground",
                "data-[state=open]:bg-muted data-[state=open]:text-foreground",
              )}
              aria-label={`Trigger actions for ${trigger.id}`}
            >
              <MoreHorizontal className="size-3.5" aria-hidden />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            side="bottom"
            className="w-44"
            onClick={(event) => event.stopPropagation()}
          >
            <DropdownMenuLabel className="text-xs font-normal text-muted-foreground">
              Trigger
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="gap-2 text-xs"
              onClick={() => {
                setMenuOpen(false);
                onCopyUrl(trigger);
              }}
            >
              <Copy className="size-3.5" aria-hidden />
              Copy public URL
            </DropdownMenuItem>
            <DropdownMenuItem
              className="gap-2 text-xs"
              onClick={() => {
                setMenuOpen(false);
                onOpen(trigger);
              }}
            >
              <ArrowRight className="size-3.5" aria-hidden />
              View events
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className={cn(
                "gap-2 text-xs",
                isCodeManaged
                  ? "pointer-events-none text-muted-foreground/60"
                  : "text-destructive focus:text-destructive",
              )}
              disabled={isCodeManaged}
              onClick={() => {
                setMenuOpen(false);
                if (!isCodeManaged) onDelete(trigger);
              }}
            >
              <Trash className="size-3.5" aria-hidden />
              {isCodeManaged ? "Delete (in code)" : "Delete trigger"}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </TableCell>
    </TableRow>
  );
}

export function TriggersPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedTriggerId = searchParams.get("trigger");
  const sourceFilter = searchParams.get("source") ?? "all";
  const ownerFilter = searchParams.get("owner") ?? "all";
  const enabledFilter = searchParams.get("enabled") ?? "all";

  const [sources, setSources] = useState<SourceCatalogEntry[]>([]);
  const [triggers, setTriggers] = useState<Trigger[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [busyTriggerId, setBusyTriggerId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<Trigger | null>(null);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const [s, t] = await Promise.all([
        fetchJson<{ sources: SourceCatalogEntry[] }>(
          `${serverUrl}/api/v1/sources`,
        ),
        fetchJson<{ triggers: Trigger[] }>(`${serverUrl}/api/v1/triggers`),
      ]);
      setSources(s.sources || []);
      setTriggers(t.triggers || []);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const selectedTrigger = useMemo(
    () => triggers.find((trigger) => trigger.id === selectedTriggerId) ?? null,
    [selectedTriggerId, triggers],
  );

  const sourceOptions = useMemo(() => {
    const names = new Set<string>(sources.map((s) => s.name));
    for (const t of triggers) names.add(t.source_name);
    return Array.from(names).sort();
  }, [sources, triggers]);

  const visibleTriggers = useMemo(() => {
    const query = search.trim().toLowerCase();
    return triggers
      .filter((trigger) => {
        if (sourceFilter !== "all" && trigger.source_name !== sourceFilter)
          return false;
        if (ownerFilter !== "all" && trigger.managed_by !== ownerFilter)
          return false;
        if (enabledFilter === "enabled" && !trigger.enabled) return false;
        if (enabledFilter === "paused" && trigger.enabled) return false;
        if (!query) return true;
        return (
          trigger.id.toLowerCase().includes(query) ||
          trigger.source_name.toLowerCase().includes(query) ||
          trigger.target_node_id.toLowerCase().includes(query) ||
          trigger.target_reasoner.toLowerCase().includes(query) ||
          (trigger.event_types ?? []).some((evt) =>
            evt.toLowerCase().includes(query),
          )
        );
      })
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  }, [enabledFilter, ownerFilter, search, sourceFilter, triggers]);

  function setQuery(key: string, value: string) {
    const next = new URLSearchParams(searchParams);
    if (value === "all" || value === "") {
      next.delete(key);
    } else {
      next.set(key, value);
    }
    setSearchParams(next, { replace: true });
  }

  function openTrigger(trigger: Trigger) {
    const next = new URLSearchParams(searchParams);
    next.set("trigger", trigger.id);
    setSearchParams(next, { replace: false });
  }

  function closeTrigger() {
    const next = new URLSearchParams(searchParams);
    next.delete("trigger");
    setSearchParams(next, { replace: false });
  }

  async function updateTrigger(triggerId: string, patch: Partial<Trigger>) {
    try {
      setBusyTriggerId(triggerId);
      await fetchJson(`${serverUrl}/api/v1/triggers/${triggerId}`, {
        method: "PUT",
        body: JSON.stringify(patch),
      });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyTriggerId(null);
    }
  }

  async function deleteTrigger(trigger: Trigger) {
    if (trigger.managed_by === "code") return;
    try {
      setBusyTriggerId(trigger.id);
      await fetchJson(`${serverUrl}/api/v1/triggers/${trigger.id}`, {
        method: "DELETE",
      });
      closeTrigger();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyTriggerId(null);
      setConfirmDelete(null);
    }
  }

  function copyTriggerUrl(trigger: Trigger) {
    void navigator.clipboard.writeText(publicIngestUrl(trigger.id));
  }

  const totalCount = triggers.length;
  const visibleCount = visibleTriggers.length;
  const hasFilters =
    sourceFilter !== "all" ||
    ownerFilter !== "all" ||
    enabledFilter !== "all" ||
    search.trim() !== "";

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-hidden">
      <PageHeader
        title="Active triggers"
        description="Inbound wirings dispatching provider events, schedules, and webhooks into reasoners."
        actions={[
          {
            label: "New trigger",
            onClick: () => setCreateOpen(true),
            variant: "default",
            icon: <Plus className="size-4" />,
          },
        ]}
      />

      {error ? (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      <div className="flex flex-col gap-3 md:flex-row md:flex-wrap md:items-center md:justify-between">
        <div className="flex flex-wrap items-center gap-3">
          <Select
            value={sourceFilter}
            onValueChange={(v) => setQuery("source", v)}
          >
            <SelectTrigger className="h-9 w-44 text-xs">
              <SelectValue placeholder="All sources" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All sources</SelectItem>
              {sourceOptions.map((name) => (
                <SelectItem key={name} value={name}>
                  {name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select
            value={ownerFilter}
            onValueChange={(v) => setQuery("owner", v)}
          >
            <SelectTrigger className="h-9 w-36 text-xs">
              <SelectValue placeholder="All owners" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All owners</SelectItem>
              <SelectItem value="code">code</SelectItem>
              <SelectItem value="ui">ui</SelectItem>
            </SelectContent>
          </Select>
          <Select
            value={enabledFilter}
            onValueChange={(v) => setQuery("enabled", v)}
          >
            <SelectTrigger className="h-9 w-36 text-xs">
              <SelectValue placeholder="All states" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All states</SelectItem>
              <SelectItem value="enabled">Enabled</SelectItem>
              <SelectItem value="paused">Paused</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground">
            {hasFilters
              ? `${visibleCount} of ${totalCount}`
              : `${totalCount} total`}
          </span>
          <div className="relative w-full md:w-64">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search triggers…"
              className="pl-8"
            />
          </div>
        </div>
      </div>

      <Card>
        <CardContent className="p-0">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead className="h-8 w-40 px-3 text-micro-plus font-medium uppercase tracking-wider text-muted-foreground/85">
                  Source
                </TableHead>
                <TableHead className="h-8 px-3 text-micro-plus font-medium uppercase tracking-wider text-muted-foreground/85">
                  Target
                </TableHead>
                <TableHead className="h-8 w-44 px-3 text-micro-plus font-medium uppercase tracking-wider text-muted-foreground/85">
                  Events
                </TableHead>
                <TableHead className="h-8 w-40 px-3 text-micro-plus font-medium uppercase tracking-wider text-muted-foreground/85">
                  Last 24h
                </TableHead>
                <TableHead className="h-8 w-20 px-3 text-micro-plus font-medium uppercase tracking-wider text-muted-foreground/85">
                  Owner
                </TableHead>
                <TableHead className="h-8 w-20 px-3 text-micro-plus font-medium uppercase tracking-wider text-muted-foreground/85">
                  Enabled
                </TableHead>
                <TableHead className="h-8 w-10 px-2" aria-label="Actions" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell
                    colSpan={7}
                    className="p-8 text-center text-xs text-muted-foreground"
                  >
                    Loading triggers…
                  </TableCell>
                </TableRow>
              ) : visibleCount === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="p-8">
                    <div className="flex flex-col items-center justify-center gap-2 text-center">
                      <Webhook
                        className="size-8 text-muted-foreground"
                        aria-hidden
                      />
                      <p className="text-sm font-medium">
                        {hasFilters
                          ? "No triggers match your filters"
                          : "No triggers yet"}
                      </p>
                      <p className="max-w-md text-xs text-muted-foreground">
                        {hasFilters
                          ? "Clear the filter or search to see all triggers."
                          : "Connect an external service from Integrations, or declare a trigger in agent code."}
                      </p>
                      {!hasFilters ? (
                        <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
                          <Button
                            size="sm"
                            onClick={() => navigate("/integrations")}
                          >
                            <Plug className="size-3.5" aria-hidden />
                            Browse integrations
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setCreateOpen(true)}
                          >
                            <Plus className="size-3.5" aria-hidden />
                            New trigger
                          </Button>
                        </div>
                      ) : null}
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                visibleTriggers.map((trigger) => (
                  <TriggerRow
                    key={trigger.id}
                    trigger={trigger}
                    selected={selectedTriggerId === trigger.id}
                    busy={busyTriggerId === trigger.id}
                    onOpen={openTrigger}
                    onToggleEnabled={(t, enabled) =>
                      void updateTrigger(t.id, { enabled })
                    }
                    onCopyUrl={copyTriggerUrl}
                    onDelete={(t) => setConfirmDelete(t)}
                  />
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <NewTriggerDialog
        open={createOpen}
        sources={sources}
        onOpenChange={setCreateOpen}
        onCreated={() => {
          setCreateOpen(false);
          void refresh();
        }}
      />

      <TriggerSheet
        open={Boolean(selectedTrigger)}
        trigger={selectedTrigger}
        serverUrl={serverUrl}
        publicUrl={selectedTrigger ? publicIngestUrl(selectedTrigger.id) : ""}
        busy={selectedTrigger ? busyTriggerId === selectedTrigger.id : false}
        onOpenChange={(open) => {
          if (!open) closeTrigger();
        }}
        onEnabledChange={(enabled) => {
          if (selectedTrigger)
            void updateTrigger(selectedTrigger.id, { enabled });
        }}
        onDelete={() => {
          if (selectedTrigger) setConfirmDelete(selectedTrigger);
        }}
      />

      <AlertDialog
        open={Boolean(confirmDelete)}
        onOpenChange={(open) => {
          if (!open) setConfirmDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete trigger {shortTriggerId(confirmDelete?.id ?? "")}?
            </AlertDialogTitle>
            <AlertDialogDescription>
              The public ingest URL will stop accepting events immediately. Any
              in-flight dispatches will finish, then the trigger row is removed.
              Code-managed triggers must be deleted from agent source.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                if (confirmDelete) void deleteTrigger(confirmDelete);
              }}
            >
              Delete trigger
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export default TriggersPage;
