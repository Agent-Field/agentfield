import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ArrowRight,
  Plug,
  Search,
} from "@/components/ui/icon-bridge";
import { PageHeader } from "@/components/PageHeader";
import { SourceIcon } from "@/components/triggers/SourceIcon";
import { NewTriggerDialog } from "@/components/triggers/NewTriggerDialog";
import { cn } from "@/lib/utils";

interface SourceCatalogEntry {
  name: string;
  kind: "http" | "loop" | string;
  secret_required: boolean;
  config_schema: Record<string, unknown>;
}

interface Trigger {
  id: string;
  source_name: string;
  enabled: boolean;
}

type SourceCategory = "Provider" | "Schedule" | "Generic";

interface SourceMeta {
  display: string;
  category: SourceCategory;
  description: string;
  /** Sample event types operators recognize at a glance. */
  highlights: string[];
}

const SOURCE_META: Record<string, SourceMeta> = {
  stripe: {
    display: "Stripe",
    category: "Provider",
    description:
      "Payments, subscriptions, and customer events signed with Stripe-Signature HMAC.",
    highlights: ["payment_intent.*", "checkout.session.*", "invoice.*"],
  },
  github: {
    display: "GitHub",
    category: "Provider",
    description:
      "Repository, pull request, and workflow events signed with X-Hub-Signature-256.",
    highlights: ["pull_request", "push", "issues", "workflow_run"],
  },
  slack: {
    display: "Slack",
    category: "Provider",
    description:
      "Workspace events and slash commands signed with Slack signing secret.",
    highlights: ["event_callback", "url_verification"],
  },
  cron: {
    display: "Cron schedule",
    category: "Schedule",
    description:
      "Run a reasoner on a recurring cron schedule. No inbound auth — schedules are server-side.",
    highlights: ["* * * * *"],
  },
  generic_hmac: {
    display: "Generic HMAC",
    category: "Generic",
    description:
      "Webhook with a custom HMAC signing scheme. Bring your own header and digest.",
    highlights: ["custom HMAC"],
  },
  generic_bearer: {
    display: "Generic Bearer",
    category: "Generic",
    description:
      "Webhook authenticated with a static bearer token in the Authorization header.",
    highlights: ["Bearer token"],
  },
};

const CATEGORY_FILTERS = [
  { label: "All", value: "all" },
  { label: "Providers", value: "Provider" },
  { label: "Schedules", value: "Schedule" },
  { label: "Generic", value: "Generic" },
] as const;

const CATEGORY_RANK: Record<SourceCategory, number> = {
  Provider: 0,
  Schedule: 1,
  Generic: 2,
};

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

function metaFor(source: SourceCatalogEntry): SourceMeta {
  return (
    SOURCE_META[source.name] ?? {
      display: source.name,
      category: source.kind === "loop" ? "Schedule" : "Generic",
      description: `${
        source.kind === "loop" ? "Schedule" : "Webhook"
      } source plugin.`,
      highlights: [],
    }
  );
}

export function IntegrationsPage() {
  const navigate = useNavigate();
  const [sources, setSources] = useState<SourceCatalogEntry[]>([]);
  const [triggers, setTriggers] = useState<Trigger[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<(typeof CATEGORY_FILTERS)[number]["value"]>(
    "all",
  );
  const [dialogOpen, setDialogOpen] = useState(false);
  const [defaultSource, setDefaultSource] = useState<string | undefined>(undefined);

  const load = useCallback(async () => {
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
    void load();
  }, [load]);

  const activeBySource = useMemo(() => {
    const counts = new Map<string, number>();
    for (const trigger of triggers) {
      counts.set(trigger.source_name, (counts.get(trigger.source_name) ?? 0) + 1);
    }
    return counts;
  }, [triggers]);

  const visible = useMemo(() => {
    const query = search.trim().toLowerCase();
    return sources
      .map((source) => ({ source, meta: metaFor(source) }))
      .filter(({ source, meta }) => {
        if (filter !== "all" && meta.category !== filter) return false;
        if (!query) return true;
        return (
          source.name.toLowerCase().includes(query) ||
          meta.display.toLowerCase().includes(query) ||
          meta.description.toLowerCase().includes(query)
        );
      })
      .sort((a, b) => {
        const cat = CATEGORY_RANK[a.meta.category] - CATEGORY_RANK[b.meta.category];
        if (cat !== 0) return cat;
        return a.meta.display.localeCompare(b.meta.display);
      });
  }, [filter, search, sources]);

  function openConnect(sourceName: string) {
    setDefaultSource(sourceName);
    setDialogOpen(true);
  }

  function openManage(sourceName: string) {
    navigate(`/triggers?source=${encodeURIComponent(sourceName)}`);
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-hidden">
      <PageHeader
        title="Integrations"
        description="Connect external services, schedules, and webhooks to your reasoners."
        actions={[
          {
            label: "New trigger",
            onClick: () => {
              setDefaultSource(undefined);
              setDialogOpen(true);
            },
            variant: "default",
          },
        ]}
      />

      {error ? (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="relative w-full md:max-w-sm">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search integrations…"
            className="pl-8"
          />
        </div>
        <div className="inline-flex items-center gap-1 rounded-md border border-border bg-muted/40 p-1">
          {CATEGORY_FILTERS.map((f) => (
            <button
              key={f.value}
              type="button"
              onClick={() => setFilter(f.value)}
              className={cn(
                "rounded-sm px-3 py-1 text-xs font-medium transition-colors",
                filter === f.value
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-44 w-full rounded-lg" />
          ))}
        </div>
      ) : visible.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center gap-2 py-12 text-center">
            <Plug className="size-8 text-muted-foreground" aria-hidden />
            <p className="text-sm font-medium">No integrations match your search</p>
            <p className="max-w-sm text-xs text-muted-foreground">
              Try clearing the filter, or check that the control plane has source plugins compiled in.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {visible.map(({ source, meta }) => {
            const activeCount = activeBySource.get(source.name) ?? 0;
            return (
              <Card
                key={source.name}
                className="group flex h-full flex-col transition-shadow hover:shadow-md focus-within:shadow-md"
              >
                <CardHeader className="flex flex-row items-start gap-3 space-y-0">
                  <SourceIcon source={source.name} size="lg" />
                  <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <CardTitle className="truncate text-base font-semibold">
                        {meta.display}
                      </CardTitle>
                      <Badge
                        variant="outline"
                        size="sm"
                        showIcon={false}
                        className="shrink-0"
                      >
                        {meta.category}
                      </Badge>
                    </div>
                    <CardDescription className="text-xs">
                      {source.kind === "loop" ? "Schedule" : "Webhook"}
                      {source.secret_required ? " · signed" : " · no auth"}
                    </CardDescription>
                  </div>
                </CardHeader>
                <CardContent className="flex flex-1 flex-col gap-4">
                  <p className="text-sm text-muted-foreground">
                    {meta.description}
                  </p>
                  {meta.highlights.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {meta.highlights.map((h) => (
                        <Badge
                          key={h}
                          variant="secondary"
                          size="sm"
                          showIcon={false}
                          className="font-mono"
                        >
                          {h}
                        </Badge>
                      ))}
                    </div>
                  ) : null}
                  <div className="mt-auto flex items-center justify-between border-t border-border/60 pt-3">
                    <span className="flex items-center gap-2 text-xs">
                      <span
                        className={cn(
                          "size-2 rounded-full",
                          activeCount > 0
                            ? "bg-status-success"
                            : "bg-muted-foreground/40",
                        )}
                        aria-hidden
                      />
                      <span className="text-muted-foreground">
                        {activeCount === 0
                          ? "Not connected"
                          : `${activeCount} active`}
                      </span>
                    </span>
                    {activeCount > 0 ? (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openManage(source.name)}
                      >
                        Manage
                        <ArrowRight className="ml-1 size-3.5" aria-hidden />
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        onClick={() => openConnect(source.name)}
                      >
                        Connect
                        <ArrowRight className="ml-1 size-3.5" aria-hidden />
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      <NewTriggerDialog
        open={dialogOpen}
        sources={sources}
        defaultSourceName={defaultSource}
        onOpenChange={setDialogOpen}
        onCreated={() => {
          setDialogOpen(false);
          void load();
        }}
      />
    </div>
  );
}

export default IntegrationsPage;
