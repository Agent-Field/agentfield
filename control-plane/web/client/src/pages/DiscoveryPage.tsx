import { useEffect, useId, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  Braces,
  CheckCircle2,
  Copy,
  Database,
  ExternalLink,
  Loader2,
  Network,
  Plus,
  RefreshCw,
  Search,
  Share2,
  ShieldCheck,
  Trash,
} from "@/components/ui/icon-bridge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AutoExpandingTextarea } from "@/components/ui/auto-expanding-textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  getARDDashboard,
  importARDEntry,
  saveARDBinding,
  saveARDPublication,
  saveARDRegistries,
  searchExternalARD,
  updateARDSettings,
} from "@/services/ardApi";
import type {
  ARDDashboard,
  ARDExternalBinding,
  ARDPublicationView,
  ARDRegistryRecord,
  ARDRuntimeSettings,
  ARDSearchResult,
} from "@/types/ard";
import { cn } from "@/lib/utils";

const QUERY_KEY = ["ard-dashboard"];

function useARDDashboard() {
  return useQuery({
    queryKey: QUERY_KEY,
    queryFn: getARDDashboard,
    staleTime: 15_000,
  });
}

function useDashboardMutation<TArgs>(
  mutationFn: (args: TArgs) => Promise<ARDDashboard>
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn,
    onSuccess: (dashboard) => {
      queryClient.setQueryData(QUERY_KEY, dashboard);
    },
  });
}

export function DiscoveryPage() {
  const { data, isLoading, isError, error, refetch, isFetching } = useARDDashboard();
  const [searchParams] = useSearchParams();
  const defaultTab = searchParams.get("target") ? "catalog" : "overview";

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <div className="h-8 w-56 rounded-md bg-muted/60 animate-pulse" />
        <div className="grid gap-3 md:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="h-28 rounded-lg bg-muted/40 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-semibold">Discovery</h1>
        <Card>
          <CardContent className="p-6">
            <p className="text-sm text-destructive">
              Could not load ARD state{error instanceof Error ? `: ${error.message}` : ""}.
            </p>
            <Button className="mt-4" variant="outline" onClick={() => refetch()}>
              <RefreshCw className="size-4" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Network className="size-5 text-primary" />
            <h1 className="text-2xl font-semibold">Discovery</h1>
          </div>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            Publish AgentField reasoners and skills through ARD, discover external resources, and explicitly opt imported entries into callability.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge active={data.config.enabled} label={data.config.enabled ? "ARD available" : "ARD disabled"} />
          <StatusBadge active={data.summary.catalog_published} label={data.summary.catalog_published ? "Catalog route on" : "Catalog route off"} />
          <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
            <RefreshCw className={cn("size-4", isFetching && "animate-spin")} />
            Refresh
          </Button>
        </div>
      </header>

      <Tabs defaultValue={defaultTab} className="flex flex-col gap-4">
        <TabsList variant="underline" className="w-full justify-start overflow-x-auto">
          <TabsTrigger value="overview" variant="underline">Overview</TabsTrigger>
          <TabsTrigger value="catalog" variant="underline">Public Catalog</TabsTrigger>
          <TabsTrigger value="search" variant="underline">External Search</TabsTrigger>
          <TabsTrigger value="imports" variant="underline">Imports</TabsTrigger>
          <TabsTrigger value="registry" variant="underline">Registry</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-0">
          <OverviewTab dashboard={data} />
        </TabsContent>
        <TabsContent value="catalog" className="mt-0">
          <CatalogTab dashboard={data} />
        </TabsContent>
        <TabsContent value="search" className="mt-0">
          <ExternalSearchTab dashboard={data} />
        </TabsContent>
        <TabsContent value="imports" className="mt-0">
          <ImportsTab dashboard={data} />
        </TabsContent>
        <TabsContent value="registry" className="mt-0">
          <RegistryTab dashboard={data} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function OverviewTab({ dashboard }: { dashboard: ARDDashboard }) {
  const saveSettings = useDashboardMutation(updateARDSettings);
  const [settings, setSettings] = useState<ARDRuntimeSettings>(() => ({
    enabled: dashboard.state.settings.enabled ?? dashboard.config.enabled,
    publish_enabled: dashboard.state.settings.publish_enabled ?? dashboard.config.publish_enabled,
    registry_public: dashboard.state.settings.registry_public ?? dashboard.config.registry_public,
    public_base_url: dashboard.state.settings.public_base_url || dashboard.config.public_base_url,
    publisher_domain: dashboard.state.settings.publisher_domain || dashboard.config.publisher_domain,
    display_name: dashboard.state.settings.display_name || dashboard.config.display_name,
    documentation_url: dashboard.state.settings.documentation_url || dashboard.config.documentation_url,
    logo_url: dashboard.state.settings.logo_url || dashboard.config.logo_url,
  }));

  const metrics = [
    ["Published reasoners", dashboard.summary.published_reasoners],
    ["Published skills", dashboard.summary.published_skills],
    ["Imported resources", dashboard.summary.imported_resources],
    ["Callable external", dashboard.summary.callable_external_resources],
  ] as const;

  return (
    <div className="grid gap-4">
      <div className="grid gap-3 md:grid-cols-4">
        {metrics.map(([label, value]) => (
          <Card key={label} variant="surface" interactive={false}>
            <CardContent className="p-4">
              <div className="text-2xl font-semibold tabular-nums">{value}</div>
              <div className="mt-1 text-xs text-muted-foreground">{label}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(360px,0.7fr)]">
        <Card interactive={false}>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Global status</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3 p-4 pt-0 sm:grid-cols-2">
            <StatusRow label="ARD enabled" active={dashboard.summary.ard_enabled} />
            <StatusRow label="Catalog published" active={dashboard.summary.catalog_published} />
            <StatusRow label="Public URL configured" active={dashboard.summary.public_url_reachable} />
            <StatusRow label="DID/trust available" active={dashboard.summary.did_available} />
            <div className="sm:col-span-2 flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/20 p-3">
              <code className="min-w-0 flex-1 truncate text-xs">{dashboard.summary.catalog_url}</code>
              <Button variant="outline" size="sm" onClick={() => copyText(dashboard.summary.catalog_url)}>
                <Copy className="size-4" />
                Copy
              </Button>
              <Button variant="outline" size="sm" onClick={() => window.open(dashboard.summary.catalog_url, "_blank")}>
                <ExternalLink className="size-4" />
                Open
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card interactive={false}>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Deployment config</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3 p-4 pt-0">
            <ToggleRow
              label="ARD enabled"
              checked={Boolean(settings.enabled)}
              disabled={dashboard.config.locked.enabled}
              onCheckedChange={(enabled) => setSettings((current) => ({ ...current, enabled }))}
            />
            <ToggleRow
              label="Publish catalog"
              checked={Boolean(settings.publish_enabled)}
              disabled={dashboard.config.locked.publish_enabled}
              onCheckedChange={(publish_enabled) => setSettings((current) => ({ ...current, publish_enabled }))}
            />
            <ToggleRow
              label="Public registry"
              checked={Boolean(settings.registry_public)}
              disabled={dashboard.config.locked.registry_public}
              onCheckedChange={(registry_public) => setSettings((current) => ({ ...current, registry_public }))}
            />
            <SettingsInput
              label="Public base URL"
              value={settings.public_base_url || ""}
              locked={dashboard.config.locked.public_base_url}
              onChange={(public_base_url) => setSettings((current) => ({ ...current, public_base_url }))}
            />
            <SettingsInput
              label="Publisher domain"
              value={settings.publisher_domain || ""}
              locked={dashboard.config.locked.publisher_domain}
              onChange={(publisher_domain) => setSettings((current) => ({ ...current, publisher_domain }))}
            />
            <SettingsInput
              label="Display name"
              value={settings.display_name || ""}
              locked={dashboard.config.locked.display_name}
              onChange={(display_name) => setSettings((current) => ({ ...current, display_name }))}
            />
            <Button
              className="justify-self-start"
              onClick={() => saveSettings.mutate(settings)}
              disabled={saveSettings.isPending}
            >
              <CheckCircle2 className="size-4" />
              Save discovery settings
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function CatalogTab({ dashboard }: { dashboard: ARDDashboard }) {
  const [searchParams] = useSearchParams();
  const targetKey = searchParams.get("target");
  const initialSelection = useMemo(
    () => dashboard.publications.find((item) => item.key === targetKey) ?? dashboard.publications[0] ?? null,
    [dashboard.publications, targetKey]
  );
  const [selected, setSelected] = useState<ARDPublicationView | null>(
    initialSelection
  );
  useEffect(() => {
    if (targetKey) {
      setSelected(initialSelection);
      return;
    }
    setSelected((current) => dashboard.publications.find((item) => item.key === current?.key) ?? initialSelection);
  }, [dashboard.publications, initialSelection, targetKey]);

  if (dashboard.publications.length === 0) {
    return (
      <Card interactive={false}>
        <CardContent className="grid gap-2 p-6 text-sm text-muted-foreground">
          <div className="font-medium text-foreground">No publishable resources yet</div>
          <p>
            Register an agent node with reasoners or skills, then return here to opt each resource into ARD exposure.
          </p>
        </CardContent>
      </Card>
    );
  }

  const selectPublication = (publication: ARDPublicationView) => {
    setSelected(publication);
  };

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(360px,500px)]">
      <Card interactive={false} className="min-w-0">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Published internal resources</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[620px] table-fixed text-sm">
              <thead className="border-y border-border bg-muted/30 text-xs text-muted-foreground">
                <tr>
                  <th className="w-[34%] px-4 py-2 text-left font-medium">Name</th>
                  <th className="w-[12%] px-3 py-2 text-left font-medium">Type</th>
                  <th className="w-[12%] px-3 py-2 text-left font-medium">Node</th>
                  <th className="w-[10%] px-3 py-2 text-left font-medium">Trust</th>
                  <th className="w-[16%] px-3 py-2 text-left font-medium">Validation</th>
                  <th className="w-[16%] px-3 py-2 text-left font-medium">Updated</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {dashboard.publications.map((publication) => (
                  <tr
                    key={publication.key}
                    className={cn("cursor-pointer hover:bg-muted/30", selected?.key === publication.key && "bg-muted/40")}
                    onClick={() => selectPublication(publication)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        selectPublication(publication);
                      }
                    }}
                    tabIndex={0}
                    role="button"
                    aria-selected={selected?.key === publication.key}
                  >
                    <td className="px-4 py-3">
                      <div className="font-medium">{publication.display_name}</div>
                      <div className="truncate text-xs text-muted-foreground">{publication.description}</div>
                    </td>
                    <td className="px-3 py-3">
                      <Badge variant="metadata" showIcon={false}>{publication.target_kind}</Badge>
                    </td>
                    <td className="truncate px-3 py-3 font-mono text-xs">{publication.node_id}</td>
                    <td className="truncate px-3 py-3">{publication.entry.trustManifest?.identityType || "none"}</td>
                    <td className="px-3 py-3">
                      <PublicationBadge publication={publication} />
                    </td>
                    <td className="truncate px-3 py-3 text-xs text-muted-foreground">
                      {formatDate(publication.updated_at || publication.last_validated_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <div className="grid min-w-0 gap-4">
        {selected && <PublicationEditor key={selected.key} publication={selected} />}
        <Card interactive={false} className="min-w-0 overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Braces className="size-4" />
              ai-catalog.json
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            <pre className="max-h-[460px] max-w-full overflow-auto rounded-md border border-border bg-muted/30 p-3 text-xs leading-5">
              {JSON.stringify(dashboard.catalog, null, 2)}
            </pre>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function PublicationEditor({ publication }: { publication: ARDPublicationView }) {
  const savePublication = useDashboardMutation(saveARDPublication);
  const [draft, setDraft] = useState<ARDPublicationView>(publication);
  const descriptionId = useId();
  const queriesId = useId();
  useEffect(() => {
    setDraft(publication);
  }, [publication]);

  return (
    <Card interactive={false} className="min-w-0 overflow-hidden">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between gap-2 text-base">
          <span>Exposure</span>
          <PublicationBadge publication={publication} />
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3 p-4 pt-0">
        <div className="grid gap-2 rounded-md border border-border bg-muted/20 p-3 text-xs">
          <div className="flex items-center justify-between gap-3">
            <span className="text-muted-foreground">AgentField target</span>
            <code className="truncate font-mono">{draft.agentfield.invocationTarget}</code>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            <div>
              <div className="text-muted-foreground">Target ID</div>
              <code className="font-mono">{draft.agentfield.targetId}</code>
            </div>
            <div>
              <div className="text-muted-foreground">Health</div>
              <span>{draft.agentfield.healthStatus || "unknown"}</span>
            </div>
            <div>
              <div className="text-muted-foreground">Version</div>
              <span>{draft.agentfield.version || "unknown"}</span>
            </div>
          </div>
        </div>
        <ToggleRow
          label="Expose through ARD"
          checked={draft.published}
          onCheckedChange={(published) => setDraft((current) => ({ ...current, published }))}
        />
        <SettingsInput
          label="Display name"
          value={draft.display_name}
          onChange={(display_name) => setDraft((current) => ({ ...current, display_name }))}
        />
        <div className="grid gap-1.5">
          <Label htmlFor={descriptionId}>Description</Label>
          <AutoExpandingTextarea
            id={descriptionId}
            value={draft.description}
            onChange={(event) => setDraft((current) => ({ ...current, description: event.target.value }))}
            maxHeight={160}
          />
        </div>
        <SettingsInput
          label="Tags"
          value={(draft.tags || []).join(", ")}
          onChange={(value) => setDraft((current) => ({ ...current, tags: splitList(value) }))}
        />
        <div className="grid gap-1.5">
          <Label htmlFor={queriesId}>Representative queries</Label>
          <AutoExpandingTextarea
            id={queriesId}
            value={(draft.representative_queries || []).join("\n")}
            onChange={(event) => setDraft((current) => ({ ...current, representative_queries: splitLines(event.target.value) }))}
            maxHeight={120}
          />
        </div>
        <div className="grid gap-1.5">
          <Label>Artifact type</Label>
          <Select
            value={draft.artifact_type}
            onValueChange={(artifact_type) => setDraft((current) => ({ ...current, artifact_type }))}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="application/openapi+json">OpenAPI</SelectItem>
              <SelectItem value="application/mcp-server+json">MCP server</SelectItem>
              <SelectItem value="application/a2a-agent-card+json">A2A agent card</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <SettingsInput
          label="Artifact URL override"
          value={draft.artifact_url_override || ""}
          onChange={(artifact_url_override) => setDraft((current) => ({ ...current, artifact_url_override }))}
        />
        {publication.validation_errors && publication.validation_errors.length > 0 && (
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-muted-foreground">
            {publication.validation_errors.map((item) => (
              <div key={item}>{item}</div>
            ))}
          </div>
        )}
        <Button onClick={() => savePublication.mutate(draft)} disabled={savePublication.isPending}>
          <Share2 className="size-4" />
          Save exposure
        </Button>
      </CardContent>
    </Card>
  );
}

function ExternalSearchTab({ dashboard }: { dashboard: ARDDashboard }) {
  const importEntry = useDashboardMutation((args: { result: ARDSearchResult }) =>
    importARDEntry(args.result, args.result.source)
  );
  const searchMutation = useMutation({ mutationFn: searchExternalARD });
  const [query, setQuery] = useState("");

  return (
    <div className="grid gap-4">
      <Card interactive={false}>
        <CardContent className="flex flex-col gap-3 p-4 md:flex-row md:items-end">
          <div className="grid flex-1 gap-1.5">
            <Label>Search configured ARD registries</Label>
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="review contracts, summarize filings, generate tests"
              disabled={!dashboard.config.external_search_enabled}
            />
          </div>
          <Button
            onClick={() => searchMutation.mutate({ query: { text: query }, pageSize: 20, federation: "none" })}
            disabled={!dashboard.config.external_search_enabled || searchMutation.isPending}
          >
            {searchMutation.isPending ? <Loader2 className="size-4 animate-spin" /> : <Search className="size-4" />}
            {searchMutation.isPending ? "Searching" : "Search"}
          </Button>
        </CardContent>
      </Card>

      {!dashboard.config.external_search_enabled && (
        <p className="text-sm text-muted-foreground">External search is disabled by deployment config.</p>
      )}

      {searchMutation.isError && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="size-4" />
          {searchMutation.error instanceof Error ? searchMutation.error.message : "External ARD search failed"}
        </div>
      )}

      {searchMutation.data?.sources && (
        <div className="flex flex-wrap gap-2">
          {searchMutation.data.sources.map((source) => (
            <Badge key={source.url} variant={source.status === "ok" ? "success" : "degraded"}>
              {source.url}: {source.status}
            </Badge>
          ))}
        </div>
      )}

      {searchMutation.data && searchMutation.data.results.length === 0 && (
        <Card interactive={false}>
          <CardContent className="p-6 text-sm text-muted-foreground">
            No external ARD resources matched this search.
          </CardContent>
        </Card>
      )}

      <div className="grid gap-3">
        {(searchMutation.data?.results || []).map((result) => (
          <Card key={`${result.source}-${result.identifier}`} variant="surface" interactive={false}>
            <CardContent className="flex flex-col gap-3 p-4 md:flex-row md:items-start md:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="font-medium">{result.displayName}</h3>
                  <Badge variant="metadata" showIcon={false}>{result.type}</Badge>
                  <Badge variant="outline" showIcon={false}>{result.source || "registry"}</Badge>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">{result.description || result.identifier}</p>
                <div className="mt-2 flex flex-wrap gap-1">
                  {(result.tags || []).slice(0, 6).map((tag) => (
                    <Badge key={tag} variant="secondary" size="sm" showIcon={false}>{tag}</Badge>
                  ))}
                </div>
              </div>
              <Button variant="outline" onClick={() => importEntry.mutate({ result })} disabled={importEntry.isPending}>
                <Plus className="size-4" />
                Import
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function ImportsTab({ dashboard }: { dashboard: ARDDashboard }) {
  const saveBinding = useDashboardMutation((binding: ARDExternalBinding) =>
    saveARDBinding(binding.external_entry_id, binding)
  );

  if (dashboard.imports.length === 0) {
    return (
      <Card interactive={false}>
        <CardContent className="p-6 text-sm text-muted-foreground">
          Imported external ARD resources will appear here. Search registries first, then import an entry before making it callable.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid gap-3">
      {dashboard.imports.map((item) => {
        const binding = item.binding || {
          external_entry_id: item.entry.id,
          callable: false,
          local_target: `external.${slugify(item.entry.publisher || "vendor")}.${slugify(item.entry.display_name)}`,
          adapter: "openapi",
          timeout_ms: 30000,
          allowed_operations: [],
          policy: "",
        };
        return (
          <Card key={item.entry.id} variant="surface" interactive={false}>
            <CardContent className="grid gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_360px]">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="font-medium">{item.entry.display_name}</h3>
                  <Badge variant={item.status === "callable" ? "success" : "secondary"}>{item.status}</Badge>
                  <Badge variant="metadata" showIcon={false}>{item.entry.type}</Badge>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">{item.entry.description || item.entry.identifier}</p>
                <div className="mt-3 grid gap-1 text-xs text-muted-foreground">
                  <div>Source: {item.entry.source_registry || "unknown"}</div>
                  <div>Publisher: {item.entry.publisher || "unknown"}</div>
                  <div className="truncate">Identifier: {item.entry.identifier}</div>
                </div>
              </div>
              <BindingEditor
                binding={binding}
                invocationEnabled={dashboard.config.external_invocation_enabled}
                onSave={(next) => saveBinding.mutate(next)}
                saving={saveBinding.isPending}
              />
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

function BindingEditor({
  binding,
  invocationEnabled,
  onSave,
  saving,
}: {
  binding: ARDExternalBinding;
  invocationEnabled: boolean;
  onSave: (binding: ARDExternalBinding) => void;
  saving: boolean;
}) {
  const policyId = useId();
  const [draft, setDraft] = useState(binding);
  const [timeoutValue, setTimeoutValue] = useState(String(binding.timeout_ms || 30000));
  const [allowedOperationsValue, setAllowedOperationsValue] = useState((binding.allowed_operations || []).join(", "));
  return (
    <div className="grid gap-3 rounded-md border border-border bg-background p-3">
      <ToggleRow
        label="Make callable"
        checked={draft.callable}
        disabled={!invocationEnabled}
        onCheckedChange={(callable) => setDraft((current) => ({ ...current, callable }))}
      />
      <SettingsInput
        label="SDK target"
        value={draft.local_target}
        onChange={(local_target) => setDraft((current) => ({ ...current, local_target }))}
      />
      <div className="grid gap-1.5">
        <Label>Adapter</Label>
        <Select value={draft.adapter} onValueChange={(adapter) => setDraft((current) => ({ ...current, adapter }))}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="openapi">OpenAPI</SelectItem>
            <SelectItem value="mcp">MCP</SelectItem>
            <SelectItem value="a2a">A2A</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <SettingsInput
        label="Auth ref"
        value={draft.auth_ref || ""}
        onChange={(auth_ref) => setDraft((current) => ({ ...current, auth_ref }))}
      />
      <SettingsInput
        label="Timeout ms"
        value={timeoutValue}
        onChange={(value) => {
          setTimeoutValue(value);
          setDraft((current) => ({ ...current, timeout_ms: Number(value) || 30000 }));
        }}
      />
      <SettingsInput
        label="Allowed operations"
        value={allowedOperationsValue}
        onChange={(value) => {
          setAllowedOperationsValue(value);
          setDraft((current) => ({ ...current, allowed_operations: splitList(value) }));
        }}
      />
      <div className="grid gap-1.5">
        <Label htmlFor={policyId}>Policy</Label>
        <AutoExpandingTextarea
          id={policyId}
          value={draft.policy || ""}
          onChange={(event) => setDraft((current) => ({ ...current, policy: event.target.value }))}
          maxHeight={120}
        />
      </div>
      {!invocationEnabled && (
        <p className="text-xs text-muted-foreground">External invocation is disabled by deployment config.</p>
      )}
      <Button onClick={() => onSave(draft)} disabled={saving}>
        <Database className="size-4" />
        Save callable binding
      </Button>
    </div>
  );
}

function RegistryTab({ dashboard }: { dashboard: ARDDashboard }) {
  const saveRegistries = useDashboardMutation(saveARDRegistries);
  const [rows, setRows] = useState<ARDRegistryRecord[]>(dashboard.registries || []);
  const hasInvalidRows = rows.some((row) => row.url.trim() === "");

  return (
    <div className="grid gap-4">
      <Card interactive={false}>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Known ARD registries</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 p-4 pt-0">
          {!dashboard.config.registry_enabled && (
            <div className="flex items-center gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-muted-foreground">
              <AlertCircle className="size-4 text-amber-600" />
              Registry endpoints are disabled by deployment config.
            </div>
          )}
          {rows.map((row, index) => (
            <div key={index} className="grid gap-2 rounded-md border border-border p-3">
              <div className="grid gap-2 md:grid-cols-[1fr_1fr_180px_40px]">
                <Input
                  value={row.name || ""}
                  placeholder="Registry name"
                  onChange={(event) => updateRow(rows, setRows, index, { ...row, name: event.target.value })}
                />
                <div className="grid gap-1">
                  <Input
                    value={row.url}
                    placeholder="https://registry.example/api/v1/ard"
                    aria-invalid={row.url.trim() === ""}
                    onChange={(event) => updateRow(rows, setRows, index, { ...row, url: event.target.value })}
                  />
                  {row.url.trim() === "" && (
                    <span className="text-xs text-destructive">Registry URL is required.</span>
                  )}
                </div>
                <Select
                  value={row.submission_state || "manual"}
                  onValueChange={(submission_state) => updateRow(rows, setRows, index, { ...row, submission_state })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="manual">Manual</SelectItem>
                    <SelectItem value="submitted">Submitted</SelectItem>
                    <SelectItem value="indexed">Indexed</SelectItem>
                    <SelectItem value="failed">Failed</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove registry ${row.name || row.url || index + 1}`}
                  onClick={() => setRows((current) => current.filter((_, rowIndex) => rowIndex !== index))}
                >
                  <Trash className="size-4" />
                </Button>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="outline" size="sm" showIcon={false}>
                  {row.submission_state || "manual"}
                </Badge>
                <span>Last checked: {formatDate(row.last_checked_at)}</span>
              </div>
            </div>
          ))}
          {rows.length === 0 && (
            <div className="rounded-md border border-border bg-muted/20 p-4 text-sm text-muted-foreground">
              No ARD registries configured.
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              onClick={() => setRows((current) => [...current, { url: "", name: "", submission_state: "manual" }])}
            >
              <Plus className="size-4" />
              Add registry
            </Button>
            <Button onClick={() => saveRegistries.mutate(rows)} disabled={saveRegistries.isPending || hasInvalidRows}>
              <ShieldCheck className="size-4" />
              Save registries
            </Button>
          </div>
          <div className="rounded-md border border-border bg-muted/20 p-3 text-sm text-muted-foreground">
            Catalog URL for manual submission: <code className="text-foreground">{dashboard.summary.catalog_url}</code>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function StatusRow({ label, active }: { label: string; active: boolean }) {
  return (
    <div className="flex items-center justify-between rounded-md border border-border bg-background p-3">
      <span className="text-sm">{label}</span>
      <StatusBadge active={active} label={active ? "OK" : "Off"} />
    </div>
  );
}

function StatusBadge({ active, label }: { active: boolean; label: string }) {
  return <Badge variant={active ? "success" : "secondary"}>{label}</Badge>;
}

function PublicationBadge({ publication }: { publication: ARDPublicationView }) {
  if (!publication.published) {
    return <Badge variant="secondary">Private</Badge>;
  }
  if (publication.validation_status === "valid") {
    return <Badge variant="success">Published</Badge>;
  }
  return <Badge variant="degraded">Needs setup</Badge>;
}

function ToggleRow({
  label,
  checked,
  disabled,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-background p-3">
      <div className="min-w-0">
        <div className="text-sm font-medium">{label}</div>
        {disabled && <div className="text-xs text-muted-foreground">Locked by deployment config</div>}
      </div>
      <Switch checked={checked} disabled={disabled} onCheckedChange={onCheckedChange} />
    </div>
  );
}

function SettingsInput({
  label,
  value,
  locked,
  onChange,
}: {
  label: string;
  value: string;
  locked?: boolean;
  onChange: (value: string) => void;
}) {
  const id = useId();
  return (
    <div className="grid gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor={id}>{label}</Label>
        {locked && <Badge variant="metadata" showIcon={false}>config</Badge>}
      </div>
      <Input id={id} value={value} disabled={locked} onChange={(event) => onChange(event.target.value)} />
    </div>
  );
}

function updateRow<T>(rows: T[], setRows: (rows: T[]) => void, index: number, next: T) {
  const copy = [...rows];
  copy[index] = next;
  setRows(copy);
}

function splitList(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function splitLines(value: string): string[] {
  return value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
}

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "resource";
}

function formatDate(value?: string): string {
  if (!value) return "never";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  return date.toLocaleString();
}

function copyText(value: string) {
  void navigator.clipboard?.writeText(value);
}
