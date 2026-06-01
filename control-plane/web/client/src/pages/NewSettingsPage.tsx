import { useCallback, useEffect, useState } from "react";
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import {
  Trash,
  Plus,
  CheckCircle,
  XCircle,
  Renew,
  Eye,
  EyeOff,
  Copy,
} from "@/components/ui/icon-bridge";
import {
  getObservabilityWebhook,
  setObservabilityWebhook,
  deleteObservabilityWebhook,
  getObservabilityWebhookStatus,
  redriveDeadLetterQueue,
  clearDeadLetterQueue,
  type ObservabilityWebhookConfig,
  type ObservabilityForwarderStatus,
  type ObservabilityWebhookRequest,
} from "@/services/observabilityWebhookApi";
import { getDIDSystemStatus } from "@/services/didApi";
import {
  getNodeLogProxySettings,
  putNodeLogProxySettings,
} from "@/services/api";
import { formatRelativeTime } from "@/utils/dateFormat";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface HeaderEntry {
  key: string;
  value: string;
}

const REMOVE_WEBHOOK_COPY = {
  title: "Remove webhook configuration?",
  description:
    "The control plane will stop delivering events to this endpoint. You can reconfigure it later, but any events fired in the meantime will be dropped.",
  confirmLabel: "Remove webhook",
  keepLabel: "Keep webhook",
} as const;

const REDRIVE_DLQ_COPY = {
  title: (count: number) =>
    count > 1 ? `Retry ${count} failed events?` : "Retry this failed event?",
  description:
    "Each event will be re-delivered to the webhook. Events that still fail will return to the dead-letter queue.",
  confirmLabel: (count: number) =>
    count > 1 ? `Retry ${count} events` : "Retry event",
  keepLabel: "Cancel",
} as const;

const CLEAR_DLQ_COPY = {
  title: (count: number) =>
    count > 1
      ? `Permanently delete ${count} failed events?`
      : "Permanently delete this failed event?",
  description:
    "This cannot be undone. The events will be removed from the dead-letter queue and will not be retried.",
  confirmLabel: (count: number) =>
    count > 1 ? `Delete ${count} events` : "Delete event",
  keepLabel: "Keep events",
} as const;

// ---------------------------------------------------------------------------
// Tab: General
// ---------------------------------------------------------------------------

function GeneralTab() {
  const serverUrl =
    (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace("/api/ui/v1", "") ||
    window.location.origin;

  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(serverUrl).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">API Endpoint</CardTitle>
          <CardDescription className="text-muted-foreground">
            Point your agents to this URL using the <code className="text-xs font-mono bg-muted px-1 py-0.5 rounded">AGENTFIELD_SERVER</code> environment variable.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-2">
            <code className="text-xs font-mono bg-muted px-3 py-2 rounded flex-1 truncate">
              {serverUrl}
            </code>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 shrink-0"
              onClick={handleCopy}
            >
              {copied ? (
                <CheckCircle className="size-3 text-status-success" />
              ) : (
                <Copy className="size-3" />
              )}
              <span className="ml-1 text-xs">{copied ? "Copied" : "Copy"}</span>
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            All agent SDK calls, workflow triggers, and execution events flow through this endpoint.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Quick Start</CardTitle>
          <CardDescription className="text-muted-foreground">
            Configure your agent to connect to this instance.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="rounded-md bg-muted px-3 py-2">
            <code className="text-xs font-mono whitespace-pre-wrap break-all">
              {`export AGENTFIELD_SERVER="${serverUrl}"`}
            </code>
          </div>
          <p className="text-xs text-muted-foreground">
            Then run your agent with <code className="font-mono bg-muted px-1 py-0.5 rounded">af run</code> or start it directly.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab: Observability (migrated from ObservabilityWebhookSettingsPage)
// ---------------------------------------------------------------------------

function ObservabilityTab() {
  const [url, setUrl] = useState("");
  const [secret, setSecret] = useState("");
  const [showSecret, setShowSecret] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [headers, setHeaders] = useState<HeaderEntry[]>([]);

  const [config, setConfig] = useState<ObservabilityWebhookConfig | null>(null);
  const [status, setStatus] = useState<ObservabilityForwarderStatus | null>(null);
  const [isConfigured, setIsConfigured] = useState(false);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [redriving, setRedriving] = useState(false);
  const [clearingDlq, setClearingDlq] = useState(false);
  const [removeDialogOpen, setRemoveDialogOpen] = useState(false);
  const [redriveDialogOpen, setRedriveDialogOpen] = useState(false);
  const [clearDlqDialogOpen, setClearDlqDialogOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [configResponse, statusResponse] = await Promise.all([
        getObservabilityWebhook(),
        getObservabilityWebhookStatus(),
      ]);

      setIsConfigured(configResponse.configured);
      setConfig(configResponse.config || null);
      setStatus(statusResponse);

      if (configResponse.config) {
        setUrl(configResponse.config.url);
        setEnabled(configResponse.config.enabled);
        setHeaders(
          Object.entries(configResponse.config.headers || {}).map(([key, value]) => ({
            key,
            value,
          }))
        );
        setSecret("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load configuration");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  const handleSave = async () => {
    if (!url.trim()) {
      setError("Webhook URL is required");
      return;
    }
    try {
      new URL(url);
    } catch {
      setError("Invalid URL format");
      return;
    }

    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const request: ObservabilityWebhookRequest = {
        url: url.trim(),
        enabled,
        headers: headers.reduce(
          (acc, h) => {
            if (h.key.trim()) {
              acc[h.key.trim()] = h.value;
            }
            return acc;
          },
          {} as Record<string, string>
        ),
      };
      if (secret.trim()) {
        request.secret = secret.trim();
      }
      await setObservabilityWebhook(request);
      setSuccess("Webhook configuration saved successfully");
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save configuration");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    setDeleting(true);
    setError(null);
    setSuccess(null);
    try {
      await deleteObservabilityWebhook();
      setSuccess("Webhook configuration removed");
      setUrl("");
      setSecret("");
      setEnabled(true);
      setHeaders([]);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete configuration");
    } finally {
      setDeleting(false);
      setRemoveDialogOpen(false);
    }
  };

  const handleRedrive = async () => {
    if (!status?.dead_letter_count) return;
    setRedriving(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await redriveDeadLetterQueue();
      setSuccess(result.success ? `Successfully redrove ${result.processed} events` : result.message);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to redrive dead letter queue");
    } finally {
      setRedriving(false);
      setRedriveDialogOpen(false);
    }
  };

  const handleClearDlq = async () => {
    if (!status?.dead_letter_count) return;
    setClearingDlq(true);
    setError(null);
    setSuccess(null);
    try {
      await clearDeadLetterQueue();
      setSuccess("Dead letter queue cleared");
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to clear dead letter queue");
    } finally {
      setClearingDlq(false);
      setClearDlqDialogOpen(false);
    }
  };

  const addHeader = () => setHeaders([...headers, { key: "", value: "" }]);

  const updateHeader = (index: number, field: "key" | "value", value: string) => {
    const updated = [...headers];
    updated[index][field] = value;
    setHeaders(updated);
  };

  const removeHeader = (index: number) => setHeaders(headers.filter((_, i) => i !== index));

  if (loading) {
    return (
      <Card>
        <CardContent className="py-8">
          <div className="flex items-center justify-center">
            <Renew className="h-6 w-6 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground">Loading configuration...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {success && (
        <Alert variant="success">
          <CheckCircle className="h-4 w-4" />
          <AlertTitle>Success</AlertTitle>
          <AlertDescription className="text-status-success/90">{success}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Configuration Card */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Observability Webhook</CardTitle>
                <CardDescription className="text-muted-foreground">
                  Forward execution events, agent lifecycle events, and node status updates to an external endpoint.
                </CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={loadData} disabled={loading}>
                <Renew className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Enable toggle */}
            <div className="flex items-center justify-between rounded-lg border p-4">
              <div className="space-y-0.5">
                <Label htmlFor="obs-enabled" className="text-base font-medium">
                  Enable Webhook
                </Label>
                <p className="text-sm text-muted-foreground">
                  When enabled, events will be forwarded to the configured URL
                </p>
              </div>
              <Switch id="obs-enabled" checked={enabled} onCheckedChange={setEnabled} />
            </div>

            {/* URL */}
            <div className="space-y-2">
              <Label htmlFor="obs-url">Webhook URL</Label>
              <Input
                id="obs-url"
                type="url"
                placeholder="https://your-service.com/webhook"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                autoComplete="off"
              />
              <p className="text-sm text-muted-foreground">
                HTTPS endpoint that will receive event batches via POST requests
              </p>
            </div>

            {/* Secret */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="obs-secret">HMAC Secret (Optional)</Label>
                {config?.has_secret && (
                  <Badge
                    variant="outline"
                    showIcon={false}
                    className="gap-1 border-status-success/40 text-status-success"
                  >
                    <CheckCircle className="h-3 w-3 shrink-0" />
                    Configured
                  </Badge>
                )}
              </div>
              <div className="relative">
                <Input
                  id="obs-secret"
                  type={showSecret ? "text" : "password"}
                  placeholder={config?.has_secret ? "Enter new secret to replace existing" : "Optional signing secret"}
                  value={secret}
                  onChange={(e) => setSecret(e.target.value)}
                  className="pr-10"
                  autoComplete="new-password"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                  onClick={() => setShowSecret(!showSecret)}
                >
                  {showSecret ? (
                    <EyeOff className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <Eye className="h-4 w-4 text-muted-foreground" />
                  )}
                </Button>
              </div>
              <p className="text-sm text-muted-foreground">
                If set, requests will include an X-AgentField-Signature header with HMAC-SHA256 signature
              </p>
            </div>

            {/* Custom headers */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Custom Headers</Label>
                <Button type="button" variant="outline" size="sm" onClick={addHeader}>
                  <Plus className="h-4 w-4 mr-1" />
                  Add Header
                </Button>
              </div>
              {headers.length === 0 ? (
                <p className="text-sm text-muted-foreground">No custom headers configured</p>
              ) : (
                <div className="space-y-2">
                  {headers.map((header, index) => (
                    <div key={index} className="flex gap-2">
                      <Input
                        placeholder="Header name"
                        value={header.key}
                        onChange={(e) => updateHeader(index, "key", e.target.value)}
                        className="flex-1"
                      />
                      <Input
                        placeholder="Header value"
                        value={header.value}
                        onChange={(e) => updateHeader(index, "value", e.target.value)}
                        className="flex-1"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => removeHeader(index)}
                        className="px-2"
                      >
                        <Trash className="h-4 w-4 text-muted-foreground" />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
          <CardFooter className="flex justify-between border-t pt-6">
            <div>
              {isConfigured && (
                <Button
                  variant="destructive"
                  onClick={() => setRemoveDialogOpen(true)}
                  disabled={deleting || saving}
                >
                  {deleting ? (
                    <Renew className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Trash className="h-4 w-4 mr-2" />
                  )}
                  Remove Webhook
                </Button>
              )}
            </div>
            <Button onClick={handleSave} disabled={saving || deleting}>
              {saving ? (
                <Renew className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <CheckCircle className="h-4 w-4 mr-2" />
              )}
              {isConfigured ? "Update Configuration" : "Save Configuration"}
            </Button>
          </CardFooter>
        </Card>

        {/* Status Card */}
        <Card>
          <CardHeader>
            <CardTitle>Forwarder Status</CardTitle>
            <CardDescription className="text-muted-foreground">
              Real-time status of the event forwarder
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Status</span>
              {status?.enabled ? (
                <Badge variant="success" className="font-sans tracking-normal">
                  Active
                </Badge>
              ) : (
                <Badge variant="secondary" showIcon={false} className="gap-1">
                  <XCircle className="h-3 w-3 shrink-0" />
                  Inactive
                </Badge>
              )}
            </div>

            <div className="space-y-3 pt-2 border-t">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Events Forwarded</span>
                <span className="text-sm font-mono font-medium">
                  {status?.events_forwarded?.toLocaleString() ?? 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Events Dropped</span>
                <span className="text-sm font-mono font-medium">
                  {status?.events_dropped?.toLocaleString() ?? 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Queue Depth</span>
                <span className="text-sm font-mono font-medium">{status?.queue_depth ?? 0}</span>
              </div>
            </div>

            <div className="pt-2 border-t space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Dead Letter Queue</span>
                <span
                  className={cn(
                    "text-sm font-mono font-medium",
                    (status?.dead_letter_count ?? 0) > 0 && "text-status-warning"
                  )}
                >
                  {status?.dead_letter_count?.toLocaleString() ?? 0}
                </span>
              </div>
              {(status?.dead_letter_count ?? 0) > 0 && (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setRedriveDialogOpen(true)}
                    disabled={redriving || clearingDlq}
                    className="flex-1"
                  >
                    <Renew className={`h-3 w-3 mr-1 ${redriving ? "animate-spin" : ""}`} />
                    Redrive
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setClearDlqDialogOpen(true)}
                    disabled={redriving || clearingDlq}
                    className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                  >
                    {clearingDlq ? (
                      <Renew className="h-3 w-3 mr-1 animate-spin" />
                    ) : (
                      <Trash className="h-3 w-3 mr-1" />
                    )}
                    Clear
                  </Button>
                </div>
              )}
            </div>

            {status?.last_forwarded_at && (
              <div className="pt-2 border-t">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Last Forwarded</span>
                  <span className="text-sm">{formatRelativeTime(status.last_forwarded_at)}</span>
                </div>
              </div>
            )}

            {status?.last_error && (
              <div className="pt-2 border-t">
                <span className="text-sm text-muted-foreground">Last Error</span>
                <p className="mt-1 break-all font-mono text-xs text-status-error">
                  {status.last_error}
                </p>
              </div>
            )}

            {config && (
              <div className="pt-2 border-t space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Created</span>
                  <span className="text-sm">{formatRelativeTime(config.created_at)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Updated</span>
                  <span className="text-sm">{formatRelativeTime(config.updated_at)}</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Event Types Info */}
      <Card>
        <CardHeader>
          <CardTitle>Event Types</CardTitle>
          <CardDescription className="text-muted-foreground">
            All of the following event types are forwarded to your webhook endpoint
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <h4 className="font-medium">Execution Events</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>execution_created</li>
                <li>execution_started</li>
                <li>execution_updated</li>
                <li>execution_completed</li>
                <li>execution_failed</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium">Node Events</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>node_online</li>
                <li>node_offline</li>
                <li>node_registered</li>
                <li>node_status_changed</li>
                <li>node_health_changed</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium">Reasoner Events</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>reasoner_online</li>
                <li>reasoner_offline</li>
                <li>reasoner_updated</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      <AlertDialog open={removeDialogOpen} onOpenChange={setRemoveDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{REMOVE_WEBHOOK_COPY.title}</AlertDialogTitle>
            <AlertDialogDescription>
              {REMOVE_WEBHOOK_COPY.description}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>
              {REMOVE_WEBHOOK_COPY.keepLabel}
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={deleting}
              onClick={(event) => {
                event.preventDefault();
                void handleDelete();
              }}
            >
              {REMOVE_WEBHOOK_COPY.confirmLabel}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={redriveDialogOpen} onOpenChange={setRedriveDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {REDRIVE_DLQ_COPY.title(status?.dead_letter_count ?? 0)}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {REDRIVE_DLQ_COPY.description}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={redriving}>
              {REDRIVE_DLQ_COPY.keepLabel}
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={redriving}
              onClick={(event) => {
                event.preventDefault();
                void handleRedrive();
              }}
            >
              {REDRIVE_DLQ_COPY.confirmLabel(status?.dead_letter_count ?? 0)}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={clearDlqDialogOpen} onOpenChange={setClearDlqDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {CLEAR_DLQ_COPY.title(status?.dead_letter_count ?? 0)}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {CLEAR_DLQ_COPY.description}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={clearingDlq}>
              {CLEAR_DLQ_COPY.keepLabel}
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={clearingDlq}
              onClick={(event) => {
                event.preventDefault();
                void handleClearDlq();
              }}
            >
              {CLEAR_DLQ_COPY.confirmLabel(status?.dead_letter_count ?? 0)}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab: Identity & Trust
// ---------------------------------------------------------------------------

function IdentityTab() {
  const [serverDid, setServerDid] = useState<string>("");
  const [didStatus, setDidStatus] = useState<string>("unknown");
  const [loadingDid, setLoadingDid] = useState(true);
  const [didCopied, setDidCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;

    // Fetch system status and the actual server DID in parallel.
    // The /did/status endpoint only returns operational status — the DID itself
    // lives at /api/v1/did/agentfield-server (note: v1, not ui/v1).
    const statusFetch = getDIDSystemStatus().catch(() => null);
    const serverUrl =
      (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace("/api/ui/v1", "") ||
      window.location.origin;
    const serverDidFetch = fetch(`${serverUrl}/api/v1/did/agentfield-server`)
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);

    Promise.all([statusFetch, serverDidFetch]).then(([statusRes, serverDidRes]) => {
      if (cancelled) return;
      if (statusRes) {
        setDidStatus(statusRes.status);
      } else {
        setDidStatus("error");
      }
      if (serverDidRes && typeof serverDidRes.agentfield_server_did === "string" && serverDidRes.agentfield_server_did) {
        setServerDid(serverDidRes.agentfield_server_did);
      } else {
        setServerDid("");
      }
      setLoadingDid(false);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const handleCopyDid = () => {
    if (!serverDid) return;
    navigator.clipboard.writeText(serverDid).then(() => {
      setDidCopied(true);
      setTimeout(() => setDidCopied(false), 2000);
    });
  };

  const handleExportCredentials = async () => {
    window.open("/api/ui/v1/did/export/vcs", "_blank");
  };

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Identity & Trust</CardTitle>
          <CardDescription className="text-muted-foreground">
            Cryptographic identity and verifiable credentials configuration.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {/* VC enabled indicator */}
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0 space-y-1">
              <Label className="text-sm font-medium">Verifiable Credentials</Label>
              <p className="text-xs text-muted-foreground">Generate W3C VCs for each execution</p>
            </div>
            <Badge variant="secondary">Enabled</Badge>
          </div>

          <Separator />

          {/* DID system status */}
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0 space-y-1">
              <Label className="text-sm font-medium">DID System Status</Label>
              <p className="text-xs text-muted-foreground">Decentralised identity infrastructure</p>
            </div>
            {loadingDid ? (
              <Badge variant="outline" showIcon={false} className="gap-1 shrink-0">
                <Renew className="h-3 w-3 shrink-0 animate-spin" />
                Checking...
              </Badge>
            ) : didStatus === "ok" || didStatus === "active" ? (
              <Badge variant="success" className="shrink-0 font-sans tracking-normal">
                Online
              </Badge>
            ) : (
              <Badge variant="secondary" className="shrink-0 capitalize">
                {didStatus}
              </Badge>
            )}
          </div>

          <Separator />

          {/* Server DID */}
          <div className="flex flex-col gap-2">
            <Label htmlFor="server-did" className="text-sm font-medium">
              Server DID
            </Label>
            {loadingDid ? (
              <p className="font-mono text-xs text-muted-foreground">Loading...</p>
            ) : serverDid ? (
              <div className="flex items-center gap-2">
                <Input
                  id="server-did"
                  readOnly
                  value={serverDid}
                  className="min-h-9 flex-1 font-mono text-xs"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  className="shrink-0"
                  onClick={handleCopyDid}
                  aria-label={didCopied ? "Copied" : "Copy server DID"}
                >
                  {didCopied ? (
                    <CheckCircle className="size-3 text-status-success" />
                  ) : (
                    <Copy className="size-3" />
                  )}
                </Button>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                DID system not configured — server DID unavailable in local mode.
              </p>
            )}
          </div>

          <Button variant="outline" size="sm" onClick={handleExportCredentials} className="w-fit">
            Export All Credentials
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab: About
// ---------------------------------------------------------------------------

function AboutTab() {
  const serverUrl =
    (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace("/api/ui/v1", "") ||
    window.location.origin;

  return (
    <Card>
      <CardHeader>
        <CardTitle>About AgentField</CardTitle>
        <CardDescription className="text-muted-foreground">
          Platform version and runtime information.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Version</span>
          <span className="font-mono">0.1.63</span>
        </div>
        <Separator />
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Server URL</span>
          <span className="font-mono">{serverUrl}</span>
        </div>
        <Separator />
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Storage Mode</span>
          <Badge variant="secondary">Local (SQLite)</Badge>
        </div>
        <Separator />
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">UI Base Path</span>
          <span className="font-mono">{import.meta.env.VITE_BASE_PATH || "/ui"}</span>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Tab: Agent logs (control plane → node log proxy limits)
// ---------------------------------------------------------------------------

function NodeLogProxyTab() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [locks, setLocks] = useState<Record<string, boolean>>({});
  const [connectTimeout, setConnectTimeout] = useState("");
  const [streamIdleTimeout, setStreamIdleTimeout] = useState("");
  const [maxStreamDuration, setMaxStreamDuration] = useState("");
  const [maxTailLines, setMaxTailLines] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await getNodeLogProxySettings();
      setLocks(r.env_locks ?? {});
      const e = r.effective;
      setConnectTimeout(e.connect_timeout);
      setStreamIdleTimeout(e.stream_idle_timeout);
      setMaxStreamDuration(e.max_stream_duration);
      setMaxTailLines(String(e.max_tail_lines));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load settings");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!success) return;
    const t = setTimeout(() => setSuccess(null), 4000);
    return () => clearTimeout(t);
  }, [success]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);
    const maxLines = parseInt(maxTailLines, 10);
    if (
      !connectTimeout.trim() ||
      !streamIdleTimeout.trim() ||
      !maxStreamDuration.trim()
    ) {
      setError("Connect timeout, stream idle timeout, and max stream duration are required (Go duration strings, e.g. 30s, 5m).");
      setSaving(false);
      return;
    }
    if (!Number.isFinite(maxLines) || maxLines < 1) {
      setError("Max tail lines must be a positive integer.");
      setSaving(false);
      return;
    }
    try {
      await putNodeLogProxySettings({
        connect_timeout: connectTimeout.trim(),
        stream_idle_timeout: streamIdleTimeout.trim(),
        max_stream_duration: maxStreamDuration.trim(),
        max_tail_lines: maxLines,
      });
      setSuccess("Saved node log proxy limits.");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const anyLock = Object.values(locks).some(Boolean);

  if (loading) {
    return (
      <p className="text-sm text-muted-foreground">Loading log proxy settings…</p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {anyLock ? (
        <Alert>
          <AlertTitle className="text-sm">Environment overrides</AlertTitle>
          <AlertDescription className="text-xs">
            Some fields are locked by{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-[11px]">
              AGENTFIELD_NODE_LOG_PROXY_*
            </code>{" "}
            or{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-[11px]">
              AGENTFIELD_NODE_LOG_MAX_TAIL_LINES
            </code>
            . Unset them to edit from the UI.
          </AlertDescription>
        </Alert>
      ) : null}

      {error ? (
        <Alert variant="destructive">
          <XCircle className="size-4" />
          <AlertTitle className="text-sm">Error</AlertTitle>
          <AlertDescription className="text-xs">{error}</AlertDescription>
        </Alert>
      ) : null}

      {success ? (
        <Alert>
          <CheckCircle className="size-4 text-status-success" />
          <AlertTitle className="text-sm">Saved</AlertTitle>
          <AlertDescription className="text-xs">{success}</AlertDescription>
        </Alert>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">
            Node log proxy
          </CardTitle>
          <CardDescription className="text-muted-foreground text-xs">
            Limits for the UI&apos;s{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-[11px]">
              GET /nodes/:id/logs
            </code>{" "}
            proxy to agent NDJSON logs (tail size, connect and stream timeouts).
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="grid gap-2">
            <Label htmlFor="nlp-connect" className="text-xs">
              Connect timeout
            </Label>
            <Input
              id="nlp-connect"
              value={connectTimeout}
              onChange={(e) => setConnectTimeout(e.target.value)}
              disabled={locks.connect_timeout}
              className="font-mono text-sm h-9"
              placeholder="e.g. 10s"
            />
            {locks.connect_timeout ? (
              <Badge variant="secondary" className="w-fit text-[10px]">
                env locked
              </Badge>
            ) : null}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="nlp-idle" className="text-xs">
              Stream idle timeout
            </Label>
            <Input
              id="nlp-idle"
              value={streamIdleTimeout}
              onChange={(e) => setStreamIdleTimeout(e.target.value)}
              disabled={locks.stream_idle_timeout}
              className="font-mono text-sm h-9"
            />
            {locks.stream_idle_timeout ? (
              <Badge variant="secondary" className="w-fit text-[10px]">
                env locked
              </Badge>
            ) : null}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="nlp-maxdur" className="text-xs">
              Max stream duration
            </Label>
            <Input
              id="nlp-maxdur"
              value={maxStreamDuration}
              onChange={(e) => setMaxStreamDuration(e.target.value)}
              disabled={locks.max_stream_duration}
              className="font-mono text-sm h-9"
            />
            {locks.max_stream_duration ? (
              <Badge variant="secondary" className="w-fit text-[10px]">
                env locked
              </Badge>
            ) : null}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="nlp-tail" className="text-xs">
              Max tail lines (per request)
            </Label>
            <Input
              id="nlp-tail"
              type="number"
              min={1}
              value={maxTailLines}
              onChange={(e) => setMaxTailLines(e.target.value)}
              disabled={locks.max_tail_lines}
              className="font-mono text-sm h-9"
            />
            {locks.max_tail_lines ? (
              <Badge variant="secondary" className="w-fit text-[10px]">
                env locked
              </Badge>
            ) : null}
          </div>
        </CardContent>
        <CardFooter className="flex justify-end gap-2 border-t bg-muted/20 py-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void load()}
            disabled={saving}
          >
            <Renew className="size-3.5" />
            <span className="ml-1.5">Reload</span>
          </Button>
          <Button
            type="button"
            size="sm"
            onClick={() => void handleSave()}
            disabled={saving || anyLock}
          >
            {saving ? "Saving…" : "Save"}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page root
// ---------------------------------------------------------------------------

export function NewSettingsPage() {
  return (
    <div className="flex flex-col gap-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Manage platform configuration, webhooks, and identity.
        </p>
      </div>

      <Tabs defaultValue="general">
        <TabsList variant="underline">
          <TabsTrigger value="general" variant="underline">
            General
          </TabsTrigger>
          <TabsTrigger value="observability" variant="underline">
            Observability
          </TabsTrigger>
          <TabsTrigger value="agent-logs" variant="underline">
            Agent logs
          </TabsTrigger>
          <TabsTrigger value="identity" variant="underline">
            Identity
          </TabsTrigger>
          <TabsTrigger value="about" variant="underline">
            About
          </TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="mt-6">
          <GeneralTab />
        </TabsContent>

        <TabsContent value="observability" className="mt-6">
          <ObservabilityTab />
        </TabsContent>

        <TabsContent value="agent-logs" className="mt-6">
          <NodeLogProxyTab />
        </TabsContent>

        <TabsContent value="identity" className="mt-6">
          <IdentityTab />
        </TabsContent>

        <TabsContent value="about" className="mt-6">
          <AboutTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default NewSettingsPage;
