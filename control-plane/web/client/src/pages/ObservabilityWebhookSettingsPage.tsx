import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Trash,
  Plus,
  CheckCircle,
  XCircle,
  Renew,
  Settings,
  Eye,
  EyeOff,
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
import { formatRelativeTime } from "@/utils/dateFormat";

interface HeaderEntry {
  key: string;
  value: string;
}

export function ObservabilityWebhookSettingsPage() {
  // Form state - enabled defaults to false until config is loaded
  const [url, setUrl] = useState("");
  const [secret, setSecret] = useState("");
  const [showSecret, setShowSecret] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [headers, setHeaders] = useState<HeaderEntry[]>([]);

  // Config state
  const [config, setConfig] = useState<ObservabilityWebhookConfig | null>(null);
  const [status, setStatus] = useState<ObservabilityForwarderStatus | null>(null);
  const [isConfigured, setIsConfigured] = useState(false);

  // UI state
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [redriving, setRedriving] = useState(false);
  const [clearingDlq, setClearingDlq] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Load config and status
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

      // Populate form with existing config
      if (configResponse.config) {
        setUrl(configResponse.config.url);
        setEnabled(configResponse.config.enabled);
        setHeaders(
          Object.entries(configResponse.config.headers || {}).map(([key, value]) => ({
            key,
            value,
          }))
        );
        // Secret is not returned from API, keep it empty
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

  // Clear messages after timeout
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  // Handle form submission
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

      // Only include secret if it's been entered (for updates)
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

  // Handle delete
  const handleDelete = async () => {
    if (!confirm("Are you sure you want to remove the webhook configuration?")) {
      return;
    }

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
    }
  };

  // Handle redrive
  const handleRedrive = async () => {
    if (!status?.dead_letter_count) {
      return;
    }

    if (!confirm(`Are you sure you want to retry sending ${status.dead_letter_count} failed events?`)) {
      return;
    }

    setRedriving(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await redriveDeadLetterQueue();
      if (result.success) {
        setSuccess(`Successfully redrove ${result.processed} events`);
      } else {
        setSuccess(result.message);
      }
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to redrive dead letter queue");
    } finally {
      setRedriving(false);
    }
  };

  // Handle clear DLQ
  const handleClearDlq = async () => {
    if (!status?.dead_letter_count) {
      return;
    }

    if (!confirm(`Are you sure you want to permanently delete ${status.dead_letter_count} failed events? This cannot be undone.`)) {
      return;
    }

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
    }
  };

  // Header management
  const addHeader = () => {
    setHeaders([...headers, { key: "", value: "" }]);
  };

  const updateHeader = (index: number, field: "key" | "value", value: string) => {
    const newHeaders = [...headers];
    newHeaders[index][field] = value;
    setHeaders(newHeaders);
  };

  const removeHeader = (index: number) => {
    setHeaders(headers.filter((_, i) => i !== index));
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Settings className="h-8 w-8 text-muted-foreground" />
          <div>
            <h1 className="text-2xl font-bold">Observability Webhook</h1>
            <p className="text-muted-foreground">Configure external event forwarding</p>
          </div>
        </div>
        <Card>
          <CardContent className="py-8">
            <div className="flex items-center justify-center">
              <Renew className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="ml-2 text-muted-foreground">Loading configuration...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Settings className="h-8 w-8 text-muted-foreground" />
          <div>
            <h1 className="text-2xl font-bold">Observability Webhook</h1>
            <p className="text-muted-foreground">
              Forward all platform events to an external endpoint
            </p>
          </div>
        </div>
        <Button variant="outline" size="sm" onClick={loadData} disabled={loading}>
          <Renew className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Alerts */}
      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="border-green-500 bg-green-500/10">
          <CheckCircle className="h-4 w-4 text-green-500" />
          <AlertTitle className="text-green-500">Success</AlertTitle>
          <AlertDescription className="text-green-600">{success}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Configuration Card */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Webhook Configuration</CardTitle>
            <CardDescription>
              Configure where to forward execution events, agent lifecycle events, and node status
              updates
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Enable/Disable Toggle */}
            <div className="flex items-center justify-between rounded-lg border p-4">
              <div className="space-y-0.5">
                <Label htmlFor="enabled" className="text-base font-medium">
                  Enable Webhook
                </Label>
                <p className="text-sm text-muted-foreground">
                  When enabled, events will be forwarded to the configured URL
                </p>
              </div>
              <Switch id="enabled" checked={enabled} onCheckedChange={setEnabled} />
            </div>

            {/* URL Input */}
            <div className="space-y-2">
              <Label htmlFor="url">Webhook URL</Label>
              <Input
                id="url"
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

            {/* Secret Input */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="secret">HMAC Secret (Optional)</Label>
                {config?.has_secret && (
                  <Badge variant="outline" className="text-green-600 border-green-600">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    Configured
                  </Badge>
                )}
              </div>
              <div className="relative">
                <Input
                  id="secret"
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
                If set, requests will include an X-AgentField-Signature header with HMAC-SHA256
                signature
              </p>
            </div>

            {/* Custom Headers */}
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
                <Button variant="destructive" onClick={handleDelete} disabled={deleting || saving}>
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
            <CardDescription>Real-time status of the event forwarder</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Connection Status */}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Status</span>
              {status?.enabled ? (
                <Badge variant="default" className="bg-green-500">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Active
                </Badge>
              ) : (
                <Badge variant="secondary">
                  <XCircle className="h-3 w-3 mr-1" />
                  Inactive
                </Badge>
              )}
            </div>

            {/* Stats */}
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
                <span className="text-sm font-mono font-medium">
                  {status?.queue_depth ?? 0}
                </span>
              </div>
            </div>

            {/* Dead Letter Queue */}
            <div className="pt-2 border-t space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Dead Letter Queue</span>
                <span className={`text-sm font-mono font-medium ${(status?.dead_letter_count ?? 0) > 0 ? 'text-amber-500' : ''}`}>
                  {status?.dead_letter_count?.toLocaleString() ?? 0}
                </span>
              </div>
              {(status?.dead_letter_count ?? 0) > 0 && (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRedrive}
                    disabled={redriving || clearingDlq}
                    className="flex-1"
                  >
                    {redriving ? (
                      <Renew className="h-3 w-3 mr-1 animate-spin" />
                    ) : (
                      <Renew className="h-3 w-3 mr-1" />
                    )}
                    Redrive
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleClearDlq}
                    disabled={redriving || clearingDlq}
                    className="text-red-500 hover:text-red-600 hover:bg-red-50"
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

            {/* Last Activity */}
            {status?.last_forwarded_at && (
              <div className="pt-2 border-t">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Last Forwarded</span>
                  <span className="text-sm">
                    {formatRelativeTime(status.last_forwarded_at)}
                  </span>
                </div>
              </div>
            )}

            {/* Last Error */}
            {status?.last_error && (
              <div className="pt-2 border-t">
                <span className="text-sm text-muted-foreground">Last Error</span>
                <p className="text-sm text-red-500 mt-1 font-mono text-xs break-all">
                  {status.last_error}
                </p>
              </div>
            )}

            {/* Config Info */}
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
          <CardDescription>
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
    </div>
  );
}

export default ObservabilityWebhookSettingsPage;
