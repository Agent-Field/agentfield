import { useEffect, useMemo, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface SourceCatalogEntry {
  name: string;
  kind: "http" | "loop" | string;
  secret_required: boolean;
  config_schema: Record<string, unknown>;
}

interface NewTriggerDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sources: SourceCatalogEntry[];
  defaultSourceName?: string;
  onCreated: () => void;
}

interface SourceHints {
  reasoner: string;
  eventTypes: string;
  secretEnv: string;
  configJson: string;
}

const DEFAULT_HINTS: SourceHints = {
  reasoner: "handle_event",
  eventTypes: "",
  secretEnv: "MY_SECRET",
  configJson: "{}",
};

const SOURCE_HINTS: Record<string, SourceHints> = {
  stripe: {
    reasoner: "handle_payment",
    eventTypes: "payment_intent.succeeded, invoice.paid",
    secretEnv: "STRIPE_WEBHOOK_SECRET",
    configJson: "{}",
  },
  github: {
    reasoner: "handle_pr",
    eventTypes: "pull_request, push",
    secretEnv: "GITHUB_WEBHOOK_SECRET",
    configJson: "{}",
  },
  slack: {
    reasoner: "handle_message",
    eventTypes: "event_callback",
    secretEnv: "SLACK_SIGNING_SECRET",
    configJson: "{}",
  },
  cron: {
    reasoner: "handle_tick",
    eventTypes: "",
    secretEnv: "",
    configJson: '{"expression": "* * * * *", "timezone": "UTC"}',
  },
  generic_hmac: {
    reasoner: "handle_event",
    eventTypes: "",
    secretEnv: "MY_HMAC_SECRET",
    configJson: "{}",
  },
  generic_bearer: {
    reasoner: "handle_event",
    eventTypes: "",
    secretEnv: "MY_BEARER_TOKEN",
    configJson: "{}",
  },
};

function hintsFor(sourceName: string): SourceHints {
  return SOURCE_HINTS[sourceName] ?? DEFAULT_HINTS;
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
  return res.json();
}

export function NewTriggerDialog({
  open,
  onOpenChange,
  sources,
  defaultSourceName,
  onCreated,
}: NewTriggerDialogProps) {
  const [sourceName, setSourceName] = useState(
    defaultSourceName || sources[0]?.name || "",
  );
  const [targetNodeId, setTargetNodeId] = useState("");
  const [targetReasoner, setTargetReasoner] = useState("");
  const [eventTypes, setEventTypes] = useState("");
  const [secretEnv, setSecretEnv] = useState("");
  const [configJson, setConfigJson] = useState("{}");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedSource = useMemo(
    () => sources.find((s) => s.name === sourceName),
    [sources, sourceName],
  );
  const hints = useMemo(() => hintsFor(sourceName), [sourceName]);
  const requiresSecret = selectedSource?.secret_required ?? false;
  const isLoopSource = selectedSource?.kind === "loop";
  const showEventTypes = !isLoopSource;

  // Sync source selection when the dialog is reopened with a different
  // defaultSourceName (e.g. user clicked Connect on a specific Integrations
  // card). Without this, the internal `sourceName` state stays at its initial
  // value from first mount and ignores subsequent prop changes.
  useEffect(() => {
    if (!open) return;
    if (defaultSourceName) {
      setSourceName(defaultSourceName);
    } else if (!sourceName && sources.length > 0) {
      setSourceName(sources[0].name);
    }
  }, [open, defaultSourceName, sources, sourceName]);

  // When the source changes (or the dialog opens), preload the config JSON
  // textarea with that source's recommended starter — empty `{}` for
  // signed webhooks, a real cron expression for loop sources, etc. We only
  // overwrite when the user hasn't typed a custom value yet (configJson is
  // still the default "{}" or the previous source's hint).
  useEffect(() => {
    if (!open) return;
    if (!selectedSource) return;
    setConfigJson((current) => {
      // Don't trample a user-entered config — only refresh if current value is
      // a known starter (the previous source's default or empty).
      const knownStarters = new Set(
        Object.values(SOURCE_HINTS).map((h) => h.configJson),
      );
      knownStarters.add("{}");
      knownStarters.add("");
      if (knownStarters.has(current.trim())) return hints.configJson;
      return current;
    });
  }, [open, selectedSource, hints.configJson]);

  const handleOpenChange = (newOpen: boolean) => {
    if (newOpen === false) {
      setTargetNodeId("");
      setTargetReasoner("");
      setEventTypes("");
      setSecretEnv("");
      setConfigJson("{}");
      setError(null);
    }
    onOpenChange(newOpen);
  };

  async function submit() {
    setError(null);
    setSubmitting(true);
    try {
      let cfg: Record<string, unknown> | undefined;
      try {
        cfg = configJson.trim() ? JSON.parse(configJson) : {};
      } catch (e) {
        setError(`Invalid config JSON: ${(e as Error).message}`);
        setSubmitting(false);
        return;
      }
      await fetchJson(`${serverUrl}/api/v1/triggers`, {
        method: "POST",
        body: JSON.stringify({
          source_name: sourceName,
          target_node_id: targetNodeId,
          target_reasoner: targetReasoner,
          event_types: eventTypes
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          secret_env_var: secretEnv,
          config: cfg,
          enabled: true,
        }),
      });
      handleOpenChange(false);
      onCreated();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>New trigger</DialogTitle>
          <DialogDescription>
            {isLoopSource
              ? "Bind a recurring schedule to a reasoner. The control plane fires the trigger on the schedule defined in the config below."
              : "Bind an inbound event source to a reasoner. The control plane verifies provider signatures using the env-var-named secret before dispatching."}
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-2">
          <div className="grid gap-1.5">
            <Label>Source</Label>
            <Select value={sourceName} onValueChange={setSourceName}>
              <SelectTrigger>
                <SelectValue placeholder="Pick a source" />
              </SelectTrigger>
              <SelectContent>
                {sources.map((s) => (
                  <SelectItem key={s.name} value={s.name}>
                    {s.name} ({s.kind})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="grid gap-1.5">
              <Label>Target node</Label>
              <Input
                value={targetNodeId}
                onChange={(e) => setTargetNodeId(e.target.value)}
                placeholder="my-agent"
              />
            </div>
            <div className="grid gap-1.5">
              <Label>Target reasoner</Label>
              <Input
                value={targetReasoner}
                onChange={(e) => setTargetReasoner(e.target.value)}
                placeholder={hints.reasoner}
              />
            </div>
          </div>

          {showEventTypes ? (
            <div className="grid gap-1.5">
              <Label>Event types (comma-separated, blank = all)</Label>
              <Input
                value={eventTypes}
                onChange={(e) => setEventTypes(e.target.value)}
                placeholder={hints.eventTypes || "(blank for all events)"}
              />
            </div>
          ) : null}

          {requiresSecret ? (
            <div className="grid gap-1.5">
              <Label>Secret env var</Label>
              <Input
                value={secretEnv}
                onChange={(e) => setSecretEnv(e.target.value)}
                placeholder={hints.secretEnv}
              />
              <p className="text-xs text-muted-foreground">
                The control plane reads this env var at request time — the
                secret value never leaves the server.
              </p>
            </div>
          ) : null}

          <div className="grid gap-1.5">
            <Label>Config (JSON, source-specific)</Label>
            <textarea
              value={configJson}
              onChange={(e) => setConfigJson(e.target.value)}
              rows={isLoopSource ? 3 : 4}
              className="rounded-md border border-input bg-background px-3 py-2 font-mono text-xs"
            />
            {selectedSource ? (
              <p className="text-xs text-muted-foreground">
                Schema:{" "}
                <code>{JSON.stringify(selectedSource.config_schema)}</code>
              </p>
            ) : null}
          </div>

          {error ? (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          ) : null}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={submitting}
          >
            Cancel
          </Button>
          <Button onClick={submit} disabled={submitting}>
            {submitting ? "Creating…" : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
