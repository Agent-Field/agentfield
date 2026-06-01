import { useCallback, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { verifyProvenanceAudit } from "@/services/vcApi";
import type { ProvenanceVerificationResponse } from "@/types/did";
import { cn } from "@/lib/utils";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { Upload, FileCheck2, AlertTriangle, Info } from "lucide-react";

export function VerifyProvenancePage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [jsonText, setJsonText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProvenanceVerificationResponse | null>(null);

  const loadFile = useCallback((f: File) => {
    setError(null);
    void f.text().then(setJsonText).catch(() => setError("Could not read file"));
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (!f) return;
      loadFile(f);
    },
    [loadFile],
  );

  const onFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) loadFile(f);
      e.target.value = "";
    },
    [loadFile],
  );

  const onVerify = async () => {
    setError(null);
    setResult(null);
    const trimmed = jsonText.trim();
    if (!trimmed) {
      setError("Paste JSON or drop a file first.");
      return;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed) as unknown;
    } catch {
      setError("Invalid JSON — fix syntax and try again.");
      return;
    }
    setLoading(true);
    try {
      const res = await verifyProvenanceAudit(parsed);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification request failed");
    } finally {
      setLoading(false);
    }
  };

  const score = result?.comprehensive?.overall_score ?? 0;
  const tamper = result?.comprehensive?.security_analysis?.tamper_evidence ?? [];

  return (
    <div className="flex flex-col gap-6 p-4 md:p-6">
      <div className="flex flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-lg font-semibold tracking-tight md:text-xl">
            Audit provenance
          </h1>
          <HoverCard openDelay={200} closeDelay={80}>
            <HoverCardTrigger asChild>
              <button
                type="button"
                className="rounded-full p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                aria-label="How audit works"
              >
                <Info className="size-4" aria-hidden />
              </button>
            </HoverCardTrigger>
            <HoverCardContent
              side="bottom"
              align="start"
              className="w-80 max-w-[min(20rem,calc(100vw-2rem))] space-y-3"
            >
              <div className="space-y-2">
                <p className="text-sm font-semibold leading-snug text-foreground">
                  Same checks as the CLI and agent API
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Verification uses the same pipeline as the CLI verify command and the public verify
                  endpoint. Equivalent calls:
                </p>
                <ul className="list-none space-y-1 border-l-2 border-border pl-3 text-xs font-mono leading-relaxed text-muted-foreground">
                  <li>af vc verify</li>
                  <li>POST /api/v1/did/verify-audit</li>
                  <li>POST /api/ui/v1/did/verify-audit</li>
                </ul>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  You can upload workflow VC audit exports, execution bundles, or a bare W3C
                  VerifiableCredential JSON document.
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Browser verification stays offline and uses bundled DID data only. Use the CLI if
                  you need remote DID resolution during verification.
                </p>
              </div>
            </HoverCardContent>
          </HoverCard>
        </div>
        <p className="text-sm text-muted-foreground">
          Upload the same JSON you exported from a run, or other provenance JSON you trust, to confirm
          signatures and spot tampering for audit.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="border-border/80 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Upload className="size-4 text-muted-foreground" aria-hidden />
              Document
            </CardTitle>
            <CardDescription>
              Drag and drop a <span className="font-medium">.json</span> file onto the area below, click
              to browse, or paste JSON.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,application/json"
              className="sr-only"
              aria-label="Browse for JSON file to audit"
              onChange={onFileInputChange}
            />
            <button
              type="button"
              className={cn(
                "w-full rounded-lg border border-dashed border-border/80 bg-muted/30 p-6 text-center text-sm text-muted-foreground",
                "transition-colors hover:border-primary/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              )}
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <span className="block font-medium text-foreground">Drop JSON here</span>
              <span className="mt-1 block text-xs">or click to browse</span>
            </button>
            <div className="space-y-2">
              <Label htmlFor="prov-json">JSON</Label>
              <textarea
                id="prov-json"
                value={jsonText}
                onChange={(e) => setJsonText(e.target.value)}
                placeholder='{ "workflow_id": "...", "component_vcs": [ ... ] }'
                spellCheck={false}
                className={cn(
                  "min-h-[220px] w-full rounded-md border border-input bg-transparent px-3 py-2 font-mono text-xs shadow-sm",
                  "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                )}
              />
            </div>
            <Alert>
              <AlertTitle>HTTP audit stays offline</AlertTitle>
              <AlertDescription>
                Browser verification uses bundled DID data only. If you need remote DID resolution,
                run <code className="text-micro">af vc verify --resolve-web</code> locally.
              </AlertDescription>
            </Alert>
            <Button
              type="button"
              onClick={() => void onVerify()}
              disabled={loading}
              className="w-full sm:w-auto"
            >
              {loading ? "Running audit…" : "Run audit"}
            </Button>
            {error && (
              <Alert variant="destructive">
                <AlertTriangle className="size-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        <Card className="border-border/80 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <FileCheck2 className="size-4 text-muted-foreground" aria-hidden />
              Result
            </CardTitle>
            <CardDescription>
              Cryptographic and integrity checks for audit and compliance review.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {!result && !loading && (
              <p className="text-sm text-muted-foreground">
                Run an audit to see issuer resolution, per-execution signature status, tamper signals,
                and scores.
              </p>
            )}
            {loading && (
              <p className="text-sm text-muted-foreground">Running audit…</p>
            )}
            {result && (
              <>
                <Alert variant={result.valid ? "default" : "destructive"}>
                  <AlertTitle className="flex flex-wrap items-center gap-2">
                    {result.valid ? "Audit passed" : "Audit failed"}
                    <Badge variant={result.valid ? "secondary" : "destructive"}>
                      {result.type}
                    </Badge>
                  </AlertTitle>
                  <AlertDescription className="space-y-2">
                    <p>{result.message}</p>
                    {result.workflow_id && (
                      <p className="text-xs text-muted-foreground">
                        Workflow <code className="rounded bg-muted px-1">{result.workflow_id}</code>
                      </p>
                    )}
                  </AlertDescription>
                </Alert>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Overall score</span>
                    <span className="font-medium tabular-nums">{score.toFixed(1)} / 100</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all",
                        result.valid ? "bg-primary" : "bg-destructive",
                      )}
                      style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
                    />
                  </div>
                </div>

                {tamper.length > 0 && (
                  <Alert variant="destructive">
                    <AlertTitle>Tamper evidence</AlertTitle>
                    <AlertDescription>
                      <ul className="list-inside list-disc text-sm">
                        {tamper.map((t) => (
                          <li key={t}>{t}</li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}

                <Separator />

                <div>
                  <h3 className="mb-2 text-sm font-medium">Executions</h3>
                  <div className="rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[100px]">Execution</TableHead>
                          <TableHead>VC</TableHead>
                          <TableHead className="w-[90px]">Valid</TableHead>
                          <TableHead className="w-[90px]">Sig</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {(result.component_results ?? []).map((row) => (
                          <TableRow key={`${row.vc_id}-${row.execution_id}`}>
                            <TableCell className="font-mono text-xs">
                              {row.execution_id ? `…${row.execution_id.slice(-8)}` : "—"}
                            </TableCell>
                            <TableCell className="max-w-[140px] truncate font-mono text-xs">
                              {row.vc_id}
                            </TableCell>
                            <TableCell>
                              <Badge variant={row.valid ? "secondary" : "destructive"}>
                                {row.valid ? "yes" : "no"}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <Badge
                                variant={row.signature_valid ? "outline" : "destructive"}
                              >
                                {row.signature_valid ? "ok" : "fail"}
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>

                {(result.did_resolutions ?? []).length > 0 && (
                  <>
                    <Separator />
                    <div>
                      <h3 className="mb-2 text-sm font-medium">DID resolution</h3>
                      <ul className="space-y-1 text-xs text-muted-foreground">
                        {result.did_resolutions!.map((d) => (
                          <li key={d.did} className="flex flex-wrap gap-x-2">
                            <code className="rounded bg-muted px-1">{d.did}</code>
                            <span>{d.success ? d.resolved_from : d.error}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </>
                )}

                {(result.comprehensive?.critical_issues?.length ?? 0) > 0 && (
                  <>
                    <Separator />
                    <div>
                      <h3 className="mb-2 text-sm font-medium text-destructive">
                        Critical issues
                      </h3>
                      <ul className="space-y-2 text-sm">
                        {result.comprehensive!.critical_issues.map((issue, i) => (
                          <li key={i} className="rounded-md border border-destructive/30 bg-destructive/5 p-2">
                            <span className="font-medium">{issue.type}</span>
                            <p className="text-muted-foreground">{issue.description}</p>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <p className="text-center text-xs text-muted-foreground">
        Tip: from a run, use{" "}
        <Link to="/runs" className="underline underline-offset-2">
          Runs
        </Link>{" "}
        → Export provenance → download, then audit here.
      </p>
    </div>
  );
}
