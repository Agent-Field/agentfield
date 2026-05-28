import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown } from "@/components/ui/icon-bridge";
import { Copy, Check } from "lucide-react";
import { useState } from "react";

function ProvenanceRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  const [copied, setCopied] = useState(false);
  const short =
    value.length > 40 ? `${value.slice(0, 16)}…${value.slice(-12)}` : value;

  return (
    <div className="flex min-w-0 items-center gap-2 py-1 first:pt-0 last:pb-0">
      <span className="w-[7rem] shrink-0 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <code
        className="min-w-0 flex-1 truncate text-xs font-mono leading-tight text-foreground/90"
        title={value}
      >
        {short}
      </code>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        className="shrink-0 text-muted-foreground hover:text-foreground [&_svg]:size-3.5"
        aria-label={`Copy ${label}`}
        onClick={() => {
          void navigator.clipboard.writeText(value).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          });
        }}
      >
        {copied ? <Check /> : <Copy />}
      </Button>
    </div>
  );
}

/** Collapsible VC provenance: caller/target DIDs and content hashes (secondary to I/O). */
export function StepProvenanceCard({
  callerDid,
  targetDid,
  inputHash,
  outputHash,
}: {
  callerDid?: string;
  targetDid?: string;
  inputHash?: string;
  outputHash?: string;
}) {
  const rows: { label: string; value: string }[] = [];
  if (callerDid?.trim()) rows.push({ label: "Caller DID", value: callerDid.trim() });
  if (targetDid?.trim()) rows.push({ label: "Target DID", value: targetDid.trim() });
  if (inputHash?.trim()) rows.push({ label: "Input hash", value: inputHash.trim() });
  if (outputHash?.trim()) rows.push({ label: "Output hash", value: outputHash.trim() });

  if (rows.length === 0) return null;

  return (
    <Collapsible defaultOpen={false}>
      <CollapsibleTrigger className="flex min-w-0 w-full items-center gap-1 text-left text-xs font-medium text-muted-foreground transition-colors hover:text-foreground">
        <ChevronDown className="size-3 shrink-0 transition-transform [[data-state=open]_&]:rotate-0 [[data-state=closed]_&]:-rotate-90" />
        Provenance (VC)
        <span className="ml-1 font-normal text-muted-foreground/70">
          ({rows.length})
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-2 min-w-0 max-w-full divide-y divide-border/50 rounded-md border border-border/60 bg-muted/25 px-2.5 py-1.5">
          {rows.map((r) => (
            <ProvenanceRow key={r.label} label={r.label} value={r.value} />
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
