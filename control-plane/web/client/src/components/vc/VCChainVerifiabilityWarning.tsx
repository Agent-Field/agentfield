import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { AlertTriangle, ChevronDown } from "@/components/ui/icon-bridge";
import { statusTone } from "@/lib/theme";
import { cn } from "@/lib/utils";
import type { WorkflowVCChainResponse } from "@/types/did";

const MAX_LISTED_DIDS = 10;
const MAX_DID_CHARS = 64;

function shortenDid(did: string): string {
  if (did.length <= MAX_DID_CHARS) return did;
  return `${did.slice(0, 40)}…${did.slice(-16)}`;
}

export interface VCChainVerifiabilityWarningProps {
  chain?: Pick<WorkflowVCChainResponse, "verifiable" | "unresolved_dids"> | null;
  className?: string;
}

/**
 * Warns that a workflow's VC chain cannot be fully verified because the
 * control plane could not resolve some DIDs. Renders nothing unless the
 * server explicitly reports `verifiable: false` (older servers omit it).
 */
export function VCChainVerifiabilityWarning({
  chain,
  className,
}: VCChainVerifiabilityWarningProps) {
  if (chain?.verifiable !== false) return null;

  const dids = (chain.unresolved_dids ?? []).filter((did) => did.trim() !== "");
  const count = dids.length;
  const listed = dids.slice(0, MAX_LISTED_DIDS);
  const hidden = count - listed.length;

  return (
    <div
      role="alert"
      className={cn(
        "flex min-w-0 items-start gap-2 rounded-lg px-3 py-2 text-xs",
        statusTone.warning.bg,
        statusTone.warning.border,
        className,
      )}
    >
      <AlertTriangle
        className={cn("mt-0.5 size-3.5 shrink-0", statusTone.warning.accent)}
        aria-hidden
      />
      <div className="min-w-0 flex-1">
        <p className={cn("m-0 font-medium leading-snug", statusTone.warning.fg)}>
          {count > 0
            ? `This audit chain cannot be fully verified: ${count} ${count === 1 ? "DID" : "DIDs"} could not be resolved`
            : "This audit chain cannot be fully verified: some DIDs could not be resolved"}
        </p>
        {count > 0 ? (
          <Collapsible defaultOpen={false}>
            <CollapsibleTrigger className="mt-1 flex items-center gap-1 text-left text-micro-plus font-medium text-muted-foreground transition-colors hover:text-foreground">
              <ChevronDown className="size-3 shrink-0 transition-transform [[data-state=open]_&]:rotate-0 [[data-state=closed]_&]:-rotate-90" />
              Unresolved DIDs
            </CollapsibleTrigger>
            <CollapsibleContent>
              <ul className="mt-1 min-w-0 space-y-0.5 pl-4">
                {listed.map((did) => (
                  <li
                    key={did}
                    className="truncate font-mono text-micro-plus text-foreground"
                    title={did}
                  >
                    {shortenDid(did)}
                  </li>
                ))}
                {hidden > 0 ? (
                  <li className="text-micro-plus text-muted-foreground">
                    +{hidden} more
                  </li>
                ) : null}
              </ul>
            </CollapsibleContent>
          </Collapsible>
        ) : null}
      </div>
    </div>
  );
}
