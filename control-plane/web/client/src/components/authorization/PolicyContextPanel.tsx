import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { statusTone } from "../../lib/theme";
import type { AccessPolicy } from "../../services/accessPoliciesApi";

interface PolicyContextPanelProps {
  tags: string[];
  policies: AccessPolicy[];
}

export function PolicyContextPanel({ tags, policies }: PolicyContextPanelProps) {
  const { asCaller, asTarget } = useMemo(() => {
    if (tags.length === 0) return { asCaller: [], asTarget: [] };

    const tagSet = new Set(tags);

    const asCaller = policies.filter((p) =>
      p.caller_tags.some((t) => tagSet.has(t))
    );
    const asTarget = policies.filter((p) =>
      p.target_tags.some((t) => tagSet.has(t))
    );

    return { asCaller, asTarget };
  }, [tags, policies]);

  if (tags.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Select at least one tag to see policy impact.
      </p>
    );
  }

  if (asCaller.length === 0 && asTarget.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No existing policies reference these tags.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {asCaller.length > 0 && (
        <div>
          <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-1.5">
            As Caller
          </h4>
          <div className="space-y-1.5">
            {asCaller.map((p) => (
              <PolicyRow key={p.id} policy={p} />
            ))}
          </div>
        </div>
      )}
      {asTarget.length > 0 && (
        <div>
          <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-1.5">
            As Target
          </h4>
          <div className="space-y-1.5">
            {asTarget.map((p) => (
              <PolicyRow key={p.id} policy={p} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PolicyRow({ policy }: { policy: AccessPolicy }) {
  const functions = [
    ...(policy.allow_functions || []).map((f) => `${f}`),
    ...(policy.deny_functions || []).map((f) => `!${f}`),
  ];

  return (
    <div className="flex items-start gap-2 rounded-md border p-2 text-sm">
      <Badge
        variant="outline"
        size="sm"
        showIcon={false}
        className={cn(
          "shrink-0 uppercase tracking-wide",
          policy.action === "allow"
            ? statusTone.success.border
            : statusTone.error.border
        )}
      >
        {policy.action}
      </Badge>
      <div className="min-w-0 flex-1">
        <span className="font-medium">{policy.name}</span>
        <span className="text-muted-foreground">
          {" "}
          {policy.caller_tags.join(", ")} &rarr; {policy.target_tags.join(", ")}
        </span>
        {functions.length > 0 && (
          <div className="text-xs text-muted-foreground mt-0.5">
            Functions: {functions.join(", ")}
          </div>
        )}
      </div>
    </div>
  );
}
