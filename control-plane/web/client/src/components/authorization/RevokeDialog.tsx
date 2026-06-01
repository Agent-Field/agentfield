import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import type { AgentTagSummary } from "../../services/tagApprovalApi";

interface RevokeDialogProps {
  agent: AgentTagSummary | null;
  onRevoke: (agentId: string, reason?: string) => Promise<void>;
  onOpenChange: (open: boolean) => void;
}

export function RevokeDialog({ agent, onRevoke, onOpenChange }: RevokeDialogProps) {
  const [reason, setReason] = useState("");
  const [loading, setLoading] = useState(false);

  const handleRevoke = async () => {
    if (!agent) return;
    try {
      setLoading(true);
      await onRevoke(agent.agent_id, reason || undefined);
      onOpenChange(false);
      setReason("");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={!!agent} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Revoke Tags</DialogTitle>
          <DialogDescription>
            Revoke approved tags for agent{" "}
            <span className="font-mono font-medium">{agent?.agent_id}</span>.
            This will set the agent's status to pending.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          {agent && (agent.approved_tags?.length ?? 0) > 0 && (
            <div>
              <Label className="mb-2 block">Current Approved Tags</Label>
              <div className="flex flex-wrap gap-1.5">
                {(agent.approved_tags || []).map((tag) => (
                  <Badge
                    key={tag}
                    variant="secondary"
                    showIcon={false}
                    size="sm"
                  >
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          <div>
            <Label htmlFor="revoke-reason">Reason (optional)</Label>
            <Input
              id="revoke-reason"
              placeholder="Enter reason for revocation"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="mt-1.5"
            />
          </div>
          <p className="text-sm text-muted-foreground">
            This will revoke the agent's tag VC and set its status to pending.
          </p>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={handleRevoke} disabled={loading}>
            {loading ? "Revoking..." : "Revoke Tags"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
