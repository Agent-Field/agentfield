import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { HintIcon } from "@/components/authorization/HintIcon";
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
import { SearchBar } from "@/components/ui/SearchBar";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  useSuccessNotification,
  useErrorNotification,
} from "@/components/ui/notification";
import * as tagApprovalApi from "@/services/tagApprovalApi";
import type { AgentTagSummary } from "@/services/tagApprovalApi";
import type { AccessPolicy } from "@/services/accessPoliciesApi";
import { TooltipTagList } from "@/components/ui/tooltip-tag-list";
import { ApproveWithContextDialog } from "./ApproveWithContextDialog";
import { RevokeDialog } from "./RevokeDialog";
import { formatRelativeTime } from "@/utils/dateFormat";
import { getAgentTagRowStatus } from "@/lib/governanceUtils";

const MAX_VISIBLE_TAGS = 2;

function renderTagCell(tags: string[]) {
  if (!tags.length) {
    return <span className="text-xs italic text-muted-foreground">—</span>;
  }
  const visible = tags.slice(0, MAX_VISIBLE_TAGS);
  const overflow = tags.length - MAX_VISIBLE_TAGS;
  const content = (
    <div className="flex flex-wrap items-center gap-1">
      {visible.map((tag) => (
        <Badge key={tag} variant="secondary" size="sm" showIcon={false} className="max-w-[96px] truncate">
          {tag}
        </Badge>
      ))}
      {overflow > 0 && (
        <Badge variant="count" size="sm">
          +{overflow}
        </Badge>
      )}
    </div>
  );
  if (overflow > 0 || tags.some((t) => t.length > 15)) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="cursor-default">{content}</div>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <TooltipTagList groups={[{ tags }]} />
        </TooltipContent>
      </Tooltip>
    );
  }
  return content;
}

function SortableHead({
  label,
  column,
  sortBy,
  sortOrder,
  onSort,
}: {
  label: string;
  column: string;
  sortBy: string;
  sortOrder: "asc" | "desc";
  onSort: (field: string) => void;
}) {
  const active = sortBy === column;
  return (
    <TableHead className="h-9 px-3 text-micro-plus font-medium text-muted-foreground">
      <button
        type="button"
        onClick={() => onSort(column)}
        className="inline-flex items-center gap-1 hover:text-foreground"
      >
        {label}
        {active ? (
          sortOrder === "asc" ? (
            <ArrowUp className="size-3 opacity-70" />
          ) : (
            <ArrowDown className="size-3 opacity-70" />
          )
        ) : null}
      </button>
    </TableHead>
  );
}

export interface AgentTagsTabProps {
  policies: AccessPolicy[];
  agents: AgentTagSummary[];
  agentsLoading: boolean;
  agentsError?: Error | null;
  canMutate: boolean;
  onRefresh: () => void;
}

export function AgentTagsTab({
  policies,
  agents,
  agentsLoading,
  agentsError,
  canMutate,
  onRefresh,
}: AgentTagsTabProps) {
  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();

  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortBy, setSortBy] = useState("registered_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  const [approveAgent, setApproveAgent] = useState<AgentTagSummary | null>(null);
  const [rejectAgent, setRejectAgent] = useState<AgentTagSummary | null>(null);
  const [revokeAgent, setRevokeAgent] = useState<AgentTagSummary | null>(null);
  const [rejectReason, setRejectReason] = useState("");
  const [rejectLoading, setRejectLoading] = useState(false);

  const handleSortChange = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(field);
      setSortOrder("desc");
    }
  };

  const statusCounts = useMemo(() => {
    const pending = agents.filter((a) => getAgentTagRowStatus(a) === "pending_approval").length;
    const approved = agents.filter((a) => getAgentTagRowStatus(a) === "active").length;
    const other = agents.length - pending - approved;
    return { pending, approved, other };
  }, [agents]);

  const filteredAgents = useMemo(() => {
    let result = agents;
    if (statusFilter !== "all") {
      result = result.filter((a) => {
        const s = getAgentTagRowStatus(a);
        if (statusFilter === "pending") return s === "pending_approval";
        if (statusFilter === "approved") return s === "active";
        if (statusFilter === "other") return s === "other";
        return true;
      });
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (a) =>
          a.agent_id.toLowerCase().includes(q) ||
          (a.proposed_tags || []).some((t) => t.toLowerCase().includes(q)) ||
          (a.approved_tags || []).some((t) => t.toLowerCase().includes(q)),
      );
    }
    const sorted = [...result].sort((a, b) => {
      let cmp = 0;
      switch (sortBy) {
        case "agent_id":
          cmp = a.agent_id.localeCompare(b.agent_id);
          break;
        case "registered_at":
          cmp = new Date(a.registered_at).getTime() - new Date(b.registered_at).getTime();
          break;
        default:
          cmp = 0;
      }
      return sortOrder === "asc" ? cmp : -cmp;
    });
    return sorted;
  }, [agents, statusFilter, searchQuery, sortBy, sortOrder]);

  const handleApprove = async (agentId: string, selectedTags: string[]) => {
    try {
      await tagApprovalApi.approveAgentTags(agentId, {
        approved_tags: selectedTags,
      });
      showSuccess(`Tags approved for ${agentId}`);
      onRefresh();
    } catch (err: unknown) {
      showError("Failed to approve tags", err instanceof Error ? err.message : undefined);
      throw err;
    }
  };

  const handleReject = async () => {
    if (!rejectAgent) return;
    try {
      setRejectLoading(true);
      await tagApprovalApi.rejectAgentTags(rejectAgent.agent_id, {
        reason: rejectReason || undefined,
      });
      showSuccess(`Tags rejected for ${rejectAgent.agent_id}`);
      setRejectAgent(null);
      setRejectReason("");
      onRefresh();
    } catch (err: unknown) {
      showError("Failed to reject tags", err instanceof Error ? err.message : undefined);
    } finally {
      setRejectLoading(false);
    }
  };

  const handleRevoke = async (agentId: string, reason?: string) => {
    try {
      await tagApprovalApi.revokeAgentTags(agentId, reason);
      showSuccess(`Tags revoked for ${agentId}`);
      onRefresh();
    } catch (err: unknown) {
      showError("Failed to revoke tags", err instanceof Error ? err.message : undefined);
      throw err;
    }
  };

  if (agentsError) {
    return (
      <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive">
        {agentsError instanceof Error ? agentsError.message : "Failed to load agents"}
      </div>
    );
  }

  const readOnly = !canMutate;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0 flex-1 lg:max-w-md">
          <SearchBar
            value={searchQuery}
            onChange={setSearchQuery}
            placeholder="Search agent id or tags…"
            size="sm"
            wrapperClassName="w-full"
            inputClassName="border-border/80 bg-background shadow-sm"
          />
        </div>
        <div className="flex flex-wrap items-center gap-3 lg:justify-end">
          {readOnly && (
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <span>View only</span>
              <HintIcon label="Why view only">
                Approvals need admin APIs enabled on the server. Browsing tags still works.
              </HintIcon>
            </div>
          )}
        <Tabs value={statusFilter} onValueChange={setStatusFilter} className="w-full lg:w-auto">
          <TabsList variant="segmented" density="cosy" className="h-9 flex-wrap">
            <TabsTrigger variant="segmented" size="sm" value="all" className="gap-1">
              All
              <span className="tabular-nums text-micro text-muted-foreground">
                {agents.length}
              </span>
            </TabsTrigger>
            <TabsTrigger variant="segmented" size="sm" value="pending" className="gap-1">
              Pending
              <span className="tabular-nums text-micro text-muted-foreground">
                {statusCounts.pending}
              </span>
            </TabsTrigger>
            <TabsTrigger variant="segmented" size="sm" value="approved" className="gap-1">
              Approved
              <span className="tabular-nums text-micro text-muted-foreground">
                {statusCounts.approved}
              </span>
            </TabsTrigger>
            <TabsTrigger variant="segmented" size="sm" value="other" className="gap-1">
              Other
              <span className="tabular-nums text-micro text-muted-foreground">
                {statusCounts.other}
              </span>
            </TabsTrigger>
          </TabsList>
        </Tabs>
        </div>
      </div>

      <TooltipProvider delayDuration={300}>
        <div className="rounded-lg border border-border bg-card">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <SortableHead
                  label="Agent"
                  column="agent_id"
                  sortBy={sortBy}
                  sortOrder={sortOrder}
                  onSort={handleSortChange}
                />
                <TableHead className="h-9 px-3 text-micro-plus font-medium text-muted-foreground">
                  Requested
                </TableHead>
                <TableHead className="h-9 px-3 text-micro-plus font-medium text-muted-foreground">
                  Granted
                </TableHead>
                <TableHead className="h-9 w-28 px-3 text-center text-micro-plus font-medium text-muted-foreground">
                  Status
                </TableHead>
                <SortableHead
                  label="Registered"
                  column="registered_at"
                  sortBy={sortBy}
                  sortOrder={sortOrder}
                  onSort={handleSortChange}
                />
                {!readOnly && (
                  <TableHead className="h-9 w-36 px-3 text-right text-micro-plus font-medium text-muted-foreground">
                    Actions
                  </TableHead>
                )}
              </TableRow>
            </TableHeader>
            <TableBody>
              {agentsLoading ? (
                <TableRow>
                  <TableCell
                    colSpan={readOnly ? 5 : 6}
                    className="p-8 text-center text-muted-foreground"
                  >
                    Loading agents…
                  </TableCell>
                </TableRow>
              ) : filteredAgents.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={readOnly ? 5 : 6} className="p-8">
                    <div className="flex flex-col items-center justify-center py-6 text-center">
                      <p className="text-sm font-medium text-muted-foreground">No agents</p>
                      <p className="mt-1 max-w-sm text-xs text-muted-foreground">
                        {agents.length === 0
                          ? "Registered agents with tag metadata will appear here."
                          : "No rows match the current filters."}
                      </p>
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                filteredAgents.map((item) => {
                  const status = getAgentTagRowStatus(item);
                  return (
                    <TableRow key={item.agent_id}>
                      <TableCell className="px-3 py-2 font-mono text-xs font-medium align-top">
                        {item.agent_id}
                      </TableCell>
                      <TableCell className="min-w-0 px-3 py-2 align-top">
                        {renderTagCell(item.proposed_tags || [])}
                      </TableCell>
                      <TableCell className="min-w-0 px-3 py-2 align-top">
                        {renderTagCell(item.approved_tags || [])}
                      </TableCell>
                      <TableCell className="px-3 py-2 text-center align-top">
                        {status === "pending_approval" && (
                          <Badge variant="pending" size="sm" showIcon={false}>
                            Pending
                          </Badge>
                        )}
                        {status === "active" && (
                          <Badge variant="success" size="sm" showIcon={false}>
                            Approved
                          </Badge>
                        )}
                        {status === "other" && (
                          <Badge variant="secondary" size="sm" showIcon={false}>
                            Other
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="px-3 py-2 align-top">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="cursor-default whitespace-nowrap text-muted-foreground">
                              {formatRelativeTime(item.registered_at)}
                            </span>
                          </TooltipTrigger>
                          <TooltipContent side="top">
                            {new Date(item.registered_at).toLocaleString()}
                          </TooltipContent>
                        </Tooltip>
                      </TableCell>
                      {!readOnly && (
                        <TableCell className="px-3 py-2 text-right align-top">
                          {status === "pending_approval" && (
                            <div className="flex justify-end gap-0.5">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 text-xs"
                                onClick={() => setApproveAgent(item)}
                              >
                                Approve
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 text-xs"
                                onClick={() => setRejectAgent(item)}
                              >
                                Reject
                              </Button>
                            </div>
                          )}
                          {status === "active" && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs"
                              onClick={() => setRevokeAgent(item)}
                            >
                              Revoke
                            </Button>
                          )}
                        </TableCell>
                      )}
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>
      </TooltipProvider>

      <ApproveWithContextDialog
        agent={approveAgent}
        policies={policies}
        onApprove={handleApprove}
        onOpenChange={(open) => !open && setApproveAgent(null)}
      />

      <Dialog open={!!rejectAgent} onOpenChange={(open) => !open && setRejectAgent(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject tags</DialogTitle>
            <DialogDescription>
              Reject proposed tags for{" "}
              <span className="font-mono font-medium">{rejectAgent?.agent_id}</span>. The
              agent may be set offline depending on server rules.
            </DialogDescription>
          </DialogHeader>
          <div className="py-2">
            <Label htmlFor="reject-reason">Reason (optional)</Label>
            <Input
              id="reject-reason"
              placeholder="Reason for rejection"
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              className="mt-1.5"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRejectAgent(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleReject} disabled={rejectLoading}>
              {rejectLoading ? "Rejecting…" : "Reject"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <RevokeDialog
        agent={revokeAgent}
        onRevoke={handleRevoke}
        onOpenChange={(open) => !open && setRevokeAgent(null)}
      />
    </div>
  );
}
