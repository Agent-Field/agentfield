import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { CompactTable } from "@/components/ui/CompactTable";
import { Badge } from "@/components/ui/badge";
import { FastTableSearch } from "@/components/ui/FastTableSearch";
import { SegmentedStatusFilter } from "@/components/ui/segmented-status-filter";
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
import * as tagApprovalApi from "../../services/tagApprovalApi";
import type { AgentTagSummary } from "../../services/tagApprovalApi";
import type { AccessPolicy } from "../../services/accessPoliciesApi";
import { TooltipTagList } from "@/components/ui/tooltip-tag-list";
import { ApproveWithContextDialog } from "./ApproveWithContextDialog";
import { RevokeDialog } from "./RevokeDialog";
import { formatRelativeTime } from "../../utils/dateFormat";

const GRID_TEMPLATE =
  "minmax(120px,1fr) minmax(100px,1fr) minmax(100px,1fr) 86px 120px 200px";

const MAX_VISIBLE_TAGS = 2;

function renderTagCell(tags: string[]) {
  if (!tags.length)
    return (
      <span className="text-xs text-muted-foreground italic">&mdash;</span>
    );
  const visible = tags.slice(0, MAX_VISIBLE_TAGS);
  const overflow = tags.length - MAX_VISIBLE_TAGS;
  const content = (
    <div className="flex items-center gap-1 overflow-hidden">
      {visible.map((tag) => (
        <Badge
          key={tag}
          variant="secondary"
          size="sm"
          showIcon={false}
          className="truncate max-w-[96px]"
        >
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

type TagStatus = "pending_approval" | "active" | "other";

function getTagStatus(agent: AgentTagSummary): TagStatus {
  if (agent.lifecycle_status === "pending_approval") return "pending_approval";
  if (
    agent.lifecycle_status === "active" ||
    agent.lifecycle_status === "online" ||
    agent.lifecycle_status === "ready" ||
    agent.lifecycle_status === "offline" ||
    agent.lifecycle_status === "degraded" ||
    agent.lifecycle_status === "starting"
  )
    return "active";
  return "other";
}

interface AgentTagsTabProps {
  policies: AccessPolicy[];
  onPendingCountChange: (count: number) => void;
  refreshKey?: number;
}

export function AgentTagsTab({
  policies,
  onPendingCountChange,
  refreshKey,
}: AgentTagsTabProps) {
  const [agents, setAgents] = useState<AgentTagSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();

  // Stable ref for showError to avoid infinite re-render loop in useCallback
  const showErrorRef = useRef(showError);
  showErrorRef.current = showError;

  // Search & filter
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  // Sort
  const [sortBy, setSortBy] = useState("registered_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  // Dialogs
  const [approveAgent, setApproveAgent] = useState<AgentTagSummary | null>(
    null
  );
  const [rejectAgent, setRejectAgent] = useState<AgentTagSummary | null>(null);
  const [revokeAgent, setRevokeAgent] = useState<AgentTagSummary | null>(null);
  const [rejectReason, setRejectReason] = useState("");
  const [rejectLoading, setRejectLoading] = useState(false);

  const fetchAgents = useCallback(async () => {
    try {
      setLoading(true);
      const data = await tagApprovalApi.listAllAgentsWithTags();
      setAgents(data.agents || []);
    } catch (err: unknown) {
      showErrorRef.current(
        "Failed to fetch agents",
        err instanceof Error ? err.message : undefined
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  // Re-fetch when parent triggers a refresh
  useEffect(() => {
    if (refreshKey !== undefined && refreshKey > 0) {
      fetchAgents();
    }
  }, [refreshKey, fetchAgents]);

  // Update pending count
  useEffect(() => {
    const pending = agents.filter(
      (a) => getTagStatus(a) === "pending_approval"
    ).length;
    onPendingCountChange(pending);
  }, [agents, onPendingCountChange]);

  const handleApprove = async (agentId: string, selectedTags: string[]) => {
    try {
      await tagApprovalApi.approveAgentTags(agentId, {
        approved_tags: selectedTags,
      });
      showSuccess(`Tags approved for agent ${agentId}`);
      fetchAgents();
    } catch (err: unknown) {
      showError(
        "Failed to approve tags",
        err instanceof Error ? err.message : undefined
      );
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
      showSuccess(`Tags rejected for agent ${rejectAgent.agent_id}`);
      setRejectAgent(null);
      setRejectReason("");
      fetchAgents();
    } catch (err: unknown) {
      showError(
        "Failed to reject tags",
        err instanceof Error ? err.message : undefined
      );
    } finally {
      setRejectLoading(false);
    }
  };

  const handleRevoke = async (agentId: string, reason?: string) => {
    try {
      await tagApprovalApi.revokeAgentTags(agentId, reason);
      showSuccess(`Tags revoked for agent ${agentId}`);
      fetchAgents();
    } catch (err: unknown) {
      showError(
        "Failed to revoke tags",
        err instanceof Error ? err.message : undefined
      );
      throw err;
    }
  };

  const handleSortChange = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(field);
      setSortOrder("desc");
    }
  };

  // Status counts
  const statusCounts = useMemo(() => {
    const pending = agents.filter(
      (a) => getTagStatus(a) === "pending_approval"
    ).length;
    const approved = agents.filter(
      (a) => getTagStatus(a) === "active"
    ).length;
    const other = agents.length - pending - approved;
    return { pending, approved, other };
  }, [agents]);

  // Filtered + sorted data
  const filteredAgents = useMemo(() => {
    let result = agents;

    // Status filter
    if (statusFilter !== "all") {
      result = result.filter((a) => {
        const s = getTagStatus(a);
        if (statusFilter === "pending") return s === "pending_approval";
        if (statusFilter === "approved") return s === "active";
        if (statusFilter === "other") return s === "other";
        return true;
      });
    }

    // Search
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (a) =>
          a.agent_id.toLowerCase().includes(q) ||
          (a.proposed_tags || []).some((t) => t.toLowerCase().includes(q)) ||
          (a.approved_tags || []).some((t) => t.toLowerCase().includes(q))
      );
    }

    // Sort
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

  const statusOptions = useMemo(
    () => [
      { value: "all", label: "All", count: agents.length },
      { value: "pending", label: "Pending", count: statusCounts.pending },
      { value: "approved", label: "Approved", count: statusCounts.approved },
      { value: "other", label: "Other", count: statusCounts.other },
    ],
    [agents.length, statusCounts]
  );

  const columns = [
    {
      key: "agent_id",
      header: "Agent ID",
      sortable: true,
      align: "left" as const,
      render: (item: AgentTagSummary) => (
        <span className="font-mono text-sm font-medium truncate block">
          {item.agent_id}
        </span>
      ),
    },
    {
      key: "proposed_tags",
      header: "Requested",
      sortable: false,
      align: "left" as const,
      render: (item: AgentTagSummary) =>
        renderTagCell(item.proposed_tags || []),
    },
    {
      key: "approved_tags",
      header: "Granted",
      sortable: false,
      align: "left" as const,
      render: (item: AgentTagSummary) =>
        renderTagCell(item.approved_tags || []),
    },
    {
      key: "status",
      header: "Status",
      sortable: false,
      align: "center" as const,
      render: (item: AgentTagSummary) => {
        const status = getTagStatus(item);
        switch (status) {
          case "pending_approval":
            return <Badge variant="secondary" size="sm" showIcon={false}>Pending</Badge>;
          case "active":
            return <Badge variant="secondary" size="sm" showIcon={false}>Approved</Badge>;
          default:
            return <Badge variant="secondary" size="sm" showIcon={false}>{status}</Badge>;
        }
      },
    },
    {
      key: "registered_at",
      header: "Registered",
      sortable: true,
      align: "left" as const,
      render: (item: AgentTagSummary) => (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="text-xs text-muted-foreground cursor-default whitespace-nowrap">
              {formatRelativeTime(item.registered_at)}
            </span>
          </TooltipTrigger>
          <TooltipContent side="top">
            {new Date(item.registered_at).toLocaleString()}
          </TooltipContent>
        </Tooltip>
      ),
    },
    {
      key: "actions",
      header: "Actions",
      sortable: false,
      align: "right" as const,
      render: (item: AgentTagSummary) => {
        const status = getTagStatus(item);
        if (status === "pending_approval") {
          return (
            <div className="flex items-center gap-1 whitespace-nowrap">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={(e: React.MouseEvent) => {
                  e.stopPropagation();
                  setApproveAgent(item);
                }}
              >
                Approve
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={(e: React.MouseEvent) => {
                  e.stopPropagation();
                  setRejectAgent(item);
                }}
              >
                Reject
              </Button>
            </div>
          );
        }
        if (status === "active") {
          return (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                setRevokeAgent(item);
              }}
            >
              Revoke
            </Button>
          );
        }
        return null;
      },
    },
  ];

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      {/* Toolbar */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex-1 min-w-[200px] max-w-sm">
          <FastTableSearch
            onSearch={setSearchQuery}
            placeholder="Search agents or tags..."
            resultCount={filteredAgents.length}
            totalCount={agents.length}
          />
        </div>
        <SegmentedStatusFilter
          value={statusFilter}
          onChange={setStatusFilter}
          options={statusOptions}
        />
      </div>

      {/* Table */}
      <div className="flex-1 min-h-0">
        <TooltipProvider>
          <CompactTable
            data={filteredAgents}
            columns={columns}
            loading={loading}
            hasMore={false}
            isFetchingMore={false}
            sortBy={sortBy}
            sortOrder={sortOrder}
            onSortChange={handleSortChange}
            gridTemplate={GRID_TEMPLATE}
            getRowKey={(item) => item.agent_id}
            emptyState={{
              title: "No agents",
              description:
                "Agents with tags will appear here once they register with the control plane.",
            }}
          />
        </TooltipProvider>
      </div>

      {/* Approve With Context Dialog */}
      <ApproveWithContextDialog
        agent={approveAgent}
        policies={policies}
        onApprove={handleApprove}
        onOpenChange={(open) => !open && setApproveAgent(null)}
      />

      {/* Reject Dialog */}
      <Dialog
        open={!!rejectAgent}
        onOpenChange={(open) => !open && setRejectAgent(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject Tags</DialogTitle>
            <DialogDescription>
              Reject the proposed tags for agent{" "}
              <span className="font-mono font-medium">
                {rejectAgent?.agent_id}
              </span>
              . The agent will be set to offline status.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label htmlFor="reject-reason">Reason (optional)</Label>
            <Input
              id="reject-reason"
              placeholder="Enter reason for rejection"
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              className="mt-1.5"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRejectAgent(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleReject}
              disabled={rejectLoading}
            >
              {rejectLoading ? "Rejecting..." : "Reject Tags"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Revoke Dialog */}
      <RevokeDialog
        agent={revokeAgent}
        onRevoke={handleRevoke}
        onOpenChange={(open) => !open && setRevokeAgent(null)}
      />
    </div>
  );
}
