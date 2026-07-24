import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Plus, Trash } from "lucide-react";
import { HintIcon } from "@/components/authorization/HintIcon";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "@/components/ui/icon-bridge";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { statusTone } from "@/lib/theme";
import { cn } from "@/lib/utils";
import * as policiesApi from "@/services/accessPoliciesApi";
import type { AccessPolicy, AccessPolicyRequest } from "@/services/accessPoliciesApi";
import { TooltipTagList } from "@/components/ui/tooltip-tag-list";
import { PolicyFormDialog } from "./PolicyFormDialog";

const MAX_VISIBLE_TAGS = 2;

function renderRuleCell(callerTags: string[], targetTags: string[]) {
  const hasOverflow =
    callerTags.length > MAX_VISIBLE_TAGS ||
    targetTags.length > MAX_VISIBLE_TAGS ||
    callerTags.some((t) => t.length > 15) ||
    targetTags.some((t) => t.length > 15);

  const renderSide = (tags: string[]) => {
    if (!tags.length) {
      return (
        <Badge variant="secondary" size="sm" showIcon={false} className="italic opacity-60">
          *
        </Badge>
      );
    }
    const visible = tags.slice(0, MAX_VISIBLE_TAGS);
    const overflow = tags.length - MAX_VISIBLE_TAGS;
    return (
      <>
        {visible.map((tag) => (
          <Badge
            key={tag}
            variant="secondary"
            size="sm"
            showIcon={false}
            className="max-w-[96px] truncate"
          >
            {tag}
          </Badge>
        ))}
        {overflow > 0 && (
          <Badge variant="count" size="sm">
            +{overflow}
          </Badge>
        )}
      </>
    );
  };

  const content = (
    <div className="flex min-w-0 items-center gap-1 overflow-hidden">
      {renderSide(callerTags)}
      <ArrowRight className="mx-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
      {renderSide(targetTags)}
    </div>
  );

  if (hasOverflow) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="min-w-0 cursor-default overflow-hidden">{content}</div>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-sm">
          <TooltipTagList
            groups={[
              { label: "From", tags: callerTags },
              { label: "To", tags: targetTags },
            ]}
          />
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

export interface AccessRulesTabProps {
  policies: AccessPolicy[];
  loading: boolean;
  onRefresh: () => void;
  canMutate: boolean;
  fetchError?: Error | null;
}

export function AccessRulesTab({
  policies,
  loading,
  onRefresh,
  canMutate,
  fetchError,
}: AccessRulesTabProps) {
  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();

  const [sortBy, setSortBy] = useState("priority");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [searchQuery, setSearchQuery] = useState("");
  const [actionFilter, setActionFilter] = useState("all");

  const [formOpen, setFormOpen] = useState(false);
  const [editPolicy, setEditPolicy] = useState<AccessPolicy | null>(null);
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);

  const handleSortChange = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(field);
      setSortOrder("desc");
    }
  };

  const filteredPolicies = useMemo(() => {
    let result = policies;
    if (actionFilter !== "all") {
      result = result.filter((p) => p.action === actionFilter);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          (p.description ?? "").toLowerCase().includes(q) ||
          p.caller_tags.some((t) => t.toLowerCase().includes(q)) ||
          p.target_tags.some((t) => t.toLowerCase().includes(q)) ||
          (p.allow_functions || []).some((f) => f.toLowerCase().includes(q)) ||
          (p.deny_functions || []).some((f) => f.toLowerCase().includes(q)),
      );
    }
    const sorted = [...result].sort((a, b) => {
      let cmp = 0;
      switch (sortBy) {
        case "name":
          cmp = a.name.localeCompare(b.name);
          break;
        case "priority":
          cmp = a.priority - b.priority;
          break;
        default:
          cmp = 0;
      }
      return sortOrder === "asc" ? cmp : -cmp;
    });
    return sorted;
  }, [policies, actionFilter, searchQuery, sortBy, sortOrder]);

  const actionCounts = useMemo(
    () => ({
      all: policies.length,
      allow: policies.filter((p) => p.action === "allow").length,
      deny: policies.filter((p) => p.action === "deny").length,
    }),
    [policies],
  );

  const openCreate = () => {
    setEditPolicy(null);
    setFormOpen(true);
  };

  const openEdit = (p: AccessPolicy) => {
    setEditPolicy(p);
    setFormOpen(true);
  };

  const handleSave = async (req: AccessPolicyRequest, editId?: number) => {
    try {
      if (editId) {
        await policiesApi.updatePolicy(editId, req);
        showSuccess(`Policy "${req.name}" updated`);
      } else {
        await policiesApi.createPolicy(req);
        showSuccess(`Policy "${req.name}" created`);
      }
      onRefresh();
    } catch (err: unknown) {
      showError("Failed to save policy", err instanceof Error ? err.message : undefined);
      throw err;
    }
  };

  const handleDelete = async () => {
    if (deleteId === null) return;
    try {
      setDeleting(true);
      await policiesApi.deletePolicy(deleteId);
      showSuccess("Policy deleted");
      setDeleteId(null);
      onRefresh();
    } catch (err: unknown) {
      showError("Failed to delete policy", err instanceof Error ? err.message : undefined);
    } finally {
      setDeleting(false);
    }
  };

  if (!canMutate) {
    return (
      <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
        <span>Policies unavailable</span>
        <HintIcon label="Why policies are unavailable">
          Enable authorization on the server and set the browser admin token on Access management.
          Agent tags may still load read-only.
        </HintIcon>
      </div>
    );
  }

  if (fetchError) {
    return (
      <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive">
        {fetchError instanceof Error ? fetchError.message : "Failed to load policies"}
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0 flex-1 lg:max-w-md">
          <SearchBar
            value={searchQuery}
            onChange={setSearchQuery}
            placeholder="Search policies, tags, functions…"
            size="sm"
            wrapperClassName="w-full"
            inputClassName="border-border/80 bg-background shadow-sm"
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Tabs
            value={actionFilter}
            onValueChange={setActionFilter}
            className="w-full sm:w-auto"
          >
            <TabsList variant="segmented" density="cosy" className="h-9 w-full sm:w-auto">
              <TabsTrigger variant="segmented" size="sm" value="all" className="gap-1">
                All
                <span className="tabular-nums text-micro text-muted-foreground">
                  {actionCounts.all}
                </span>
              </TabsTrigger>
              <TabsTrigger variant="segmented" size="sm" value="allow" className="gap-1">
                Allow
                <span className="tabular-nums text-micro text-muted-foreground">
                  {actionCounts.allow}
                </span>
              </TabsTrigger>
              <TabsTrigger variant="segmented" size="sm" value="deny" className="gap-1">
                Deny
                <span className="tabular-nums text-micro text-muted-foreground">
                  {actionCounts.deny}
                </span>
              </TabsTrigger>
            </TabsList>
          </Tabs>
          <Button size="sm" className="shrink-0 gap-1.5" onClick={openCreate}>
            <Plus className="size-3.5" />
            New policy
          </Button>
        </div>
      </div>

      <TooltipProvider delayDuration={300}>
        <div className="rounded-lg border border-border bg-card">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <SortableHead
                  label="Name"
                  column="name"
                  sortBy={sortBy}
                  sortOrder={sortOrder}
                  onSort={handleSortChange}
                />
                <TableHead className="h-9 px-3 text-micro-plus font-medium text-muted-foreground min-w-[12rem]">
                  Rule
                </TableHead>
                <TableHead className="h-9 w-24 px-3 text-center text-micro-plus font-medium text-muted-foreground">
                  Action
                </TableHead>
                <SortableHead
                  label="Priority"
                  column="priority"
                  sortBy={sortBy}
                  sortOrder={sortOrder}
                  onSort={handleSortChange}
                />
                <TableHead className="h-9 w-24 px-3 text-right text-micro-plus font-medium text-muted-foreground">
                  {/* actions */}
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={5} className="p-8 text-center text-muted-foreground">
                    Loading policies…
                  </TableCell>
                </TableRow>
              ) : filteredPolicies.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="p-8">
                    <div className="flex flex-col items-center justify-center py-6 text-center">
                      <p className="text-sm font-medium text-muted-foreground">
                        No policies match
                      </p>
                      <p className="mt-1 max-w-sm text-xs text-muted-foreground">
                        {policies.length === 0
                          ? "Create a policy to constrain cross-agent calls by caller and target tags."
                          : "Try another search or filter."}
                      </p>
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                filteredPolicies.map((item) => {
                  const needsNameTooltip =
                    item.name.length > 28 || (item.description ?? "").length > 48;
                  const nameCell = (
                    <div className="min-w-0">
                      <div className="truncate font-medium">{item.name}</div>
                      {item.description && (
                        <div className="mt-0.5 truncate text-micro-plus text-muted-foreground">
                          {item.description}
                        </div>
                      )}
                    </div>
                  );
                  return (
                    <TableRow key={item.id}>
                      <TableCell className="max-w-[14rem] px-3 py-2 align-top">
                        {needsNameTooltip ? (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <div className="min-w-0 cursor-default">{nameCell}</div>
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs" side="top">
                              <p className="text-xs font-medium">{item.name}</p>
                              {item.description && (
                                <p className="mt-1 text-xs text-muted-foreground">
                                  {item.description}
                                </p>
                              )}
                            </TooltipContent>
                          </Tooltip>
                        ) : (
                          nameCell
                        )}
                      </TableCell>
                      <TableCell className="min-w-0 px-3 py-2 align-top">
                        {renderRuleCell(item.caller_tags, item.target_tags)}
                      </TableCell>
                      <TableCell className="px-3 py-2 text-center align-top">
                        <Badge
                          variant="outline"
                          size="sm"
                          showIcon={false}
                          className={cn(
                            "uppercase tracking-wide",
                            item.action === "allow"
                              ? statusTone.success.border
                              : statusTone.error.border,
                          )}
                        >
                          {item.action}
                        </Badge>
                      </TableCell>
                      <TableCell className="px-3 py-2 text-center align-top">
                        <Badge variant="metadata" size="sm">
                          {item.priority}
                        </Badge>
                      </TableCell>
                      <TableCell className="px-3 py-2 text-right align-top">
                        <div className="flex justify-end gap-0.5">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs"
                            onClick={() => openEdit(item)}
                          >
                            Edit
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            className="text-status-error hover:bg-status-error-bg"
                            title="Delete policy"
                            onClick={() => setDeleteId(item.id)}
                          >
                            <Trash className="size-3.5" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>
      </TooltipProvider>

      <PolicyFormDialog
        open={formOpen}
        onOpenChange={setFormOpen}
        editPolicy={editPolicy}
        onSave={handleSave}
      />

      <Dialog open={deleteId !== null} onOpenChange={(open) => !open && setDeleteId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete policy</DialogTitle>
            <DialogDescription>
              This cannot be undone. Cross-agent calls may behave differently
              immediately after removal.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteId(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
              {deleting ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
