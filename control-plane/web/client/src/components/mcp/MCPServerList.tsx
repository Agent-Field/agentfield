import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useMemo, useState } from "react";

import { useMode } from "@/contexts/ModeContext";
import { cn } from "@/lib/utils";
import type { MCPServerAction, MCPServerHealthForUI } from "@/types/agentfield";
import {
  CheckmarkFilled,
  CircleDash,
  ErrorFilled,
  Play,
  Restart,
  Search,
  ServerProxy,
  Stop,
  WarningFilled,
} from "@/components/ui/icon-bridge";
import { MCPServerCard } from "./MCPServerCard";

interface MCPServerListProps {
  servers: MCPServerHealthForUI[];
  nodeId: string;
  onServerAction?: (
    action: MCPServerAction,
    serverAlias: string
  ) => Promise<void>;
  onBulkAction?: (
    action: MCPServerAction,
    serverAliases: string[]
  ) => Promise<void>;
  isLoading?: boolean;
  className?: string;
}

type ServerStatusFilter = "all" | "running" | "stopped" | "error" | "starting";

/**
 * Display list of all MCP servers for a node
 * Groups servers by status with search, filter, and bulk operations
 */
export function MCPServerList({
  servers,
  nodeId,
  onServerAction,
  onBulkAction,
  isLoading = false,
  className,
}: MCPServerListProps) {
  const { mode } = useMode();
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<ServerStatusFilter>("all");
  const [selectedServers, setSelectedServers] = useState<Set<string>>(
    new Set()
  );
  const [bulkActionLoading, setBulkActionLoading] =
    useState<MCPServerAction | null>(null);

  const isDeveloperMode = mode === "developer";

  // Filter and search servers
  const filteredServers = useMemo(() => {
    return servers.filter((server) => {
      // Status filter
      if (statusFilter !== "all" && server.status !== statusFilter) {
        return false;
      }

      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          server.alias.toLowerCase().includes(query) ||
          (server.error_message &&
            server.error_message.toLowerCase().includes(query))
        );
      }

      return true;
    });
  }, [servers, statusFilter, searchQuery]);

  // Group servers by status
  const serversByStatus = useMemo(() => {
    const groups = {
      running: [] as MCPServerHealthForUI[],
      starting: [] as MCPServerHealthForUI[],
      stopped: [] as MCPServerHealthForUI[],
      error: [] as MCPServerHealthForUI[],
      unknown: [] as MCPServerHealthForUI[],
    };

    filteredServers.forEach((server) => {
      if (server.status in groups) {
        groups[server.status as keyof typeof groups].push(server);
      } else {
        groups.unknown.push(server);
      }
    });

    return groups;
  }, [filteredServers]);

  // Status counts for filter badges
  const statusCounts = useMemo(() => {
    return servers.reduce((counts, server) => {
      counts[server.status] = (counts[server.status] || 0) + 1;
      return counts;
    }, {} as Record<string, number>);
  }, [servers]);

  const handleBulkAction = async (action: MCPServerAction) => {
    if (!onBulkAction || selectedServers.size === 0 || bulkActionLoading)
      return;

    try {
      setBulkActionLoading(action);
      await onBulkAction(action, Array.from(selectedServers));
      setSelectedServers(new Set());
    } catch (error) {
      console.error(`Failed to ${action} selected servers:`, error);
    } finally {
      setBulkActionLoading(null);
    }
  };

  const handleSelectAll = (status?: string) => {
    const serversToSelect = status
      ? servers.filter((s) => s.status === status)
      : filteredServers;

    const newSelected = new Set(selectedServers);
    const allSelected = serversToSelect.every((s) => newSelected.has(s.alias));

    if (allSelected) {
      // Deselect all
      serversToSelect.forEach((s) => newSelected.delete(s.alias));
    } else {
      // Select all
      serversToSelect.forEach((s) => newSelected.add(s.alias));
    }

    setSelectedServers(newSelected);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return CheckmarkFilled;
      case "starting":
        return WarningFilled;
      case "error":
        return ErrorFilled;
      case "stopped":
        return CircleDash;
      default:
        return WarningFilled;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running":
        return "text-green-600";
      case "starting":
        return "text-blue-600";
      case "error":
        return "text-red-600";
      case "stopped":
        return "text-gray-600";
      default:
        return "text-gray-500";
    }
  };

  const renderServerGroup = (
    status: string,
    servers: MCPServerHealthForUI[]
  ) => {
    if (servers.length === 0) return null;

    const StatusIcon = getStatusIcon(status);
    const statusColor = getStatusColor(status);
    const groupSelected = servers.every((s) => selectedServers.has(s.alias));
    const someSelected = servers.some((s) => selectedServers.has(s.alias));

    return (
      <div key={status} className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <StatusIcon className={cn("w-4 h-4", statusColor)} />
            <h3 className="font-medium capitalize">{status}</h3>
            <Badge variant="secondary" className="text-xs">
              {servers.length}
            </Badge>
          </div>

          {isDeveloperMode && servers.length > 1 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleSelectAll(status)}
              className="text-xs"
            >
              {groupSelected
                ? "Deselect All"
                : someSelected
                ? "Select All"
                : "Select All"}
            </Button>
          )}
        </div>

        <div className="grid gap-3">
          {servers.map((server) => (
            <div key={server.alias} className="relative">
              {isDeveloperMode && (
                <div className="absolute top-3 left-3 z-10">
                  <input
                    type="checkbox"
                    checked={selectedServers.has(server.alias)}
                    onChange={(e) => {
                      const newSelected = new Set(selectedServers);
                      if (e.target.checked) {
                        newSelected.add(server.alias);
                      } else {
                        newSelected.delete(server.alias);
                      }
                      setSelectedServers(newSelected);
                    }}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                </div>
              )}
              <MCPServerCard
                server={server}
                nodeId={nodeId}
                onAction={onServerAction}
                isLoading={isLoading}
                className={cn(
                  isDeveloperMode && "pl-10",
                  selectedServers.has(server.alias) && "ring-2 ring-blue-500"
                )}
              />
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ServerProxy className="w-5 h-5 text-muted-foreground" />
            <CardTitle>MCP Servers</CardTitle>
            <Badge variant="secondary">{servers.length} total</Badge>
          </div>

          {isDeveloperMode && selectedServers.size > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-body-small">
                {selectedServers.size} selected
              </span>
              <div className="flex gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleBulkAction("start")}
                  disabled={bulkActionLoading !== null}
                  className="text-green-600 hover:text-green-700"
                >
                  {bulkActionLoading === "start" ? (
                    <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  Start
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleBulkAction("stop")}
                  disabled={bulkActionLoading !== null}
                  className="text-red-600 hover:text-red-700"
                >
                  {bulkActionLoading === "stop" ? (
                    <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  ) : (
                    <Stop className="w-4 h-4" />
                  )}
                  Stop
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleBulkAction("restart")}
                  disabled={bulkActionLoading !== null}
                >
                  {bulkActionLoading === "restart" ? (
                    <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  ) : (
                    <Restart className="w-4 h-4" />
                  )}
                  Restart
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Search and Filters */}
        <div className="flex flex-col sm:flex-row gap-3 mt-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search servers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          <div className="flex gap-2 flex-wrap">
            <Button
              variant={statusFilter === "all" ? "default" : "outline"}
              size="sm"
              onClick={() => setStatusFilter("all")}
            >
              All ({servers.length})
            </Button>
            {Object.entries(statusCounts).map(([status, count]) => (
              <Button
                key={status}
                variant={statusFilter === status ? "default" : "outline"}
                size="sm"
                onClick={() => setStatusFilter(status as ServerStatusFilter)}
                className="capitalize"
              >
                {status} ({count})
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {filteredServers.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            {searchQuery || statusFilter !== "all"
              ? "No servers match your filters"
              : "No MCP servers found"}
          </div>
        ) : (
          <div className="space-y-6">
            {Object.entries(serversByStatus).map(([status, servers]) =>
              renderServerGroup(status, servers)
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
