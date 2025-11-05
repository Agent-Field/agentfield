import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { useMode } from "@/contexts/ModeContext";
import { cn } from "@/lib/utils";
import type { MCPServerAction, MCPServerHealthForUI } from "@/types/haxen";
import {
  CheckmarkFilled,
  ErrorFilled,
  Play,
  Restart,
  Settings,
  Stop,
  Warning,
} from "@/components/ui/icon-bridge";
import * as React from "react";
import { useState } from "react";

interface MCPServerControlsProps {
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

interface ActionConfirmation {
  action: MCPServerAction;
  servers: string[];
  isOpen: boolean;
}

/**
 * Server management control panel
 * Provides start/stop/restart buttons with confirmation and bulk operations
 */
export function MCPServerControls({
  servers,
  nodeId: _nodeId,
  onServerAction,
  onBulkAction,
  isLoading = false,
  className,
}: MCPServerControlsProps) {
  const { mode } = useMode();
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<ActionConfirmation>({
    action: "start",
    servers: [],
    isOpen: false,
  });

  const isDeveloperMode = mode === "developer";

  // Only show in developer mode
  if (!isDeveloperMode) {
    return null;
  }

  const runningServers = servers.filter((s) => s.status === "running");
  const stoppedServers = servers.filter((s) => s.status === "stopped");
  const errorServers = servers.filter((s) => s.status === "error");

  const handleSingleAction = async (
    action: MCPServerAction,
    serverAlias: string
  ) => {
    if (!onServerAction || actionLoading) return;

    // Show confirmation for destructive actions
    if (action === "stop" || action === "restart") {
      setConfirmation({
        action,
        servers: [serverAlias],
        isOpen: true,
      });
      return;
    }

    await executeAction(action, [serverAlias]);
  };

  const handleBulkAction = async (
    action: MCPServerAction,
    serverList: MCPServerHealthForUI[]
  ) => {
    if (!onBulkAction || actionLoading || serverList.length === 0) return;

    const serverAliases = serverList.map((s) => s.alias);

    // Show confirmation for bulk actions
    setConfirmation({
      action,
      servers: serverAliases,
      isOpen: true,
    });
  };

  const executeAction = async (
    action: MCPServerAction,
    serverAliases: string[]
  ) => {
    try {
      setActionLoading(`${action}-${serverAliases.join(",")}`);

      if (serverAliases.length === 1 && onServerAction) {
        await onServerAction(action, serverAliases[0]);
      } else if (serverAliases.length > 1 && onBulkAction) {
        await onBulkAction(action, serverAliases);
      }
    } catch (error) {
      console.error(`Failed to ${action} servers:`, error);
    } finally {
      setActionLoading(null);
      setConfirmation((prev) => ({ ...prev, isOpen: false }));
    }
  };

  const getActionText = (action: MCPServerAction) => {
    switch (action) {
      case "start":
        return "Start";
      case "stop":
        return "Stop";
      case "restart":
        return "Restart";
      default:
        return action;
    }
  };

  const getActionIcon = (action: MCPServerAction) => {
    switch (action) {
      case "start":
        return Play;
      case "stop":
        return Stop;
      case "restart":
        return Restart;
      default:
        return Settings;
    }
  };

  const getActionColor = (action: MCPServerAction) => {
    switch (action) {
      case "start":
        return "text-green-600 hover:text-green-700";
      case "stop":
        return "text-red-600 hover:text-red-700";
      case "restart":
        return "text-blue-600 hover:text-blue-700";
      default:
        return "";
    }
  };

  const isActionLoading = (action: string, servers: string[]) => {
    return actionLoading === `${action}-${servers.join(",")}`;
  };

  const renderActionButton = (
    action: MCPServerAction,
    serverList: MCPServerHealthForUI[],
    label: string
  ) => {
    if (serverList.length === 0) return null;

    const ActionIcon = getActionIcon(action);
    const loading = isActionLoading(
      action,
      serverList.map((s) => s.alias)
    );

    return (
      <Button
        variant="outline"
        size="sm"
        onClick={() => handleBulkAction(action, serverList)}
        disabled={isLoading || !!actionLoading}
        className={getActionColor(action)}
      >
        {loading ? (
          <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent mr-2" />
        ) : (
          <ActionIcon className="w-4 h-4 mr-2" />
        )}
        {label} ({serverList.length})
      </Button>
    );
  };

  return (
    <>
      <Card className={cn("w-full", className)}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Settings className="w-5 h-5 text-muted-foreground" />
              <CardTitle>Server Controls</CardTitle>
              <Badge variant="secondary">{servers.length} servers</Badge>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Status Summary */}
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <div className="flex items-center justify-center mb-1">
                <CheckmarkFilled className="w-5 h-5 text-green-600" />
              </div>
              <div className="text-heading-3 text-green-700">
                {runningServers.length}
              </div>
              <div className="text-body-small">Running</div>
            </div>

            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-center mb-1">
                <Stop className="w-5 h-5 text-gray-600" />
              </div>
              <div className="text-heading-3 text-gray-700">
                {stoppedServers.length}
              </div>
              <div className="text-body-small">Stopped</div>
            </div>

            <div className="text-center p-3 bg-red-50 rounded-lg">
              <div className="flex items-center justify-center mb-1">
                <ErrorFilled className="w-5 h-5 text-red-600" />
              </div>
              <div className="text-heading-3 text-red-700">
                {errorServers.length}
              </div>
              <div className="text-body-small">Error</div>
            </div>
          </div>

          {/* Bulk Actions */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Bulk Actions</h3>

            <div className="flex flex-wrap gap-2">
              {renderActionButton(
                "start",
                stoppedServers.concat(errorServers),
                "Start All Stopped"
              )}
              {renderActionButton("stop", runningServers, "Stop All Running")}
              {renderActionButton(
                "restart",
                runningServers,
                "Restart All Running"
              )}
            </div>

            {servers.length > 0 && (
              <>
                <Separator />
                <div className="flex flex-wrap gap-2">
                  {renderActionButton(
                    "restart",
                    servers,
                    "Restart All Servers"
                  )}
                  {renderActionButton("stop", servers, "Stop All Servers")}
                </div>
              </>
            )}
          </div>

          {/* Individual Server Quick Actions */}
          {servers.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium">Individual Controls</h3>

              <div className="space-y-2 max-h-64 overflow-y-auto">
                {servers.map((server) => (
                  <div
                    key={server.alias}
                    className="flex items-center justify-between p-3 border rounded-md bg-gray-50"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          "w-2 h-2 rounded-full",
                          server.status === "running"
                            ? "bg-green-500"
                            : server.status === "error"
                            ? "bg-red-500"
                            : "bg-gray-400"
                        )}
                      />
                      <span className="font-medium">{server.alias}</span>
                      <Badge variant="outline" className="text-xs capitalize">
                        {server.status}
                      </Badge>
                    </div>

                    <div className="flex gap-1">
                      {server.status !== "running" && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() =>
                            handleSingleAction("start", server.alias)
                          }
                          disabled={isLoading || !!actionLoading}
                          className="text-green-600 hover:text-green-700 p-1"
                        >
                          {isActionLoading("start", [server.alias]) ? (
                            <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          ) : (
                            <Play className="w-4 h-4" />
                          )}
                        </Button>
                      )}

                      {server.status === "running" && (
                        <>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() =>
                              handleSingleAction("restart", server.alias)
                            }
                            disabled={isLoading || !!actionLoading}
                            className="text-blue-600 hover:text-blue-700 p-1"
                          >
                            {isActionLoading("restart", [server.alias]) ? (
                              <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                            ) : (
                              <Restart className="w-4 h-4" />
                            )}
                          </Button>

                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() =>
                              handleSingleAction("stop", server.alias)
                            }
                            disabled={isLoading || !!actionLoading}
                            className="text-red-600 hover:text-red-700 p-1"
                          >
                            {isActionLoading("stop", [server.alias]) ? (
                              <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                            ) : (
                              <Stop className="w-4 h-4" />
                            )}
                          </Button>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No Servers Message */}
          {servers.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <Settings className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">No MCP servers to control</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Confirmation Dialog */}
      <Dialog
        open={confirmation.isOpen}
        onOpenChange={(open) =>
          setConfirmation((prev) => ({ ...prev, isOpen: open }))
        }
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Warning className="w-5 h-5 text-yellow-600" />
              Confirm Action
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to{" "}
              {getActionText(confirmation.action).toLowerCase()} the following
              server{confirmation.servers.length > 1 ? "s" : ""}?
            </DialogDescription>
          </DialogHeader>

          <div className="py-4">
            <div className="space-y-2">
              {confirmation.servers.map((serverAlias) => (
                <div
                  key={serverAlias}
                  className="flex items-center gap-2 p-2 bg-gray-50 rounded"
                >
                  <Settings className="w-4 h-4 text-muted-foreground" />
                  <span className="font-medium">{serverAlias}</span>
                </div>
              ))}
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() =>
                setConfirmation((prev) => ({ ...prev, isOpen: false }))
              }
              disabled={!!actionLoading}
            >
              Cancel
            </Button>
            <Button
              onClick={() =>
                executeAction(confirmation.action, confirmation.servers)
              }
              disabled={!!actionLoading}
              className={cn(
                confirmation.action === "stop" && "bg-red-600 hover:bg-red-700",
                confirmation.action === "restart" &&
                  "bg-blue-600 hover:bg-blue-700"
              )}
            >
              {actionLoading ? (
                <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent mr-2" />
              ) : (
                React.createElement(getActionIcon(confirmation.action), {
                  className: "w-4 h-4 mr-2",
                })
              )}
              {getActionText(confirmation.action)}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
