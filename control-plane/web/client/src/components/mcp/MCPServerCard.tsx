import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Play,
  Stop,
  Restart,
  Tools,
  Time,
  ServerProxy,
  ChevronDown,
  ChevronUp
} from '@/components/ui/icon-bridge';
import { cn } from '@/lib/utils';
import { useMode } from '@/contexts/ModeContext';
import { MCPHealthIndicator } from './MCPHealthIndicator';
import type { MCPServerHealthForUI, MCPServerAction } from '@/types/agentfield';

interface MCPServerCardProps {
  server: MCPServerHealthForUI;
  nodeId: string;
  onAction?: (action: MCPServerAction, serverAlias: string) => Promise<void>;
  isLoading?: boolean;
  className?: string;
}

/**
 * Display individual MCP server status and information
 * Shows health indicators, uptime, and basic metrics with action buttons
 */
export function MCPServerCard({
  server,
  nodeId: _nodeId,
  onAction,
  isLoading = false,
  className
}: MCPServerCardProps) {
  const { mode } = useMode();
  const [isExpanded, setIsExpanded] = useState(false);
  const [actionLoading, setActionLoading] = useState<MCPServerAction | null>(null);

  const isDeveloperMode = mode === 'developer';

  const handleAction = async (action: MCPServerAction) => {
    if (!onAction || actionLoading) return;
    
    try {
      setActionLoading(action);
      await onAction(action, server.alias);
    } catch (error) {
      console.error(`Failed to ${action} server ${server.alias}:`, error);
    } finally {
      setActionLoading(null);
    }
  };

  const formatUptime = (uptimeFormatted?: string) => {
    if (!uptimeFormatted || uptimeFormatted === 'N/A') return 'N/A';
    return uptimeFormatted;
  };

  const formatLastHealthCheck = (lastCheck?: string) => {
    if (!lastCheck) return 'Never';
    const date = new Date(lastCheck);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  const getActionButtons = () => {
    const buttons = [];
    
    if (server.status === 'running') {
      buttons.push(
        <Button
          key="stop"
          variant="outline"
          size="sm"
          onClick={() => handleAction('stop')}
          disabled={isLoading || !!actionLoading}
          className="text-red-600 hover:text-red-700"
        >
          {actionLoading === 'stop' ? (
            <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
          ) : (
            <Stop className="w-4 h-4" />
          )}
          Stop
        </Button>
      );
      
      buttons.push(
        <Button
          key="restart"
          variant="outline"
          size="sm"
          onClick={() => handleAction('restart')}
          disabled={isLoading || !!actionLoading}
        >
          {actionLoading === 'restart' ? (
            <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
          ) : (
            <Restart className="w-4 h-4" />
          )}
          Restart
        </Button>
      );
    } else {
      buttons.push(
        <Button
          key="start"
          variant="outline"
          size="sm"
          onClick={() => handleAction('start')}
          disabled={isLoading || !!actionLoading}
          className="text-green-600 hover:text-green-700"
        >
          {actionLoading === 'start' ? (
            <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          Start
        </Button>
      );
    }
    
    return buttons;
  };

  return (
    <Card className={cn('transition-all duration-200 hover:shadow-md', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <ServerProxy className="w-5 h-5 text-muted-foreground" />
            <div>
              <CardTitle className="text-heading-3">{server.alias}</CardTitle>
              <div className="flex items-center gap-2 mt-1">
                <MCPHealthIndicator
                  status={server.status}
                  size="sm"
                  uptime={server.uptime_formatted}
                />
                {server.tool_count > 0 && (
                  <Badge variant="secondary" className="text-xs">
                    <Tools className="w-3 h-3 mr-1" />
                    {server.tool_count} tools
                  </Badge>
                )}
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {isDeveloperMode && (
              <div className="flex gap-1">
                {getActionButtons()}
              </div>
            )}
            
            {isDeveloperMode && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsExpanded(!isExpanded)}
                className="p-1"
              >
                {isExpanded ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        {/* Basic Info - Always Visible */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <Time className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">Uptime:</span>
            <span className="font-medium">{formatUptime(server.uptime_formatted)}</span>
          </div>
          
          {server.success_rate !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Success Rate:</span>
              <span className={cn(
                'font-medium',
                server.success_rate >= 95 ? 'text-green-600' :
                server.success_rate >= 80 ? 'text-yellow-600' : 'text-red-600'
              )}>
                {server.success_rate.toFixed(1)}%
              </span>
            </div>
          )}
        </div>

        {/* Error Message - Show if present */}
        {server.error_message && (
          <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-800 font-medium">Error:</p>
            <p className="text-sm text-red-700 mt-1">{server.error_message}</p>
          </div>
        )}

        {/* Expanded Details - Developer Mode Only */}
        {isExpanded && isDeveloperMode && (
          <>
            <Separator className="my-4" />
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-4 text-sm">
                {server.port && (
                  <div>
                    <span className="text-muted-foreground">Port:</span>
                    <span className="ml-2 font-mono">{server.port}</span>
                  </div>
                )}
                
                {server.process_id && (
                  <div>
                    <span className="text-muted-foreground">PID:</span>
                    <span className="ml-2 font-mono">{server.process_id}</span>
                  </div>
                )}
                
                {server.avg_response_time_ms !== undefined && (
                  <div>
                    <span className="text-muted-foreground">Avg Response:</span>
                    <span className="ml-2 font-medium">{server.avg_response_time_ms}ms</span>
                  </div>
                )}
                
                <div>
                  <span className="text-muted-foreground">Last Check:</span>
                  <span className="ml-2 font-medium">
                    {formatLastHealthCheck(server.last_health_check ?? '')}
                  </span>
                </div>
              </div>
              
              {server.started_at && (
                <div className="text-sm">
                  <span className="text-muted-foreground">Started:</span>
                  <span className="ml-2 font-medium">
                    {new Date(server.started_at).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
