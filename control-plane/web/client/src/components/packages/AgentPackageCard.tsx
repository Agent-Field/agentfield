import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { 
  Settings, 
  Play, 
  Stop, 
  Package, 
  Time, 
  User, 
  Tag,
  InProgress,
  CheckmarkFilled,
  WarningFilled,
  CircleFilled
} from '@/components/ui/icon-bridge';
import { getConfigurationStatusBadge, getAgentStatusBadge } from '../../services/configurationApi';
import type { AgentPackage, AgentLifecycleInfo } from '../../types/haxen';

interface AgentPackageCardProps {
  package: AgentPackage;
  agentStatus?: AgentLifecycleInfo;
  onConfigure: (pkg: AgentPackage) => void;
  onStart: (pkg: AgentPackage) => void;
  onStop: (pkg: AgentPackage) => void;
  isLoading?: boolean;
}

export const AgentPackageCard: React.FC<AgentPackageCardProps> = ({
  package: pkg,
  agentStatus,
  onConfigure,
  onStart,
  onStop,
  isLoading = false
}) => {
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  
  const configStatusBadge = getConfigurationStatusBadge(pkg.configuration_status);
  const agentStatusBadge = agentStatus ? getAgentStatusBadge(agentStatus.status) : null;

  const handleAction = async (action: string, callback: () => void) => {
    setActionLoading(action);
    try {
      await callback();
    } finally {
      setActionLoading(null);
    }
  };

  const getStatusIcon = () => {
    if (agentStatus) {
      switch (agentStatus.status) {
        case 'running':
          return <CheckmarkFilled className="h-4 w-4 text-green-500" />;
        case 'error':
          return <WarningFilled className="h-4 w-4 text-red-500" />;
        case 'starting':
        case 'stopping':
          return <InProgress className="h-4 w-4 text-blue-500 animate-spin" />;
        default:
          return <CircleFilled className="h-4 w-4 text-gray-400" />;
      }
    }
    return null;
  };

  return (
    <Card className="w-full hover:shadow-md transition-shadow duration-200">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-blue-50 rounded-lg">
              <Package className="h-5 w-5 text-blue-600" />
            </div>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-heading-3 truncate">{pkg.name}</CardTitle>
              <CardDescription className="text-sm text-gray-600 mt-1 line-clamp-2">
                {pkg.description}
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            {agentStatusBadge && (
              <Badge variant={agentStatusBadge.variant} className="text-xs">
                {agentStatusBadge.label}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Package Metadata */}
        <div className="flex items-center gap-4 text-sm text-gray-500">
          <div className="flex items-center gap-1">
            <User className="h-4 w-4" />
            <span>{pkg.author}</span>
          </div>
          <div className="flex items-center gap-1">
            <Tag className="h-4 w-4" />
            <span>v{pkg.version}</span>
          </div>
          <div className="flex items-center gap-1">
            <Time className="h-4 w-4" />
            <span>
              {pkg.installed_at
                ? (() => {
                    const date = new Date(pkg.installed_at);
                    return isNaN(date.getTime()) ? 'Invalid date' : date.toLocaleDateString();
                  })()
                : 'N/A'
              }
            </span>
          </div>
        </div>

        {/* Configuration Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Configuration:</span>
            <Badge variant={configStatusBadge.variant} className="text-xs">
              {configStatusBadge.label}
            </Badge>
          </div>
        </div>

        {/* Tags */}
        {pkg.tags && pkg.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {pkg.tags.slice(0, 3).map((tag) => (
              <Badge key={tag} variant="outline" className="text-xs">
                {tag}
              </Badge>
            ))}
            {pkg.tags.length > 3 && (
              <Badge variant="outline" className="text-xs">
                +{pkg.tags.length - 3} more
              </Badge>
            )}
          </div>
        )}

        {/* Error Message */}
        {agentStatus?.error_message && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start gap-2">
              <WarningFilled className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-red-800">Error</p>
                <p className="text-xs text-red-600 mt-1">{agentStatus.error_message}</p>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons - V0: Always show all buttons */}
        <div className="flex gap-2 pt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleAction('configure', () => onConfigure(pkg))}
            disabled={isLoading || actionLoading === 'configure'}
            className="flex-1 transition-all duration-200 hover:bg-bg-hover focus-ring"
          >
            {actionLoading === 'configure' ? (
              <InProgress className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <Settings className="h-4 w-4 mr-2" />
            )}
            Configure
          </Button>

          <Button
            variant="default"
            size="sm"
            onClick={() => handleAction('start', () => onStart(pkg))}
            disabled={isLoading || actionLoading === 'start'}
            className="flex-1 bg-status-success hover:bg-status-success/90 text-white border-status-success-border transition-all duration-200 focus-ring"
          >
            {actionLoading === 'start' ? (
              <InProgress className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Start
          </Button>

          <Button
            variant="destructive"
            size="sm"
            onClick={() => handleAction('stop', () => onStop(pkg))}
            disabled={isLoading || actionLoading === 'stop'}
            className="flex-1 bg-status-error hover:bg-status-error/90 text-white border-status-error-border transition-all duration-200 focus-ring"
          >
            {actionLoading === 'stop' ? (
              <InProgress className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <Stop className="h-4 w-4 mr-2" />
            )}
            Stop
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
