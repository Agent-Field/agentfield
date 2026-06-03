import React from 'react';
import { Alert, AlertDescription } from './alert';
import { Button } from './button';
import { RefreshCw, AlertTriangle } from '@/components/ui/icon-bridge';

interface RestartRequiredBannerProps {
  agentId: string;
  onRestart?: () => void;
  onDismiss?: () => void;
  className?: string;
}

export const RestartRequiredBanner: React.FC<RestartRequiredBannerProps> = ({
  onRestart,
  onDismiss,
  className
}) => {
  return (
    <Alert variant="default" className={`border-orange-200 bg-orange-50 ${className}`}>
      <AlertTriangle className="h-4 w-4 text-orange-600" />
      <AlertDescription className="flex items-center justify-between w-full">
        <div className="flex-1">
          <span className="font-medium text-orange-800">Restart Required</span>
          <p className="text-sm text-orange-700 mt-1">
            Configuration changes have been saved. Restart the agent to apply the new environment variables.
          </p>
        </div>
        <div className="flex items-center space-x-2 ml-4">
          {onRestart && (
            <Button
              variant="outline"
              size="sm"
              onClick={onRestart}
              className="border-orange-300 text-orange-700 hover:bg-orange-100"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Restart Agent
            </Button>
          )}
          {onDismiss && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onDismiss}
              className="text-orange-600 hover:bg-orange-100"
            >
              Dismiss
            </Button>
          )}
        </div>
      </AlertDescription>
    </Alert>
  );
};
