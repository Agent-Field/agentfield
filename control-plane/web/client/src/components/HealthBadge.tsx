import React from 'react';
import type { HealthStatus } from '../types/agentfield';
import { cn } from '@/lib/utils';

interface HealthBadgeProps {
  status: HealthStatus;
}

const HealthBadge: React.FC<HealthBadgeProps> = ({ status }) => {
  const getStatusConfig = (status: HealthStatus) => {
    switch (status) {
      case 'active':
        return {
          dotColor: 'bg-green-500',
          textColor: 'text-green-700 dark:text-green-400',
          label: 'Active'
        };
      case 'inactive':
        return {
          dotColor: 'bg-red-500',
          textColor: 'text-red-700 dark:text-red-400',
          label: 'Inactive'
        };
      default:
        return {
          dotColor: 'bg-yellow-500',
          textColor: 'text-yellow-700 dark:text-yellow-400',
          label: 'Unknown'
        };
    }
  };

  const config = getStatusConfig(status);

  return (
    <div className="flex items-center gap-2">
      <div className={cn("h-2 w-2 rounded-full", config.dotColor)} />
      <span className={cn("text-xs font-medium", config.textColor)}>
        {config.label}
      </span>
    </div>
  );
};

export default HealthBadge;
