import { useState, useEffect } from 'react';
import {
  Play,
  Stop,
  InProgress,
  CheckmarkFilled,
  WarningFilled,
  Restart
} from '@/components/ui/icon-bridge';
import { Button } from './button';
import { cn } from '@/lib/utils';

export type AgentState = 'stopped' | 'starting' | 'running' | 'stopping' | 'error' | 'reconciling';

interface AgentControlButtonProps {
  agentId: string;
  currentState: AgentState;
  onToggle: (action: 'start' | 'stop' | 'reconcile') => Promise<void>;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'minimal' | 'ghost';
  showLabel?: boolean;
  disabled?: boolean;
  className?: string;
}

const stateConfig = {
  stopped: {
    icon: Play,
    action: 'start' as const,
    label: 'Start',
    description: 'Start agent',
    variant: 'ghost' as const,
    iconColor: 'text-muted-foreground hover:text-status-success'
  },
  starting: {
    icon: InProgress,
    action: null,
    label: 'Starting...',
    description: 'Agent is starting',
    variant: 'ghost' as const,
    iconColor: 'text-status-info'
  },
  running: {
    icon: Stop,
    action: 'stop' as const,
    label: 'Stop',
    description: 'Stop agent',
    variant: 'ghost' as const,
    iconColor: 'text-muted-foreground hover:text-status-error'
  },
  stopping: {
    icon: InProgress,
    action: null,
    label: 'Stopping...',
    description: 'Agent is stopping',
    variant: 'ghost' as const,
    iconColor: 'text-status-warning'
  },
  error: {
    icon: WarningFilled,
    action: 'reconcile' as const,
    label: 'Reconcile',
    description: 'Reconcile agent state',
    variant: 'ghost' as const,
    iconColor: 'text-status-warning hover:text-foreground'
  },
  reconciling: {
    icon: Restart,
    action: null,
    label: 'Reconciling...',
    description: 'Reconciling agent state',
    variant: 'ghost' as const,
    iconColor: 'text-muted-foreground'
  }
};

const sizeConfig = {
  sm: {
    button: 'h-8 w-8',
    icon: 'h-4 w-4',
    text: 'text-xs'
  },
  md: {
    button: 'h-9 w-9',
    icon: 'h-5 w-5',
    text: 'text-sm'
  },
  lg: {
    button: 'h-10 w-10',
    icon: 'h-6 w-6',
    text: 'text-base'
  }
};

export function AgentControlButton({
  agentId,
  currentState,
  onToggle,
  size = 'md',
  variant = 'default',
  showLabel = false,
  disabled = false,
  className
}: AgentControlButtonProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);

  const config = stateConfig[currentState];
  const sizeStyles = sizeConfig[size];
  const IconComponent = config.icon;
  const isAnimated = currentState === 'starting' || currentState === 'stopping' || currentState === 'reconciling';
  const canInteract = config.action && !disabled && !isProcessing;

  // Success feedback animation
  useEffect(() => {
    if (showSuccess) {
      const timer = setTimeout(() => setShowSuccess(false), 1200);
      return () => clearTimeout(timer);
    }
  }, [showSuccess]);

  const handleClick = async () => {
    if (!canInteract || !config.action) return;

    setIsProcessing(true);

    try {
      await onToggle(config.action);
      setShowSuccess(true);
    } catch (error) {
      console.error("Failed to toggle agent action", {
        action: config.action,
        agentId,
        error,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const getButtonVariant = () => {
    if (variant === 'minimal') return 'ghost';
    return config.variant;
  };

  const buttonClasses = cn(
    // Base styles for enterprise look
    'relative transition-all duration-200 ease-out',
    'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background',
    'group overflow-hidden',

    // Size
    sizeStyles.button,

    // Interactive states
    canInteract && 'cursor-pointer hover:scale-105 active:scale-95',
    !canInteract && 'cursor-default',

    // Disabled state
    disabled && 'opacity-50 cursor-not-allowed',

    className
  );

  const iconClasses = cn(
    sizeStyles.icon,
    'transition-all duration-200',
    config.iconColor,
    isAnimated && 'animate-spin',
    showSuccess && 'scale-110'
  );

  if (showLabel) {
    return (
      <div className="flex items-center space-x-2">
        <Button
          variant={getButtonVariant()}
          size="icon"
          className={buttonClasses}
          onClick={handleClick}
          disabled={!canInteract}
          title={config.description}
          aria-label={`${config.label} ${agentId}`}
        >
          {/* Success checkmark overlay */}
          {showSuccess && (
            <div className="absolute inset-0 flex items-center justify-center">
              <CheckmarkFilled className={cn(sizeStyles.icon, 'text-status-success animate-pulse')} />
            </div>
          )}

          {/* Main icon */}
          <IconComponent
            className={cn(
              iconClasses,
              showSuccess && 'opacity-0'
            )}
          />
        </Button>

        {/* Label text */}
        <span className={cn(
          sizeStyles.text,
          'text-muted-foreground font-medium',
          isProcessing && 'animate-pulse'
        )}>
          {config.label}
        </span>
      </div>
    );
  }

  return (
    <Button
      variant={getButtonVariant()}
      size="icon"
      className={buttonClasses}
      onClick={handleClick}
      disabled={!canInteract}
      title={config.description}
      aria-label={`${config.label} ${agentId}`}
    >
      {/* Success checkmark overlay */}
      {showSuccess && (
        <div className="absolute inset-0 flex items-center justify-center">
          <CheckmarkFilled className={cn(sizeStyles.icon, 'text-status-success animate-pulse')} />
        </div>
      )}

      {/* Main icon */}
      <IconComponent
        className={cn(
          iconClasses,
          showSuccess && 'opacity-0'
        )}
      />

      {/* Processing indicator */}
      {isProcessing && (
        <div className="absolute inset-0 rounded-lg animate-pulse bg-accent/20" />
      )}
    </Button>
  );
}
