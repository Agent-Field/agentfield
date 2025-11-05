import * as React from "react";
import { Time, DataBase } from "@/components/ui/icon-bridge";
import { cn } from "../../lib/utils";

/**
 * Reusable data formatting components for consistent data display
 * across the Haxen application. These components ensure uniform
 * formatting of timestamps, durations, file sizes, and other common data types.
 */

// Duration formatter component
interface DurationDisplayProps {
  durationMs?: number;
  className?: string;
  showIcon?: boolean;
}

export function DurationDisplay({
  durationMs,
  className,
  showIcon = false
}: DurationDisplayProps) {
  const formatDuration = (ms?: number) => {
    if (!ms) return 'N/A';

    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`;
    return `${(ms / 3600000).toFixed(1)}h`;
  };

  return (
    <span className={cn("text-mono-foundation", className)}>
      {showIcon && <Time size={12} className="inline mr-1" />}
      {formatDuration(durationMs)}
    </span>
  );
}

// Timestamp formatter component
interface TimestampDisplayProps {
  timestamp: string | Date | null;
  format?: 'relative' | 'absolute' | 'smart';
  className?: string;
  showIcon?: boolean;
}

export function TimestampDisplay({
  timestamp,
  format = 'relative',
  className,
  showIcon = false
}: TimestampDisplayProps) {
  const formatTimestamp = (ts: string | Date | null, formatType: string) => {
    if (!ts) return 'Never';

    const date = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();

    if (formatType === 'absolute') {
      return date.toLocaleDateString();
    }

    if (formatType === 'smart') {
      // Use relative for recent, absolute for old
      if (diffMs < 604800000) { // 7 days
        return formatRelativeTime(diffMs);
      }
      return date.toLocaleDateString();
    }

    // Default relative formatting
    return formatRelativeTime(diffMs);
  };

  const formatRelativeTime = (diffMs: number) => {
    if (diffMs < 60000) return 'Just now';
    if (diffMs < 3600000) return `${Math.floor(diffMs / 60000)}m ago`;
    if (diffMs < 86400000) return `${Math.floor(diffMs / 3600000)}h ago`;
    if (diffMs < 604800000) return `${Math.floor(diffMs / 86400000)}d ago`;
    return `${Math.floor(diffMs / 604800000)}w ago`;
  };

  return (
    <span className={cn("timestamp-foundation", className)}>
      {showIcon && <Time size={12} className="inline mr-1" />}
      {formatTimestamp(timestamp, format)}
    </span>
  );
}

// File size formatter component
interface FileSizeDisplayProps {
  bytes?: number;
  className?: string;
  showIcon?: boolean;
}

export function FileSizeDisplay({
  bytes,
  className,
  showIcon = false
}: FileSizeDisplayProps) {
  const formatSize = (b?: number) => {
    if (!b) return 'N/A';

    if (b < 1024) return `${b}B`;
    if (b < 1048576) return `${(b / 1024).toFixed(1)}KB`;
    if (b < 1073741824) return `${(b / 1048576).toFixed(1)}MB`;
    return `${(b / 1073741824).toFixed(1)}GB`;
  };

  return (
    <span className={cn("text-mono-foundation", className)}>
      {showIcon && <DataBase size={12} className="inline mr-1" />}
      {formatSize(bytes)}
    </span>
  );
}

// Execution ID formatter component
interface ExecutionIdDisplayProps {
  executionId: string;
  showCopy?: boolean;
  truncate?: boolean;
  className?: string;
}

export function ExecutionIdDisplay({
  executionId,
  showCopy = false,
  truncate = true,
  className
}: ExecutionIdDisplayProps) {
  const displayId = truncate && executionId.length > 12
    ? `${executionId.slice(0, 8)}...${executionId.slice(-4)}`
    : executionId;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(executionId);
    } catch (err) {
      console.error('Failed to copy execution ID:', err);
    }
  };

  return (
    <span className={cn("execution-id-foundation", className)}>
      <span>{displayId}</span>
      {showCopy && (
        <button
          onClick={handleCopy}
          className="ml-1 opacity-60 hover:opacity-100 transition-opacity"
          title="Copy execution ID"
        >
          📋
        </button>
      )}
    </span>
  );
}

// Data size comparison component (input/output sizes)
interface DataSizeComparisonProps {
  inputSize?: number;
  outputSize?: number;
  className?: string;
  showIcons?: boolean;
}

export function DataSizeComparison({
  inputSize,
  outputSize,
  className,
  showIcons = false
}: DataSizeComparisonProps) {
  return (
    <span className={cn("text-mono-foundation", className)}>
      {showIcons && <DataBase size={12} className="inline mr-1" />}
      <FileSizeDisplay bytes={inputSize} /> / <FileSizeDisplay bytes={outputSize} />
    </span>
  );
}

// Agent capability summary component
interface AgentCapabilitySummaryProps {
  reasonerCount: number;
  skillCount: number;
  className?: string;
  format?: 'full' | 'compact' | 'minimal';
}

export function AgentCapabilitySummary({
  reasonerCount,
  skillCount,
  className,
  format = 'full'
}: AgentCapabilitySummaryProps) {
  const formatCapabilities = () => {
    switch (format) {
      case 'compact':
        return `${reasonerCount}R/${skillCount}S`;
      case 'minimal':
        return `${reasonerCount + skillCount}`;
      default:
        return `${reasonerCount} reasoners, ${skillCount} skills`;
    }
  };

  return (
    <span className={cn("text-secondary-foundation", className)}>
      {formatCapabilities()}
    </span>
  );
}

// Percentage formatter component
interface PercentageDisplayProps {
  value: number;
  decimals?: number;
  className?: string;
}

export function PercentageDisplay({
  value,
  decimals = 1,
  className
}: PercentageDisplayProps) {
  const percentage = (value * 100).toFixed(decimals);

  return (
    <span className={cn("text-mono-foundation", className)}>
      {percentage}%
    </span>
  );
}

// Counter/metrics display component
interface MetricDisplayProps {
  label: string;
  value: number | string;
  icon?: React.ReactNode;
  format?: 'number' | 'abbreviated';
  className?: string;
}

export function MetricDisplay({
  label,
  value,
  icon,
  format = 'number',
  className
}: MetricDisplayProps) {
  const formatValue = (val: number | string) => {
    if (typeof val === 'string') return val;
    if (format === 'abbreviated') {
      if (val >= 1000000) return `${(val / 1000000).toFixed(1)}M`;
      if (val >= 1000) return `${(val / 1000).toFixed(1)}K`;
    }
    return val.toString();
  };

  return (
    <div className={cn("flex items-center gap-1 text-sm", className)}>
      {icon}
      <span className="text-muted-foreground">{label}:</span>
      <span className="font-medium">{formatValue(value)}</span>
    </div>
  );
}

// Utility function for consistent text highlighting
export function highlightSearchText(text: string, searchQuery?: string) {
  if (!searchQuery) return text;

  const regex = new RegExp(`(${searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const parts = text.split(regex);

  return parts.map((part, index) =>
    regex.test(part) ? (
      <mark
        key={index}
        className="bg-yellow-200 dark:bg-yellow-800 px-1 rounded"
      >
        {part}
      </mark>
    ) : (
      part
    )
  );
}

// Status health indicator
interface StatusHealthDisplayProps {
  status: string;
  healthPercentage?: number;
  className?: string;
}

export function StatusHealthDisplay({
  status,
  healthPercentage,
  className
}: StatusHealthDisplayProps) {
  const getHealthColor = (health?: number) => {
    if (!health) return "text-muted-foreground";
    if (health >= 80) return "text-green-600";
    if (health >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <span className="text-sm capitalize">{status}</span>
      {healthPercentage !== undefined && (
        <span className={cn("text-xs font-mono", getHealthColor(healthPercentage))}>
          ({healthPercentage}%)
        </span>
      )}
    </div>
  );
}
