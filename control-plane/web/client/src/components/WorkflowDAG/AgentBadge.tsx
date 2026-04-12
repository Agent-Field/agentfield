import { cn } from "../../lib/utils";
import { agentColorManager } from "../../utils/agentColorManager";

interface AgentBadgeProps {
  agentName: string;
  agentId?: string;
  size?: "sm" | "md" | "lg";
  className?: string;
  showTooltip?: boolean;
}

export function AgentBadge({
  agentName,
  agentId,
  size = "md",
  className,
  showTooltip = true,
}: AgentBadgeProps) {
  const agentColor = agentColorManager.getAgentColor(agentName, agentId);
  const initials = agentColorManager.getAgentInitials(agentName);

  const sizeClasses = {
    sm: "w-5 h-5 text-nano font-bold",
    md: "w-6 h-6 text-micro font-bold",
    lg: "w-8 h-8 text-xs font-bold",
  };

  return (
    <div
      className={cn(
        // Base styling
        "relative inline-flex items-center justify-center rounded-full",
        "border shadow-sm",
        "transition-all duration-200 ease-out",
        // Size variants
        sizeClasses[size],
        // Hover effects
        "hover:scale-110 hover:shadow-md",
        className
      )}
      style={{
        backgroundColor: agentColor.primary,
        color: agentColor.text,
        boxShadow: `0 0 0 1px color-mix(in srgb, ${agentColor.primary} 35%, transparent), 0 2px 8px -2px color-mix(in srgb, var(--foreground) 14%, transparent)`,
        borderColor: `color-mix(in srgb, ${agentColor.primary} 25%, var(--border))`,
      }}
      title={showTooltip ? `Agent: ${agentName}` : undefined}
    >
      {/* Badge content */}
      <span className="select-none leading-none">{initials}</span>

      {/* Subtle glow effect */}
      <div
        className="absolute inset-0 -z-10 rounded-full opacity-30 blur-sm"
        style={{
          backgroundColor: `color-mix(in srgb, ${agentColor.primary} 60%, transparent)`,
        }}
      />
    </div>
  );
}

// Utility component for displaying agent color as a simple dot
export function AgentColorDot({
  agentName,
  agentId,
  size = 8,
  className,
}: {
  agentName: string;
  agentId?: string;
  size?: number;
  className?: string;
}) {
  const agentColor = agentColorManager.getAgentColor(agentName, agentId);

  return (
    <div
      className={cn("rounded-full border shadow-sm", className)}
      style={{
        width: size,
        height: size,
        backgroundColor: agentColor.primary,
        borderColor: `color-mix(in srgb, ${agentColor.primary} 25%, var(--border))`,
        boxShadow: `0 0 0 1px color-mix(in srgb, ${agentColor.primary} 35%, transparent)`,
      }}
      title={`Agent: ${agentName}`}
    />
  );
}
