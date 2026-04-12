import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";
import { normalizeExecutionStatus } from "../../utils/status";

interface EnhancedEdgeProps extends EdgeProps {
  data?: {
    status?: string;
    duration?: number;
    animated?: boolean;
  };
}

export function EnhancedEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
}: EnhancedEdgeProps) {
  const status = normalizeExecutionStatus(data?.status);
  const isAnimated = data?.animated || status === "running";
  const duration = data?.duration;

  // Calculate path with enhanced curves
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    curvature: 0.3, // More pronounced curves
  });

  // Status-based styling
  const getStatusStyle = () => {
    const baseStyle = {
      stroke: (() => {
        // Edge color reflects this edge's OWN canonical status — never
        // propagated from a parent run. A running child under a cancelled
        // parent must still render blue.
        switch (status) {
          case "succeeded":
            return "var(--status-success)";
          case "failed":
            return "var(--status-error)";
          case "running":
            return "var(--status-info)";
          case "paused":
          case "waiting":
          case "pending":
          case "queued":
            return "var(--status-warning)";
          case "cancelled":
            return "color-mix(in srgb, var(--muted-foreground) 65%, transparent)";
          default:
            return "color-mix(in srgb, var(--muted-foreground) 65%, transparent)";
        }
      })(),
      strokeWidth: 2,
      strokeLinecap: "round" as const,
      strokeLinejoin: "round" as const,
      filter:
        "drop-shadow(0 1px 2px color-mix(in srgb, var(--foreground) 12%, transparent))",
    };

    switch (status) {
      case "succeeded":
        return {
          ...baseStyle,
          strokeWidth: 2.5,
        };
      case "failed":
        return {
          ...baseStyle,
          strokeWidth: 2.5,
          strokeDasharray: "8,4",
        };
      case "running":
        return {
          ...baseStyle,
          strokeWidth: 3,
          strokeDasharray: "12,8",
          animation: "dash 2s linear infinite",
        };
      case "paused":
      case "waiting":
      case "pending":
      case "queued":
        return {
          ...baseStyle,
          strokeWidth: 2,
          opacity: 0.7,
        };
      case "cancelled":
        return {
          ...baseStyle,
          strokeWidth: 2,
          strokeDasharray: "4,4",
          opacity: 0.55,
        };
      default:
        return {
          ...baseStyle,
          opacity: 0.6,
        };
    }
  };

  const edgeStyle = {
    ...style,
    ...getStatusStyle(),
  };

  // Enhanced marker end
  const enhancedMarkerEnd = `url(#arrowclosed-${status})`;

  return (
    <>
      <defs>
        <marker
          id={`arrowclosed-${status}`}
          markerWidth={status === "running" ? 16 : 12}
          markerHeight={status === "running" ? 16 : 12}
          refX="9"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon
            points="0,0 0,6 9,3"
            fill={edgeStyle.stroke}
            stroke={edgeStyle.stroke}
          />
        </marker>
      </defs>
      <BaseEdge
        path={edgePath}
        markerEnd={enhancedMarkerEnd}
        style={edgeStyle}
      />

      {/* Duration label for completed edges */}
      {duration && status === "succeeded" && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              fontSize: 10,
              pointerEvents: "all",
            }}
            className="nodrag nopan"
          >
            <div className="bg-card/90 backdrop-blur-sm border border-border rounded px-1.5 py-0.5 text-muted-foreground font-mono text-xs shadow-sm">
              {duration < 1000
                ? `${duration}ms`
                : `${(duration / 1000).toFixed(1)}s`}
            </div>
          </div>
        </EdgeLabelRenderer>
      )}

      {/* Animated flow particles for running edges */}
      {isAnimated && status === "running" && (
        <g>
          <circle r="3" fill="var(--status-running)">
            <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
            <animate
              attributeName="opacity"
              values="0;1;1;0"
              dur="2s"
              repeatCount="indefinite"
            />
          </circle>
        </g>
      )}

      <style>{`
        @keyframes dash {
          to {
            stroke-dashoffset: -20;
          }
        }
      `}</style>
    </>
  );
}

// Custom edge types
export const customEdgeTypes = {
  enhanced: EnhancedEdge,
};
