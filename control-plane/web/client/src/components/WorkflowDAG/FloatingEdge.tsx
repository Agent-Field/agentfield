import { getBezierPath, useInternalNode, EdgeLabelRenderer, Position } from "@xyflow/react";
import { normalizeExecutionStatus } from "../../utils/status";
import { getEdgeParams } from "./EdgeUtils";

interface FloatingEdgeProps {
  id: string;
  source: string;
  target: string;
  markerEnd?: string;
  style?: React.CSSProperties;
  // React Flow always injects these default handle coords — use them as fallback
  // when the internal node isn't measured yet.
  sourceX?: number;
  sourceY?: number;
  targetX?: number;
  targetY?: number;
  data?: {
    status?: string;
    duration?: number;
    animated?: boolean;
    emphasis?: 'focus' | 'search' | 'muted' | 'default';
  };
}

function FloatingEdge({ id, source, target, style = {}, data, sourceX = 0, sourceY = 0, targetX = 0, targetY = 0 }: FloatingEdgeProps) {
  const sourceNode = useInternalNode(source);
  const targetNode = useInternalNode(target);

  // Derive path coordinates: prefer floating intersection math when both
  // internal nodes are available (gives cleaner edge routing), otherwise fall
  // back to the default handle positions React Flow injects so edges are
  // visible even before the first layout/measure cycle completes.
  let sx: number, sy: number, tx: number, ty: number;
  let sourcePos: Position, targetPos: Position;

  if (sourceNode && targetNode) {
    const params = getEdgeParams(sourceNode, targetNode);
    sx = params.sx; sy = params.sy; tx = params.tx; ty = params.ty;
    sourcePos = params.sourcePos; targetPos = params.targetPos;
  } else {
    sx = sourceX; sy = sourceY; tx = targetX; ty = targetY;
    sourcePos = Position.Bottom; targetPos = Position.Top;
  }

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX: sx,
    sourceY: sy,
    sourcePosition: sourcePos,
    targetPosition: targetPos,
    targetX: tx,
    targetY: ty,
  });

  const status = data?.status || "default";
  const canonicalStatus = normalizeExecutionStatus(status);
  const isAnimated = data?.animated || canonicalStatus === "running";
  const duration = data?.duration;

  // Status-based styling
  const getStatusStyle = () => {
    const baseStyle = {
      stroke: (() => {
        // CSS vars store raw HSL components (e.g. "142 76% 36%"), so we must
        // wrap them in hsl() for SVG stroke attributes. The status here is the
        // edge's OWN canonical status — never propagated from a parent run, so
        // a child can be running blue even when the root is cancelled.
        switch (canonicalStatus) {
          case "succeeded":
            return "hsl(var(--status-success))";
          case "failed":
            return "hsl(var(--status-error))";
          case "running":
            return "hsl(var(--status-info))";
          case "paused":
          case "waiting":
          case "pending":
          case "queued":
            return "hsl(var(--status-warning))";
          case "cancelled":
            return "hsl(var(--muted-foreground))";
          default:
            return "hsl(var(--muted-foreground))";
        }
      })(),
      strokeWidth: 2,
      strokeLinecap: "round" as const,
      strokeLinejoin: "round" as const,
      filter:
        "drop-shadow(0 1px 2px rgba(255,255,255,0.08))",
    };

    switch (canonicalStatus) {
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

  const emphasis = data?.emphasis;

  const edgeStyle = {
    ...getStatusStyle(),
    ...style,
  } as React.CSSProperties;

  if (emphasis === 'muted') {
    edgeStyle.opacity = Math.min(Number(edgeStyle.opacity ?? 1), 0.18);
    edgeStyle.filter = 'grayscale(80%)';
    edgeStyle.strokeDasharray = edgeStyle.strokeDasharray || '6,4';
  } else if (emphasis === 'focus') {
    edgeStyle.opacity = 1;
    edgeStyle.strokeWidth = Math.max(Number(edgeStyle.strokeWidth ?? 2.5), 3.6);
    edgeStyle.filter = `${edgeStyle.filter || ''} drop-shadow(0 0 6px rgba(34,197,94,0.4))`.trim();
  } else if (emphasis === 'search') {
    edgeStyle.opacity = Math.max(Number(edgeStyle.opacity ?? 0.85), 0.9);
    edgeStyle.strokeWidth = Math.max(Number(edgeStyle.strokeWidth ?? 2.4), 3);
    edgeStyle.filter = `${edgeStyle.filter || ''} drop-shadow(0 0 6px rgba(59,130,246,0.4))`.trim();
  }

  // Use edge id in marker id to avoid duplicate-id conflicts across edges
  const markerKey = `${id}-${canonicalStatus}`;
  const enhancedMarkerEnd = `url(#arrowclosed-${markerKey})`;
  const strokeColor = (edgeStyle.stroke as string) || "hsl(var(--muted-foreground))";

  return (
    <>
      <defs>
        <marker
          id={`arrowclosed-${markerKey}`}
          markerWidth={canonicalStatus === "running" ? 16 : 12}
          markerHeight={canonicalStatus === "running" ? 16 : 12}
          refX="9"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon
            points="0,0 0,6 9,3"
            fill={strokeColor}
            stroke={strokeColor}
          />
        </marker>
      </defs>

      <path
        id={id}
        className="react-flow__edge-path"
        d={edgePath}
        markerEnd={enhancedMarkerEnd}
        style={edgeStyle}
      />

      {/* Duration label for completed edges */}
      {duration && canonicalStatus === "succeeded" && (
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
            <div className="rounded border border-border bg-card px-1.5 py-0.5 font-mono text-sm text-muted-foreground shadow-sm backdrop-blur-sm">
              {duration < 1000
                ? `${duration}ms`
                : `${(duration / 1000).toFixed(1)}s`}
            </div>
          </div>
        </EdgeLabelRenderer>
      )}

      {/* Animated flow particles for running edges */}
      {isAnimated && canonicalStatus === "running" && (
        <g>
          <circle r="3" fill={strokeColor}>
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

export default FloatingEdge;
