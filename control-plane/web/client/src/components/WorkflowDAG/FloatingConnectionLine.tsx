import { getBezierPath } from '@xyflow/react';
import { getEdgeParams } from './EdgeUtils';

interface FloatingConnectionLineProps {
  toX: number;
  toY: number;
  fromPosition: any;
  toPosition: any;
  fromNode: any;
}

function FloatingConnectionLine({
  toX,
  toY,
  fromPosition,
  toPosition,
  fromNode,
}: FloatingConnectionLineProps) {
  if (!fromNode) {
    return null;
  }

  // Create a mock target node at the cursor position
  const targetNode = {
    id: 'connection-target',
    measured: {
      width: 1,
      height: 1,
    },
    internals: {
      positionAbsolute: { x: toX, y: toY },
    },
  };

  const { sx, sy, tx, ty, sourcePos, targetPos } = getEdgeParams(
    fromNode,
    targetNode,
  );

  const [edgePath] = getBezierPath({
    sourceX: sx,
    sourceY: sy,
    sourcePosition: sourcePos || fromPosition,
    targetPosition: targetPos || toPosition,
    targetX: tx || toX,
    targetY: ty || toY,
  });

  return (
    <g>
      <path
        fill="none"
        stroke="var(--status-info)"
        strokeWidth={2}
        className="animated"
        d={edgePath}
        style={{
          strokeDasharray: "5,5",
          animation: "dash 1s linear infinite",
        }}
      />
      <circle
        cx={tx || toX}
        cy={ty || toY}
        fill="var(--card)"
        r={4}
        stroke="var(--status-info)"
        strokeWidth={2}
      />
      <style>{`
        @keyframes dash {
          to {
            stroke-dashoffset: -10;
          }
        }
      `}</style>
    </g>
  );
}

export default FloatingConnectionLine;
