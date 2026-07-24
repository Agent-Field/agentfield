import type { CSSProperties, ReactElement, ReactNode } from 'react'

export type DotFieldVariant = 'flow' | 'pulse' | 'orbit'

interface EmptyStateProps {
  title: string
  description: ReactNode
  variant?: DotFieldVariant
  size?: 'hero' | 'compact'
  supporting?: ReactNode
  action?: ReactNode
}

type DotStyle = CSSProperties & {
  '--dot-rest': number
  '--dot-peak': number
  '--dot-delay': string
  '--dot-duration': string
}

const DOT_COLUMNS = 13
const DOT_ROWS = 6
const DOT_COUNT = DOT_COLUMNS * DOT_ROWS

function motifStrength(variant: DotFieldVariant, column: number, row: number): number {
  const centerX = (DOT_COLUMNS - 1) / 2
  const centerY = (DOT_ROWS - 1) / 2

  if (variant === 'pulse') {
    const pulseLine = [3, 3, 3, 3, 2, 4, 1, 4, 2, 3, 3, 3, 3][column]
    return Math.max(0, 1 - Math.abs(row - pulseLine) * 0.48)
  }

  if (variant === 'orbit') {
    const distance = Math.hypot((column - centerX) / 1.75, row - centerY)
    return Math.max(0, 1 - Math.abs(distance - 1.75) * 0.55)
  }

  const flowLine = centerY + Math.sin(column * 0.68) * 1.25
  return Math.max(0, 1 - Math.abs(row - flowLine) * 0.42)
}

function dotStyle(index: number, variant: DotFieldVariant): DotStyle {
  const column = index % DOT_COLUMNS
  const row = Math.floor(index / DOT_COLUMNS)
  const noise = ((index * 47 + row * 17 + column * 11) % 29) / 29
  const strength = motifStrength(variant, column, row)
  const centerX = (DOT_COLUMNS - 1) / 2
  const centerY = (DOT_ROWS - 1) / 2

  const duration =
    variant === 'flow'
      ? 11 + noise * 3
      : variant === 'pulse'
        ? 13 + noise * 3
        : 15 + noise * 3

  const phase =
    variant === 'flow'
      ? column * 0.72 + row * 0.24
      : variant === 'pulse'
        ? Math.abs(column - centerX) * 0.78 + row * 0.12
        : ((Math.atan2(row - centerY, column - centerX) + Math.PI) / (Math.PI * 2)) * 11

  return {
    '--dot-rest': Number((0.11 + strength * 0.13).toFixed(2)),
    '--dot-peak': Number((0.2 + strength * 0.44 + noise * 0.04).toFixed(2)),
    '--dot-delay': `${(-phase - noise * 1.8).toFixed(2)}s`,
    '--dot-duration': `${duration.toFixed(2)}s`
  }
}

/**
 * AgentField's shared no-data texture. The same dot vocabulary changes shape
 * by context: flow on Home, signal on Activity, orbit for configuration data.
 */
export function DotField({
  variant = 'flow',
  compact = false
}: {
  variant?: DotFieldVariant
  compact?: boolean
}): ReactElement {
  return (
    <div className={`dot-field ${variant}${compact ? ' compact' : ''}`} aria-hidden="true">
      {Array.from({ length: DOT_COUNT }, (_, index) => (
        <span
          key={index}
          className="dot-field-point"
          style={dotStyle(index, variant)}
        />
      ))}
    </div>
  )
}

export function EmptyState({
  title,
  description,
  variant = 'flow',
  size = 'compact',
  supporting,
  action
}: EmptyStateProps): ReactElement {
  const compact = size === 'compact'
  return (
    <div className={`empty-state ${size}`}>
      <DotField variant={variant} compact={compact} />
      <div className="empty-state-copy">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
      {supporting ? <div className="empty-state-supporting">{supporting}</div> : null}
      {action ? <div className="empty-state-actions">{action}</div> : null}
    </div>
  )
}
