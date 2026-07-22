import type { ReactElement } from 'react'

/**
 * Layout-matched loading skeletons (DESIGN.md §4.15) — replace "Loading…"
 * text placeholders. Shimmer lives in styles.css (`skeleton-*` rules) and
 * goes static under prefers-reduced-motion. Render these only when data is
 * genuinely absent (first load), never over real content on poll refreshes.
 */
export function SkeletonRow(): ReactElement {
  return (
    <div className="skeleton-row" aria-hidden="true">
      <span className="skeleton-dot" />
      <span className="skeleton-bars">
        <span className="skeleton-bar" style={{ width: '40%' }} />
        <span className="skeleton-bar" style={{ width: '65%' }} />
      </span>
    </div>
  )
}

export function SkeletonRows({ count = 3 }: { count?: number }): ReactElement {
  return (
    <div role="status" aria-label="Loading">
      {Array.from({ length: count }, (_, i) => (
        <SkeletonRow key={i} />
      ))}
    </div>
  )
}
