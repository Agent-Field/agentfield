import type { ReactElement } from 'react'

/**
 * Schematic empty-state identity marks (DESIGN.md §4.16) — same visual
 * family as the sidebar `•af` orbit mark: hairline circles in
 * `--text-tertiary` with one `--accent` dot. Monochrome line-art, never
 * illustration-y. The gentle 6s ambient pulse lives in styles.css
 * (`empty-mark-ambient`) and is gated by prefers-reduced-motion.
 */

/** Agents / marketplace empty: orbit with a faint second, empty orbit ring. */
export function OrbitMark(): ReactElement {
  return (
    <svg
      className="empty-mark empty-mark-ambient"
      viewBox="0 0 44 44"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="22" cy="22" r="10" stroke="currentColor" strokeWidth="1.5" />
      <circle
        cx="22"
        cy="22"
        r="17"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeDasharray="2 5"
        opacity="0.4"
      />
      <circle cx="22" cy="22" r="3" fill="var(--accent)" />
    </svg>
  )
}

/** Activity empty: flat pulse line rising once into a heartbeat blip. */
export function PulseMark(): ReactElement {
  return (
    <svg
      className="empty-mark empty-mark-ambient"
      viewBox="0 0 44 44"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M4 22H14L17 13L21 29L24 22H36"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="39" cy="22" r="2.5" fill="var(--accent)" />
    </svg>
  )
}
