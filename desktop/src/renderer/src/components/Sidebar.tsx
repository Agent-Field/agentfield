import type { ReactElement } from 'react'
import { m, useReducedMotion } from 'motion/react'
import type { View } from '../../../shared/deeplink'
import type { CpTone } from '../App'
import { AgentFieldDesktopBrand } from './AgentFieldDesktopBrand'

// Re-exported so view components keep one import site; the canonical list
// lives in shared/deeplink.ts, where agentfield:// URLs resolve to views.
export type { View }

interface SidebarProps {
  view: View
  onSelect: (view: View) => void
  cpTone: CpTone
  cpLabel: string
  /** Prefer starting the control plane when the status pill is actionable. */
  onStartControlPlane?: () => void
  startingControlPlane?: boolean
}

function Icon({ d }: { d: string }) {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={d} />
    </svg>
  )
}

// Simple line icons (24px grid, stroked), kept inline so the CSP stays strict.
const ICONS: Record<View, string> = {
  home: 'M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM9 22V12h6v10',
  install: 'M12 3v12M7 10l5 5l5-5M4 21h16',
  agents: 'M12 8V4H8M2 14h2M20 14h2M15 13v2M9 13v2M16 20H8a2 2 0 0 1-2-2v-6a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2z',
  activity: 'M3 12h4l3-8l4 16l3-8h4',
  settings: 'M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6'
}

// Nav v2 (DESIGN.md §2.1): 4 items. Installing is an action on the Agents
// library ("+ Add agent"), not a place — `install` stays a valid deep-link
// View that App maps to Agents with add-mode open.
const NAV: Array<{ id: View; label: string }> = [
  { id: 'home', label: 'Home' },
  { id: 'agents', label: 'Agents' },
  { id: 'activity', label: 'Activity' },
  { id: 'settings', label: 'Settings' }
]

// UI-furniture spring (DESIGN.md §5.1) — fast, settles with no felt overshoot.
const NAV_SPRING = { type: 'spring', stiffness: 500, damping: 40 } as const

export function Sidebar({
  view,
  onSelect,
  cpTone,
  cpLabel,
  onStartControlPlane,
  startingControlPlane
}: SidebarProps): ReactElement {
  const reducedMotion = useReducedMotion()
  const needsAttention = cpTone === 'red' || cpTone === 'yellow'
  const isMac = window.agentfield.platform === 'darwin'
  const mod = isMac ? '⌘' : 'Ctrl+'

  const handleStatusClick = (): void => {
    if (onStartControlPlane) onStartControlPlane()
    else onSelect('settings')
  }

  const statusPill = (
    <span className={`status-pill ${cpTone}`} title="Control plane">
      <span className="status-dot" aria-hidden="true" />
      {cpLabel}
    </span>
  )

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <AgentFieldDesktopBrand />
      </div>
      <nav className="sidebar-nav">
        {NAV.map((item, index) => {
          const active = view === item.id
          return (
            <button
              key={item.id}
              type="button"
              className={`nav-item ${active ? 'active' : ''}`}
              aria-current={active ? 'page' : undefined}
              title={`${item.label} ${mod}${index + 1}`}
              onClick={() => onSelect(item.id)}
            >
              {/* The signature micro-interaction (DESIGN.md §5.2): one shared
                  background element springs between nav items. Icon + label
                  color still crossfades via the CSS .nav-item transition.
                  Reduced motion → static background, no layout animation. */}
              {active &&
                (reducedMotion ? (
                  <span className="nav-active-bg" aria-hidden="true" />
                ) : (
                  <m.span
                    className="nav-active-bg"
                    layoutId="nav-active"
                    transition={NAV_SPRING}
                    aria-hidden="true"
                  />
                ))}
              <Icon d={ICONS[item.id]} />
              {item.label}
            </button>
          )
        })}
      </nav>
      <div className="sidebar-foot">
        {needsAttention ? (
          <button
            type="button"
            className="status-pill-button"
            onClick={handleStatusClick}
            disabled={startingControlPlane}
            title="Start AgentField server"
          >
            {statusPill}
          </button>
        ) : (
          statusPill
        )}
      </div>
    </aside>
  )
}
