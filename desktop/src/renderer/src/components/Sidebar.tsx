import type { ReactElement } from 'react'
import type { View } from '../../../shared/deeplink'
import type { CpTone } from '../App'

// Re-exported so view components keep one import site; the canonical list
// lives in shared/deeplink.ts, where agentfield:// URLs resolve to views.
export type { View }

interface SidebarProps {
  view: View
  onSelect: (view: View) => void
  cpTone: CpTone
  cpLabel: string
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
  dashboard: 'M3 3h7v9H3zM14 3h7v5h-7zM14 12h7v9h-7zM3 16h7v5H3z',
  agents: 'M5 7h14v10H5zM9 21h6M12 17v4M9 3v4M15 3v4',
  activity: 'M3 12h4l3 -8l4 16l3 -8h4',
  install: 'M12 3v12M7 10l5 5l5 -5M4 21h16',
  settings: 'M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6'
}

const NAV: Array<{ id: View; label: string }> = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'agents', label: 'Agents' },
  { id: 'activity', label: 'Activity' },
  { id: 'install', label: 'Install' },
  { id: 'settings', label: 'Settings' }
]

export function Sidebar({ view, onSelect, cpTone, cpLabel }: SidebarProps): ReactElement {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">AgentField</div>
      <nav className="sidebar-nav">
        {NAV.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${view === item.id ? 'active' : ''}`}
            onClick={() => onSelect(item.id)}
          >
            <Icon d={ICONS[item.id]} />
            {item.label}
          </button>
        ))}
      </nav>
      <div className="sidebar-foot">
        <span className={`status-pill ${cpTone}`} title="Control plane">
          <span className="status-dot" aria-hidden="true" />
          {cpLabel}
        </span>
      </div>
    </aside>
  )
}
