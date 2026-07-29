import type { ReactElement } from 'react'

/** AgentField's field dot, framed for the desktop product. */
export function AgentFieldDesktopBrand(): ReactElement {
  return (
    <div className="desktop-brand-lockup" aria-label="AgentField Desktop">
      <svg
        className="desktop-brand-mark"
        width="28"
        height="28"
        viewBox="0 0 28 28"
        fill="none"
        aria-hidden="true"
      >
        <rect className="desktop-brand-screen" x="2.5" y="3" width="23" height="17.5" rx="4" />
        <circle className="desktop-brand-dot" cx="9.25" cy="11.75" r="3.75" />
        <path className="desktop-brand-detail" d="M15.75 9.75h5M15.75 13.75h3.25" />
        <path className="desktop-brand-stand" d="M10.25 25h7.5M14 20.75V25" />
      </svg>

      <span className="desktop-brand-copy">
        <span className="desktop-brand-name">AgentField</span>
        <span className="desktop-brand-edition">Desktop</span>
      </span>
    </div>
  )
}
