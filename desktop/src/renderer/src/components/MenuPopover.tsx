import { useLayoutEffect, useRef, useState } from 'react'
import type { ReactElement, ReactNode } from 'react'
import { createPortal } from 'react-dom'

/**
 * Overflow ("⋯") trigger + menu. The popover is portaled to <body> with
 * fixed coordinates because in-place absolute positioning gets clipped by
 * `.panel`'s rounded-corner `overflow: hidden` and re-anchored by
 * `.market-card`'s hover transform. Same escape hatch as `.af-tooltip`.
 */
export function MenuPopover({
  open,
  onToggle,
  disabled,
  ariaLabel,
  children
}: {
  open: boolean
  onToggle: () => void
  disabled?: boolean
  ariaLabel: string
  children: ReactNode
}): ReactElement {
  const buttonRef = useRef<HTMLButtonElement>(null)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)

  // Fixed positioning doesn't follow the trigger, so re-place on any
  // scroll (capture: the scrolling container isn't the window) or resize.
  useLayoutEffect(() => {
    if (!open) {
      setPos(null)
      return
    }
    const button = buttonRef.current
    if (!button) return
    const place = () => {
      const rect = button.getBoundingClientRect()
      setPos({ left: rect.right, top: rect.bottom + 4 })
    }
    place()
    window.addEventListener('scroll', place, true)
    window.addEventListener('resize', place)
    return () => {
      window.removeEventListener('scroll', place, true)
      window.removeEventListener('resize', place)
    }
  }, [open])

  return (
    <>
      <button
        ref={buttonRef}
        className="action-button icon"
        aria-label={ariaLabel}
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={disabled}
        onClick={(event) => {
          event.stopPropagation()
          onToggle()
        }}
      >
        ⋯
      </button>
      {open &&
        pos &&
        createPortal(
          <div className="menu-popover" role="menu" style={{ left: pos.left, top: pos.top }}>
            {children}
          </div>,
          document.body
        )}
    </>
  )
}
