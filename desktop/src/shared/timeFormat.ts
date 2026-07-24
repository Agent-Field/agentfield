/**
 * Human-readable time helpers for Activity / Home run rows.
 * Relative by default ("3 minutes ago"); hover tooltip shows a short absolute.
 */

/** Elapsed / wall duration for a finished or live run. */
export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return ''
  const totalSec = Math.floor(ms / 1000)
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (totalSec < 10) return `${(ms / 1000).toFixed(1)}s`
  if (totalSec < 60) return `${totalSec}s`
  const mins = Math.floor(totalSec / 60)
  const secs = totalSec % 60
  if (mins < 60) return secs === 0 ? `${mins}m` : `${mins}m ${secs}s`
  const hours = Math.floor(mins / 60)
  const remMins = mins % 60
  if (hours < 48) return remMins === 0 ? `${hours}h` : `${hours}h ${remMins}m`
  const days = Math.floor(hours / 24)
  const remHours = hours % 24
  return remHours === 0 ? `${days}d` : `${days}d ${remHours}h`
}

/** Live elapsed since an ISO start — empty when the stamp is invalid. */
export function formatElapsed(iso: string, now = Date.now()): string {
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return ''
  return formatDuration(Math.max(0, now - t))
}

/** Relative start: "just now", "2 mins ago", "3 days ago". Never clock time. */
export function formatRelativeStarted(iso: string, now = Date.now()): string {
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return ''
  const delta = Math.max(0, now - t)
  const sec = Math.floor(delta / 1000)
  if (sec < 5) return 'just now'
  if (sec < 60) return `${sec}s ago`
  const mins = Math.floor(sec / 60)
  if (mins < 60) return mins === 1 ? '1 min ago' : `${mins} mins ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return hours === 1 ? '1 hr ago' : `${hours} hrs ago`
  const days = Math.floor(hours / 24)
  if (days < 30) return days === 1 ? '1 day ago' : `${days} days ago`
  const months = Math.floor(days / 30)
  if (months < 12) return months === 1 ? '1 month ago' : `${months} months ago`
  const years = Math.floor(days / 365)
  return years === 1 ? '1 year ago' : `${years} years ago`
}

/** Clock time only (e.g. "2:41 PM"). Date is owned by the feed group label. */
export function formatAbsoluteStarted(iso: string): string {
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return ''
  return new Date(t).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

/** Compact absolute for hover tooltips — short, no weekday/seconds noise. */
export function formatFullStarted(iso: string, now = Date.now()): string {
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return ''
  const d = new Date(t)
  const n = new Date(now)
  const sameDay =
    d.getFullYear() === n.getFullYear() &&
    d.getMonth() === n.getMonth() &&
    d.getDate() === n.getDate()
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  if (sameDay) return time
  const sameYear = d.getFullYear() === n.getFullYear()
  const date = d.toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    ...(sameYear ? {} : { year: 'numeric' })
  })
  return `${date}, ${time}`
}
