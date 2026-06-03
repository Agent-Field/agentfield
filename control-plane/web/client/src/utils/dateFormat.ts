/**
 * Format a date as relative time for recent dates, absolute for older ones
 * Examples:
 * - "Just now"
 * - "5 mins ago"
 * - "2 hours ago"
 * - "Today, 4:02 PM"
 * - "Yesterday, 3:15 PM"
 * - "Nov 10, 4:02 PM"
 * - "Oct 15, 2024"
 */
export function formatRelativeTime(date: Date | string): string {
  const now = new Date();
  const then = typeof date === 'string' ? new Date(date) : date;

  // Handle invalid or zero-value dates (e.g., Go's time.Time zero value "0001-01-01")
  if (isNaN(then.getTime()) || then.getFullYear() < 1970) {
    return '—';
  }

  const diffMs = now.getTime() - then.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  // Less than 1 minute
  if (diffMins < 1) {
    return '< 1 min ago';
  }

  // Less than 1 hour
  if (diffMins < 60) {
    return `${diffMins} min${diffMins === 1 ? '' : 's'} ago`;
  }

  // Less than 24 hours
  if (diffHours < 24) {
    return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
  }

  // Today
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  if (then >= todayStart) {
    return `Today, ${then.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })}`;
  }

  // Yesterday
  const yesterdayStart = new Date(todayStart);
  yesterdayStart.setDate(yesterdayStart.getDate() - 1);
  if (then >= yesterdayStart) {
    return `Yesterday, ${then.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })}`;
  }

  // Less than 7 days
  if (diffDays < 7) {
    return then.toLocaleDateString('en-US', {
      weekday: 'short',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  }

  // This year
  if (then.getFullYear() === now.getFullYear()) {
    return then.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  }

  // Older than this year
  return then.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
}

/**
 * Compact relative time for dense UI (tables, badges, status strips).
 * Examples: "now", "5s ago", "3m ago", "2h ago", "4d ago", ">1y ago"
 * Accepts Date, string (ISO), or undefined/null (returns "—").
 */
export function formatCompactRelativeTime(date: Date | string | undefined | null): string {
  if (!date) return '—';
  const then = typeof date === 'string' ? new Date(date) : date;
  if (isNaN(then.getTime())) return '—';
  const diffMs = Date.now() - then.getTime();
  if (diffMs < 0) return 'now';
  if (diffMs > 365 * 24 * 60 * 60 * 1000) return '>1y ago';
  const secs = Math.floor(diffMs / 1000);
  if (secs < 60) return secs < 5 ? 'now' : `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/**
 * Format a date as compact single-line format
 * Example: "Nov 10, 4:02 PM"
 */
export function formatCompactDate(date: Date | string): string {
  const then = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();

  // If same year, omit year
  if (then.getFullYear() === now.getFullYear()) {
    return then.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  }

  // Different year, include year
  return then.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
}
