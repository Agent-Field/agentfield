/** Compact run id for dashboard visuals (tooltip shows full id). */
export function shortRunIdForDashboard(runId: string, tail = 6): string {
  const t = Math.max(2, tail);
  if (runId.length <= t + 2) return runId;
  return `…${runId.slice(-t)}`;
}
