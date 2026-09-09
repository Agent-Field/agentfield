/**
 * Trailing-slash trimming without a regex.
 *
 * `value.replace(/\/+$/, '')` reads well but backtracks quadratically on
 * inputs shaped like '///…x' (CodeQL js/polynomial-redos). Only the SDK's
 * LocalVerifier was ever flagged — the desktop call sites take operator
 * config, not attacker input — but there is no reason to keep four copies of
 * a pattern the scanner objects to when one linear helper does the job.
 */
export function stripTrailingSlashes(value: string): string {
  let end = value.length
  while (end > 0 && value.charCodeAt(end - 1) === 47 /* '/' */) {
    end--
  }
  return value.slice(0, end)
}
