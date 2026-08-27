export const DEFAULT_SHUTDOWN_TIMEOUT_MS = 30_000;

export function parseShutdownTimeout(value: string | undefined, warn: (message: string) => void = console.warn): number {
  if (!value?.trim()) return DEFAULT_SHUTDOWN_TIMEOUT_MS;
  const text = value.trim();
  if (/^\d+$/.test(text)) return Number(text) * 1000;
  const match = text.match(/^(\d+(?:\.\d+)?)(ms|s|m|h)$/);
  if (match) {
    const factors = { ms: 1, s: 1000, m: 60_000, h: 3_600_000 } as const;
    return Number(match[1]) * factors[match[2] as keyof typeof factors];
  }
  warn(`invalid AGENTFIELD_SHUTDOWN_TIMEOUT ${JSON.stringify(value)}; using 30s`);
  return DEFAULT_SHUTDOWN_TIMEOUT_MS;
}

export interface ServeOptions { handleSignals?: boolean }
