/** Values that switch an on-by-default AgentField flag off. */
export const DISABLED_FLAG_VALUES: readonly string[] = ['0', 'false', 'no', 'off'];

/** Reads an environment variable without assuming a Node runtime. */
export function readEnvValue(name: string): string | undefined {
  if (typeof process === 'undefined' || !process.env) return undefined;
  return process.env[name];
}

/** Returns false only for an explicitly disabled on-by-default flag. */
export function envFlagEnabled(name: string): boolean {
  return !DISABLED_FLAG_VALUES.includes((readEnvValue(name) ?? 'true').trim().toLowerCase());
}
