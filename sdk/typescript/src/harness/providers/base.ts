import type { RawResult } from '../types.js';

export interface HarnessProvider {
  execute(prompt: string, options: Record<string, unknown>): Promise<RawResult>;
}

// projectDir/project_dir is the canonical agent root; cwd is the Python-matching fallback.
export function resolveRoot(options: Record<string, unknown>): string | undefined {
  for (const value of [options.projectDir, options.project_dir, options.cwd]) {
    if (typeof value === 'string' && value.length > 0) {
      return value;
    }
  }
  return undefined;
}
