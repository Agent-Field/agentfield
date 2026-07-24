import type { RawResult } from '../types.js';

export interface HarnessProvider {
  execute(prompt: string, options: Record<string, unknown>): Promise<RawResult>;
}
