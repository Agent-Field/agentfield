import type { SkillContext } from '../context/SkillContext.js';
import type { Awaitable } from './agent.js';

export interface SkillDefinition<TInput = any, TOutput = any> {
  name: string;
  handler: SkillHandler<TInput, TOutput>;
  options?: SkillOptions;
}

export type SkillHandler<TInput = any, TOutput = any> = (
  ctx: SkillContext<TInput>
) => Awaitable<TOutput>;

export interface SkillOptions {
  tags?: string[];
  description?: string;
  inputSchema?: any;
  outputSchema?: any;
  /** Force control-plane verification instead of local verification for this skill. */
  requireRealtimeValidation?: boolean;
}
