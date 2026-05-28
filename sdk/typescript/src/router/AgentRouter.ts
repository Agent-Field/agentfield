import type { ReasonerDefinition, ReasonerHandler, ReasonerOptions } from '../types/reasoner.js';
import type { SkillDefinition, SkillHandler, SkillOptions } from '../types/skill.js';

export interface AgentRouterOptions {
  prefix?: string;
  tags?: string[];
}

export class AgentRouter {
  readonly prefix?: string;
  readonly tags?: string[];
  readonly reasoners: ReasonerDefinition[] = [];
  readonly skills: SkillDefinition[] = [];

  constructor(options: AgentRouterOptions = {}) {
    this.prefix = options.prefix;
    this.tags = options.tags;
  }

  reasoner<TInput = any, TOutput = any>(
    name: string,
    handler: ReasonerHandler<TInput, TOutput>,
    options?: ReasonerOptions
  ) {
    const fullName = this.prefix ? `${sanitize(this.prefix)}_${name}` : name;
    this.reasoners.push({ name: fullName, handler, options });
    return this;
  }

  skill<TInput = any, TOutput = any>(
    name: string,
    handler: SkillHandler<TInput, TOutput>,
    options?: SkillOptions
  ) {
    const fullName = this.prefix ? `${sanitize(this.prefix)}_${name}` : name;
    this.skills.push({ name: fullName, handler, options });
    return this;
  }
}

function sanitize(value: string) {
  return value.replace(/[^0-9a-zA-Z]+/g, '_').replace(/_+/g, '_').replace(/^_+|_+$/g, '');
}
