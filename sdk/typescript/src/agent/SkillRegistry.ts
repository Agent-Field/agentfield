import type { AgentRouter } from '../router/AgentRouter.js';
import type { SkillDefinition, SkillHandler, SkillOptions } from '../types/skill.js';

export class SkillRegistry {
  private readonly skills = new Map<string, SkillDefinition>();

  register<TInput = any, TOutput = any>(
    name: string,
    handler: SkillHandler<TInput, TOutput>,
    options?: SkillOptions
  ) {
    this.skills.set(name, { name, handler, options });
  }

  includeRouter(router: AgentRouter) {
    router.skills.forEach((skill) => {
      this.skills.set(skill.name, skill);
    });
  }

  get(name: string) {
    return this.skills.get(name);
  }

  all() {
    return Array.from(this.skills.values());
  }
}
