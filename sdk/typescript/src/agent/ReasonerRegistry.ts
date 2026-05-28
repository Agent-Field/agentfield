import type { AgentRouter } from '../router/AgentRouter.js';
import type { ReasonerDefinition, ReasonerHandler, ReasonerOptions } from '../types/reasoner.js';

export class ReasonerRegistry {
  private readonly reasoners = new Map<string, ReasonerDefinition>();

  register<TInput = any, TOutput = any>(
    name: string,
    handler: ReasonerHandler<TInput, TOutput>,
    options?: ReasonerOptions
  ) {
    this.reasoners.set(name, { name, handler, options });
  }

  includeRouter(router: AgentRouter) {
    router.reasoners.forEach((reasoner) => {
      this.reasoners.set(reasoner.name, reasoner);
    });
  }

  get(name: string) {
    return this.reasoners.get(name);
  }

  all() {
    return Array.from(this.reasoners.values());
  }
}
