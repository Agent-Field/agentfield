import { validateSessionTransport } from './sessionTransport.js';

export interface SessionDefinition {
  name: string;
  provider: string;
  transport: string;
  model?: string;
  modalities: string[];
  voice?: string;
  tools: string[];
  tags: string[];
  proposed_tags: string[];
  approved_tags: string[];
  metadata: Record<string, unknown>;
}

export interface SessionOptions {
  provider: string;
  transport: string;
  model?: string;
  modalities?: string[];
  voice?: string;
  tools?: string[];
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface SessionTurn {
  text?: string;
  transcript?: string;
  audio?: unknown;
  audioFormat?: string;
  channel?: string;
  metadata?: Record<string, unknown>;
}

export class RealtimeSession {
  readonly sessionId: string;
  readonly definition: SessionDefinition;

  constructor(sessionId: string, definition: SessionDefinition) {
    this.sessionId = sessionId;
    this.definition = definition;
  }

  async input(): Promise<SessionTurn> {
    throw new Error('session.input() is populated by the AgentField control plane transport adapter');
  }
}

export function buildSessionDefinition(name: string, options: SessionOptions): SessionDefinition {
  const capability = validateSessionTransport(options.provider, options.transport);
  return {
    name,
    provider: capability.provider,
    transport: capability.transport,
    model: options.model,
    modalities: options.modalities ?? ['audio', 'text'],
    voice: options.voice,
    tools: options.tools ?? [],
    tags: options.tags ?? [],
    proposed_tags: options.tags ?? [],
    approved_tags: [],
    metadata: options.metadata ?? {}
  };
}
