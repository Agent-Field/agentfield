export const SUPPORTED_SESSION_TRANSPORTS = {
  openai: ['webrtc', 'websocket'],
  openrouter: ['audio_turns']
} as const;

export type SessionProvider = keyof typeof SUPPORTED_SESSION_TRANSPORTS;
export type SessionTransport = (typeof SUPPORTED_SESSION_TRANSPORTS)[SessionProvider][number];

export interface SessionTransportCapability {
  provider: SessionProvider;
  transport: SessionTransport;
}

export class SessionTransportError extends Error {
  readonly provider: string;
  readonly transport: string;
  readonly supported: readonly string[];

  constructor(provider: string, transport: string, supported: readonly string[]) {
    const supportedDisplay = supported.length > 0 ? supported.join(', ') : 'none';
    super(
      `Unsupported session transport '${transport}' for provider '${provider}'. ` +
        `Supported transports: ${supportedDisplay}. AgentField does not infer ` +
        'or switch providers; set provider and transport explicitly.'
    );
    this.name = 'SessionTransportError';
    this.provider = provider;
    this.transport = transport;
    this.supported = supported;
  }
}

export function normalizeSessionTransportValue(value: string): string {
  return value.trim().toLowerCase().replace(/-/g, '_');
}

export function validateSessionTransport(provider: string, transport: string): SessionTransportCapability {
  const normalizedProvider = normalizeSessionTransportValue(provider);
  const normalizedTransport = normalizeSessionTransportValue(transport);

  if (!normalizedProvider) {
    throw new Error('Session provider is required; AgentField does not infer providers.');
  }
  if (!normalizedTransport) {
    throw new Error('Session transport is required; AgentField does not infer transports.');
  }

  const supported = SUPPORTED_SESSION_TRANSPORTS[normalizedProvider as SessionProvider];
  if (!supported) {
    const known = Object.keys(SUPPORTED_SESSION_TRANSPORTS).sort().join(', ');
    throw new Error(
      `Unknown session provider '${provider}'. Known providers: ${known}. ` +
        'Register provider capabilities before using a custom session provider.'
    );
  }

  if (!supported.includes(normalizedTransport as never)) {
    throw new SessionTransportError(normalizedProvider, normalizedTransport, supported);
  }

  return {
    provider: normalizedProvider as SessionProvider,
    transport: normalizedTransport as SessionTransport
  };
}
