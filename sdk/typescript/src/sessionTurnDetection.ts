export type TurnDetection = {
  create_response?: boolean;
  interrupt_response?: boolean;
} & ({
  type: 'server_vad';
  threshold?: number;
  prefix_padding_ms?: number;
  silence_duration_ms?: number;
  eagerness?: never;
} | {
  type: 'semantic_vad';
  eagerness?: 'auto' | 'low' | 'medium' | 'high';
  threshold?: never;
  prefix_padding_ms?: never;
  silence_duration_ms?: never;
});

export function normalizeTurnDetection(
  provider: string, transport: string, config?: TurnDetection
): TurnDetection | undefined {
  if (provider !== 'openai' || !['webrtc', 'websocket'].includes(transport)) {
    if (config !== undefined) {
      throw new Error('turn_detection requires provider=openai and transport=webrtc or websocket');
    }
    return undefined;
  }
  if (config === undefined) config = { type: 'server_vad' };
  if (config === null || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('turn_detection must be an object');
  }
  const kind = config.type;
  const common = ['type', 'create_response', 'interrupt_response'];
  let allowed: string[];
  if (kind === 'server_vad') {
    allowed = [...common, 'threshold', 'prefix_padding_ms', 'silence_duration_ms'];
  } else if (kind === 'semantic_vad') {
    allowed = [...common, 'eagerness'];
  } else {
    throw new Error('turn_detection.type must be server_vad or semantic_vad');
  }
  const supplied: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(config)) {
    if (!allowed.includes(key)) throw new Error(`turn_detection.${key} is unsupported for ${kind}`);
    if (value === undefined) continue;
    if (['create_response', 'interrupt_response'].includes(key) && typeof value !== 'boolean') {
      throw new Error(`turn_detection.${key} must be a boolean`);
    }
    if (key === 'threshold' && (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)) {
      throw new Error('turn_detection.threshold must be a finite number between 0 and 1');
    }
    if (['prefix_padding_ms', 'silence_duration_ms'].includes(key) &&
        (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0)) {
      throw new Error(`turn_detection.${key} must be a non-negative integer`);
    }
    if (key === 'eagerness' && !['auto', 'low', 'medium', 'high'].includes(value as string)) {
      throw new Error('turn_detection.eagerness must be auto, low, medium, or high');
    }
    supplied[key] = value;
  }
  const defaults = kind === 'server_vad'
    ? { threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 500 }
    : { eagerness: 'auto' };
  return { ...defaults, create_response: true, interrupt_response: true, ...supplied } as TurnDetection;
}
