export interface HarnessConfig {
  provider: 'claude-code' | 'codex' | 'gemini' | 'opencode';
  model?: string;
  /**
   * Provider-specific reasoning-effort variant (e.g. `high`, `minimal`).
   * Wins over a `#variant` suffix on `model`.
   */
  variant?: string;
  maxTurns?: number;
  maxBudgetUsd?: number;
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffFactor?: number;
  tools?: string[];
  permissionMode?: string;
  systemPrompt?: string;
  env?: Record<string, string>;
  cwd?: string;
  codexBin?: string;
  geminiBin?: string;
  opencodeBin?: string;
}

export interface HarnessOptions {
  provider?: string;
  model?: string;
  /**
   * Provider-specific reasoning-effort variant (e.g. `high`, `minimal`).
   * Wins over a `#variant` suffix on `model`.
   */
  variant?: string;
  maxTurns?: number;
  maxBudgetUsd?: number;
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffFactor?: number;
  tools?: string[];
  permissionMode?: string;
  systemPrompt?: string;
  env?: Record<string, string>;
  cwd?: string;
  codexBin?: string;
  geminiBin?: string;
  opencodeBin?: string;
  schema?: unknown;
}

export interface Metrics {
  durationMs: number;
  durationApiMs: number;
  numTurns: number;
  totalCostUsd?: number;
  usage?: Record<string, unknown>;
  sessionId: string;
  /** Token counts parsed from the provider's result payload (best effort). */
  inputTokens?: number;
  outputTokens?: number;
  cacheReadTokens?: number;
  cacheCreationTokens?: number;
  totalTokens?: number;
  /** Model reported by the provider, when available. */
  model?: string;
}

export interface RawResult {
  result?: string;
  messages: Array<Record<string, unknown>>;
  metrics: Metrics;
  isError: boolean;
  errorMessage?: string;
}

export interface HarnessResult {
  result?: string;
  parsed?: unknown;
  isError: boolean;
  errorMessage?: string;
  costUsd?: number;
  numTurns: number;
  durationMs: number;
  sessionId: string;
  messages: Array<Record<string, unknown>>;
  /** Token counts reported by the harness provider, when available. */
  inputTokens?: number;
  outputTokens?: number;
  cacheReadTokens?: number;
  cacheCreationTokens?: number;
  totalTokens?: number;
  /** Model reported by the harness provider, when available. */
  model?: string;
  readonly text: string;
}

export function createHarnessResult(partial?: Partial<Omit<HarnessResult, 'text'>>): HarnessResult {
  const r = {
    isError: false,
    numTurns: 0,
    durationMs: 0,
    sessionId: '',
    messages: [],
    ...partial,
    get text(): string {
      return this.result ?? '';
    },
  };
  return r;
}

export function createMetrics(partial?: Partial<Metrics>): Metrics {
  return { durationMs: 0, durationApiMs: 0, numTurns: 0, sessionId: '', ...partial };
}

export function createRawResult(partial?: Partial<RawResult>): RawResult {
  return { messages: [], metrics: createMetrics(), isError: false, ...partial };
}
