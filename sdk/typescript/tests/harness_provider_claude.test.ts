import { afterEach, describe, expect, it, vi } from 'vitest';

describe('ClaudeCodeProvider', () => {
  afterEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    vi.unmock('@anthropic-ai/claude-agent-sdk');
  });

  it('maps options, streams messages, and extracts final result metrics', async () => {
    const captured: { prompt?: string; options?: Record<string, unknown> } = {};

    vi.doMock(
      '@anthropic-ai/claude-agent-sdk',
      () => ({
        query: ({ prompt, options }: { prompt: string; options: Record<string, unknown> }) => {
          captured.prompt = prompt;
          captured.options = options;
          return (async function* stream() {
            yield { type: 'assistant', content: [{ type: 'text', text: 'intermediate' }] };
            yield { type: 'result', result: 'final', session_id: 'sess-1', cost_usd: 0.2, num_turns: 4 };
          })();
        },
      }),
      { virtual: true }
    );

    const { ClaudeCodeProvider } = await import('../src/harness/providers/claude.js');
    const provider = new ClaudeCodeProvider();
    const raw = await provider.execute('hello', {
      model: 'sonnet',
      cwd: '/tmp/work',
      maxTurns: 8,
      tools: ['Read', 'Write'],
      systemPrompt: 'system',
      maxBudgetUsd: 3,
      permissionMode: 'auto',
      env: { A: '1' },
    });

    expect(captured.prompt).toBe('hello');
    expect(captured.options).toEqual({
      model: 'sonnet',
      cwd: '/tmp/work',
      max_turns: 8,
      allowed_tools: ['Read', 'Write'],
      system_prompt: 'system',
      max_budget_usd: 3,
      permission_mode: 'bypassPermissions',
      env: { A: '1' },
    });
    expect(raw.isError).toBe(false);
    expect(raw.result).toBe('final');
    expect(raw.metrics.totalCostUsd).toBe(0.2);
    expect(raw.metrics.numTurns).toBe(4);
    expect(raw.metrics.sessionId).toBe('sess-1');
    expect(raw.messages).toHaveLength(2);
  });

  it('strips a #variant model suffix before handing the model to the SDK', async () => {
    const captured: { options?: Record<string, unknown> } = {};

    vi.doMock(
      '@anthropic-ai/claude-agent-sdk',
      () => ({
        query: ({ options }: { prompt: string; options: Record<string, unknown> }) => {
          captured.options = options;
          return (async function* stream() {
            yield { type: 'result', result: 'final', session_id: 'sess-1', cost_usd: 0.01, num_turns: 1 };
          })();
        },
      }),
      { virtual: true }
    );

    const { ClaudeCodeProvider } = await import('../src/harness/providers/claude.js');
    const provider = new ClaudeCodeProvider();
    const raw = await provider.execute('hello', { model: 'sonnet#high' });

    expect(captured.options?.model).toBe('sonnet');
    expect(raw.isError).toBe(false);
  });

  it('extracts token usage and model from the result message', async () => {
    vi.doMock(
      '@anthropic-ai/claude-agent-sdk',
      () => ({
        query: () =>
          (async function* stream() {
            yield {
              type: 'assistant',
              message: { model: 'claude-opus-4-8', content: [{ type: 'text', text: 'working' }] },
            };
            yield {
              type: 'result',
              result: 'final',
              session_id: 'sess-2',
              total_cost_usd: 0.5,
              num_turns: 2,
              usage: {
                input_tokens: 900,
                output_tokens: 100,
                cache_read_input_tokens: 30,
                cache_creation_input_tokens: 7,
              },
            };
          })(),
      }),
      { virtual: true }
    );

    const { ClaudeCodeProvider } = await import('../src/harness/providers/claude.js');
    const provider = new ClaudeCodeProvider();
    const raw = await provider.execute('hello', {});

    expect(raw.isError).toBe(false);
    expect(raw.metrics.totalCostUsd).toBe(0.5);
    expect(raw.metrics.inputTokens).toBe(900);
    expect(raw.metrics.outputTokens).toBe(100);
    expect(raw.metrics.cacheReadTokens).toBe(30);
    expect(raw.metrics.cacheCreationTokens).toBe(7);
    expect(raw.metrics.model).toBe('claude-opus-4-8');
    expect(raw.metrics.usage).toEqual({
      input_tokens: 900,
      output_tokens: 100,
      cache_read_input_tokens: 30,
      cache_creation_input_tokens: 7,
    });
  });

  it('leaves token metrics undefined when the result message has no usage', async () => {
    vi.doMock(
      '@anthropic-ai/claude-agent-sdk',
      () => ({
        query: () =>
          (async function* stream() {
            yield { type: 'result', result: 'final', session_id: 'sess-3' };
          })(),
      }),
      { virtual: true }
    );

    const { ClaudeCodeProvider } = await import('../src/harness/providers/claude.js');
    const provider = new ClaudeCodeProvider();
    const raw = await provider.execute('hello', {});

    expect(raw.isError).toBe(false);
    expect(raw.metrics.inputTokens).toBeUndefined();
    expect(raw.metrics.outputTokens).toBeUndefined();
    expect(raw.metrics.usage).toBeUndefined();
  });

  it('returns error result when SDK stream fails', async () => {
    vi.doMock(
      '@anthropic-ai/claude-agent-sdk',
      () => ({
        query: () =>
          (async function* stream() {
            throw new Error('sdk exploded');
          })(),
      }),
      { virtual: true }
    );

    const { ClaudeCodeProvider } = await import('../src/harness/providers/claude.js');
    const provider = new ClaudeCodeProvider();
    const raw = await provider.execute('hello', {});

    expect(raw.isError).toBe(true);
    expect(raw.result).toBeUndefined();
    expect(raw.errorMessage).toBe('sdk exploded');
    expect(raw.metrics.durationApiMs).toBeGreaterThanOrEqual(0);
  });

  it('throws helpful error when SDK is not installed', async () => {
    vi.doMock(
      '@anthropic-ai/claude-agent-sdk',
      () => {
        throw new Error('module not found');
      },
      { virtual: true }
    );
    const { ClaudeCodeProvider } = await import('../src/harness/providers/claude.js');
    const provider = new ClaudeCodeProvider();

    await expect(provider.execute('hello', {})).rejects.toThrow(/npm install @anthropic-ai\/claude-agent-sdk/);
  });
});

describe('buildProvider', () => {
  afterEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
  });

  it('routes claude-code to ClaudeCodeProvider', async () => {
    const { buildProvider } = await import('../src/harness/providers/factory.js');
    const { ClaudeCodeProvider } = await import('../src/harness/providers/claude.js');

    const provider = await buildProvider({ provider: 'claude-code' });
    expect(provider).toBeInstanceOf(ClaudeCodeProvider);
  });
});
