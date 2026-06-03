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
