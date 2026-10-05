import { describe, it, expect, vi } from 'vitest';

import {
  resolvePromptTemplates,
  defaultToolErrorFormatter,
  defaultToolResultFormatter,
  DEFAULT_TOOL_CALL_LIMIT_REACHED,
  TRACE_SOURCE_ASSISTANT,
  TRACE_SOURCE_TOOL_RESULT,
  TRACE_SOURCE_TOOL_ERROR,
  TRACE_SOURCE_TOOL_SYSTEM_PROMPT,
} from '../src/ai/PromptTemplates.js';
import { executeToolCallLoop } from '../src/ai/ToolCalling.js';

const { generateTextMock } = vi.hoisted(() => ({
  generateTextMock: vi.fn(),
}));

vi.mock('ai', () => ({
  generateText: generateTextMock,
  tool: (def: any) => def,
  jsonSchema: (s: any) => s,
  stepCountIs: (n: number) => n,
}));

const toolMap = {
  node__sum: {
    description: 'sum',
    inputSchema: { type: 'object', properties: {} },
  },
} as any;

describe('resolvePromptTemplates', () => {
  it('fills defaults and leaves toolSystemPrompt unset', () => {
    const resolved = resolvePromptTemplates();
    expect(resolved.toolCallLimitReached).toBe(DEFAULT_TOOL_CALL_LIMIT_REACHED);
    expect(resolved.toolErrorFormatter).toBe(defaultToolErrorFormatter);
    expect(resolved.toolResultFormatter).toBe(defaultToolResultFormatter);
    expect(resolved.toolSystemPrompt).toBeUndefined();
  });

  it('default formatters reproduce the historical framing', () => {
    expect(defaultToolErrorFormatter('node__sum', 'boom')).toEqual({
      error: 'boom',
      tool: 'node__sum',
    });
    expect(defaultToolResultFormatter('node__sum', { ok: true })).toEqual({
      ok: true,
    });
  });

  it('merges a partial override over defaults', () => {
    const resolved = resolvePromptTemplates({
      toolCallLimitReached: 'stop now',
    });
    expect(resolved.toolCallLimitReached).toBe('stop now');
    // Unset fields keep their defaults.
    expect(resolved.toolErrorFormatter).toBe(defaultToolErrorFormatter);
  });
});

describe('executeToolCallLoop with promptTemplates', () => {
  it('uses an overridden tool result formatter', async () => {
    const agent = { call: vi.fn().mockResolvedValue({ ok: true }) } as any;
    const toolOutputs: unknown[] = [];

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async (options: any) => {
      toolOutputs.push(await options.tools.node__sum.execute({ v: 1 }));
      return { text: 'done', steps: [{ toolCalls: [{ toolName: 'node__sum' }] }] };
    });

    await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      {
        maxTurns: 2,
        maxToolCalls: 3,
        promptTemplates: {
          toolResultFormatter: (tool, result) => `${tool}:${JSON.stringify(result)}`,
        },
      },
      false,
      () => ({ provider: 'mock' })
    );

    expect(toolOutputs[0]).toBe('node__sum:{"ok":true}');
  });

  it('uses an overridden tool-call-limit message', async () => {
    const agent = { call: vi.fn().mockResolvedValue({ ok: true }) } as any;
    const toolOutputs: unknown[] = [];

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async (options: any) => {
      toolOutputs.push(await options.tools.node__sum.execute({ v: 1 }));
      return { text: '', steps: [{ toolCalls: [{ toolName: 'node__sum' }] }] };
    });

    await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      {
        maxTurns: 2,
        maxToolCalls: 0,
        promptTemplates: { toolCallLimitReached: 'enough' },
      },
      false,
      () => ({ provider: 'mock' })
    );

    expect(toolOutputs[0]).toEqual({ error: 'enough' });
  });

  it('appends toolSystemPrompt after the caller system prompt', async () => {
    const agent = { call: vi.fn().mockResolvedValue({ ok: true }) } as any;
    let capturedSystem: string | undefined;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async (options: any) => {
      capturedSystem = options.system;
      return { text: 'done', steps: [] };
    });

    await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      {
        maxTurns: 2,
        maxToolCalls: 3,
        promptTemplates: { toolSystemPrompt: 'Use tools wisely.' },
      },
      false,
      () => ({ provider: 'mock' }),
      { system: 'You are helpful.' }
    );

    expect(capturedSystem).toBe('You are helpful.\n\nUse tools wisely.');
  });

  it('default (no templates) leaves system untouched', async () => {
    const agent = { call: vi.fn().mockResolvedValue({ ok: true }) } as any;
    let capturedSystem: string | undefined;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async (options: any) => {
      capturedSystem = options.system;
      return { text: 'done', steps: [] };
    });

    await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      { maxTurns: 2, maxToolCalls: 3 },
      false,
      () => ({ provider: 'mock' }),
      { system: 'You are helpful.' }
    );

    expect(capturedSystem).toBe('You are helpful.');
  });

  it('tags tool result and tool-system-prompt sources in the trace', async () => {
    const agent = { call: vi.fn().mockResolvedValue({ ok: true }) } as any;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async (options: any) => {
      await options.tools.node__sum.execute({ v: 1 });
      return { text: 'done', steps: [{ toolCalls: [{ toolName: 'node__sum' }] }] };
    });

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      {
        maxTurns: 2,
        maxToolCalls: 3,
        promptTemplates: { toolSystemPrompt: 'Use tools.' }
      },
      false,
      () => ({ provider: 'mock' })
    );

    const sources = (result.trace.messages ?? []).map((m) => m.source);
    expect(sources).toContain(TRACE_SOURCE_TOOL_SYSTEM_PROMPT);
    expect(sources).toContain(TRACE_SOURCE_TOOL_RESULT);
  });

  it('tags tool errors in the trace', async () => {
    const agent = { call: vi.fn().mockRejectedValue(new Error('boom')) } as any;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async (options: any) => {
      await options.tools.node__sum.execute({ v: 1 });
      return { text: 'handled', steps: [{ toolCalls: [{ toolName: 'node__sum' }] }] };
    });

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      { maxTurns: 2, maxToolCalls: 3 },
      false,
      () => ({ provider: 'mock' })
    );

    const sources = (result.trace.messages ?? []).map((m) => m.source);
    expect(sources).toContain(TRACE_SOURCE_TOOL_ERROR);
  });

  it('leaves trace.messages undefined-safe when nothing is injected', async () => {
    const agent = { call: vi.fn().mockResolvedValue({ ok: true }) } as any;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async () => {
      return { text: 'done', steps: [] };
    });

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      { maxTurns: 2, maxToolCalls: 3 },
      false,
      () => ({ provider: 'mock' })
    );

    // No tools executed and no tool system prompt, so messages stays empty.
    expect(result.trace.messages ?? []).toEqual([]);
    void TRACE_SOURCE_ASSISTANT;
  });
});
