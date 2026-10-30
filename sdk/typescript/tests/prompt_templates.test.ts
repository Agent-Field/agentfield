import { describe, it, expect, vi } from 'vitest';

import {
  resolvePromptTemplates,
  defaultToolErrorFormatter,
  defaultToolResultFormatter,
  DEFAULT_TOOL_CALL_LIMIT_REACHED,
  TRACE_SOURCE_USER,
  TRACE_SOURCE_ASSISTANT,
  TRACE_SOURCE_TOOL_RESULT,
  TRACE_SOURCE_TOOL_ERROR,
  TRACE_SOURCE_TOOL_LIMIT,
  TRACE_SOURCE_TOOL_SYSTEM_PROMPT,
} from '../src/ai/PromptTemplates.js';
import { executeToolCallLoop, buildToolConfig } from '../src/ai/ToolCalling.js';

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

  it('restores the default for an empty-string tool-call-limit message', () => {
    // ?? would keep '', so an empty override must still fall back (review #1092).
    expect(resolvePromptTemplates({ toolCallLimitReached: '' }).toolCallLimitReached).toBe(
      DEFAULT_TOOL_CALL_LIMIT_REACHED
    );
  });
});

describe('empty tool-call-limit through the loop', () => {
  it('sends the default limit message, not an empty one', async () => {
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
        promptTemplates: { toolCallLimitReached: '' }
      },
      false,
      () => ({ provider: 'mock' })
    );

    expect(toolOutputs[0]).toEqual({ error: DEFAULT_TOOL_CALL_LIMIT_REACHED });
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
      return {
        text: 'done',
        steps: [{ toolCalls: [{ toolName: 'node__sum' }] }],
        response: {
          messages: [
            {
              role: 'assistant',
              content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'node__sum' }]
            },
            {
              role: 'tool',
              content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'node__sum' }]
            },
            { role: 'assistant', content: 'done' }
          ]
        }
      };
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
    // The injected tool system prompt and user prompt are tagged up front;
    // the tool result comes from the AI SDK's response.messages.
    expect(sources).toContain(TRACE_SOURCE_TOOL_SYSTEM_PROMPT);
    expect(sources).toContain(TRACE_SOURCE_USER);
    expect(sources).toContain(TRACE_SOURCE_TOOL_RESULT);
  });

  it('tags tool errors in the trace', async () => {
    const agent = { call: vi.fn().mockRejectedValue(new Error('boom')) } as any;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async (options: any) => {
      // Pass the tool-call id to execute(), as the AI SDK does, so provenance
      // is recorded against it.
      await options.tools.node__sum.execute({ v: 1 }, { toolCallId: 'c1' });
      return {
        text: 'handled',
        steps: [{ toolCalls: [{ toolName: 'node__sum' }] }],
        response: {
          messages: [
            {
              role: 'assistant',
              content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'node__sum' }]
            },
            {
              role: 'tool',
              content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'node__sum' }]
            },
            { role: 'assistant', content: 'handled' }
          ]
        }
      };
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
    expect(sources).toContain(TRACE_SOURCE_ASSISTANT);
    // A failed tool call is tagged sdk.tool_error, not sdk.tool_result.
    expect(sources).toContain(TRACE_SOURCE_TOOL_ERROR);
    expect(sources).not.toContain(TRACE_SOURCE_TOOL_RESULT);
    const toolMsg = (result.trace.messages ?? []).find(
      (m) => m.source === TRACE_SOURCE_TOOL_ERROR
    );
    expect((toolMsg?.message as any).content[0].toolCallId).toBe('c1');
  });

  it('tags a tool-limit response as sdk.tool_limit', async () => {
    const agent = { call: vi.fn().mockResolvedValue({ ok: true }) } as any;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async (options: any) => {
      await options.tools.node__sum.execute({ v: 1 }, { toolCallId: 'c1' });
      return {
        text: 'done',
        steps: [{ toolCalls: [{ toolName: 'node__sum' }] }],
        response: {
          messages: [
            {
              role: 'tool',
              content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'node__sum' }]
            }
          ]
        }
      };
    });

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      { maxTurns: 2, maxToolCalls: 0 },
      false,
      () => ({ provider: 'mock' })
    );

    const sources = (result.trace.messages ?? []).map((m) => m.source);
    expect(sources).toContain(TRACE_SOURCE_TOOL_LIMIT);
  });

  it('tags a caller-only system prompt as user content', async () => {
    const agent = { call: vi.fn().mockResolvedValue({ ok: true }) } as any;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async () => {
      return {
        text: 'done',
        steps: [],
        response: { messages: [{ role: 'assistant', content: 'done' }] }
      };
    });

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      { maxTurns: 2, maxToolCalls: 3 },
      false,
      () => ({ provider: 'mock' }),
      { system: 'You are helpful.' }
    );

    // A caller system prompt (no tool system prompt) is caller content.
    const systemMsg = (result.trace.messages ?? []).find(
      (m) => (m.message as any).role === 'system'
    );
    expect(systemMsg?.source).toBe(TRACE_SOURCE_USER);
    expect((systemMsg?.message as any).content).toBe('You are helpful.');
  });

  it('tags the user prompt and the actual response messages', async () => {
    const agent = { call: vi.fn().mockResolvedValue({ ok: true }) } as any;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async () => {
      return {
        text: 'done',
        steps: [],
        response: { messages: [{ role: 'assistant', content: 'done' }] }
      };
    });

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      { maxTurns: 2, maxToolCalls: 3 },
      false,
      () => ({ provider: 'mock' })
    );

    // The user prompt is tagged up front; the assistant message comes from the
    // AI SDK's response.messages (review #1092).
    expect(result.trace.messages).toEqual([
      { message: { role: 'user', content: 'prompt' }, source: TRACE_SOURCE_USER },
      { message: { role: 'assistant', content: 'done' }, source: TRACE_SOURCE_ASSISTANT }
    ]);
  });
});

describe('buildToolConfig config detection', () => {
  it('treats a promptTemplates-only object as a ToolCallConfig and discovers', async () => {
    const agent = {
      discover: vi.fn().mockResolvedValue({ agents: [] })
    } as any;

    const { config } = await buildToolConfig(
      { promptTemplates: { toolCallLimitReached: 'stop' } } as any,
      agent
    );

    // Recognized as a config (discovery ran), not misread as a raw tool map.
    expect(agent.discover).toHaveBeenCalled();
    expect(config.promptTemplates).toEqual({ toolCallLimitReached: 'stop' });
  });
});

describe('native SDK tool errors (no wrapper entry)', () => {
  it('classifies an SDK-generated error-text tool result as sdk.tool_error', async () => {
    // The AI SDK rejects malformed/schema-invalid args before execute runs, so
    // no wrapper provenance exists; the error output type must classify it.
    const agent = { call: vi.fn() } as any;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async () => ({
      text: 'done',
      steps: [],
      response: {
        messages: [
          {
            role: 'assistant',
            content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'node__sum' }]
          },
          {
            role: 'tool',
            content: [
              {
                type: 'tool-result',
                toolCallId: 'c1',
                toolName: 'node__sum',
                output: { type: 'error-text', value: 'invalid arguments' }
              }
            ]
          },
          { role: 'assistant', content: 'done' }
        ]
      }
    }));

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      { maxTurns: 2, maxToolCalls: 3 },
      false,
      () => ({ provider: 'mock' })
    );

    const toolMsg = (result.trace.messages ?? []).find(
      (m) => (m.message as any).role === 'tool'
    );
    expect(toolMsg?.source).toBe(TRACE_SOURCE_TOOL_ERROR);
    // agent.call never ran (the SDK rejected before execute).
    expect(agent.call).not.toHaveBeenCalled();
  });

  it('keeps a successful result whose output contains an error field as a result', async () => {
    const agent = { call: vi.fn() } as any;

    generateTextMock.mockReset();
    generateTextMock.mockImplementationOnce(async () => ({
      text: 'done',
      steps: [],
      response: {
        messages: [
          {
            role: 'tool',
            content: [
              {
                type: 'tool-result',
                toolCallId: 'c1',
                toolName: 'node__sum',
                // Successful business output that happens to carry an error key.
                output: { type: 'json', value: { error: 'handled gracefully', ok: true } }
              }
            ]
          }
        ]
      }
    }));

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      toolMap,
      { maxTurns: 2, maxToolCalls: 3 },
      false,
      () => ({ provider: 'mock' })
    );

    const toolMsg = (result.trace.messages ?? []).find(
      (m) => (m.message as any).role === 'tool'
    );
    expect(toolMsg?.source).toBe(TRACE_SOURCE_TOOL_RESULT);
  });
});
