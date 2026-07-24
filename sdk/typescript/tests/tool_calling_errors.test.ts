import { afterEach, describe, expect, it, vi } from 'vitest';

const { generateTextMock } = vi.hoisted(() => ({
  generateTextMock: vi.fn()
}));

vi.mock('ai', () => ({
  generateText: generateTextMock,
  tool: (definition: Record<string, unknown>) => definition,
  jsonSchema: (schema: unknown) => schema,
  stepCountIs: (count: number) => ({ type: 'step-count', count })
}));

import { buildToolConfig, executeToolCallLoop } from '../src/ai/ToolCalling.js';
import type { AgentCapability } from '../src/types/agent.js';

function makeAgentCapability(invocationTarget: string, tags: string[] = []): AgentCapability {
  return {
    agentId: 'node',
    baseUrl: 'http://localhost:8001',
    version: '1.0.0',
    healthStatus: 'healthy',
    reasoners: [
      {
        id: invocationTarget.split(':').at(-1) ?? invocationTarget,
        invocationTarget,
        tags,
        inputSchema: {
          type: 'object',
          properties: {
            value: { type: 'number' }
          }
        }
      }
    ],
    skills: []
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  generateTextMock.mockReset();
});

describe('ToolCalling branch behavior', () => {
  it('forwards discovery filters and returns only the discovered tool set', async () => {
    const discover = vi.fn().mockResolvedValue({
      json: {
        capabilities: [makeAgentCapability('math:add', ['math'])]
      }
    });
    const agent = { discover, call: vi.fn() } as any;

    const result = await buildToolConfig(
      {
        tags: ['math'],
        schemaHydration: 'lazy',
        maxCandidateTools: 5
      },
      agent
    );

    expect(discover).toHaveBeenCalledWith(
      expect.objectContaining({
        tags: ['math'],
        includeInputSchema: false,
        includeDescriptions: true
      })
    );
    expect(result.needsLazyHydration).toBe(true);
    expect(Object.keys(result.tools)).toEqual(['math__add']);
  });

  it('returns the first-pass text immediately when lazy hydration selects no tools', async () => {
    generateTextMock.mockResolvedValueOnce({
      text: 'final without tools',
      steps: []
    });

    const result = await executeToolCallLoop(
      { discover: vi.fn(), call: vi.fn() } as any,
      'prompt',
      {
        math__add: {
          description: 'add two numbers',
          inputSchema: { type: 'object', properties: {} }
        }
      } as any,
      { maxTurns: 3, maxToolCalls: 2 },
      true,
      () => ({ provider: 'mock' })
    );

    expect(generateTextMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual({
      text: 'final without tools',
      trace: {
        calls: [],
        totalTurns: 0,
        totalToolCalls: 0,
        finalResponse: 'final without tools'
      }
    });
  });

  it('captures tool execution failures as structured results and trace errors', async () => {
    const agent = {
      call: vi.fn().mockRejectedValue(new Error('boom'))
    } as any;
    const toolOutputs: unknown[] = [];

    generateTextMock.mockImplementationOnce(async (options: any) => {
      toolOutputs.push(await options.tools.node__sum.execute({ value: 2 }));
      options.onStepFinish?.();
      return {
        text: 'handled tool failure',
        steps: [{ toolCalls: [{ toolName: 'node__sum' }] }]
      };
    });

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      {
        node__sum: {
          description: 'sum',
          inputSchema: { type: 'object', properties: {} }
        }
      } as any,
      { maxTurns: 4, maxToolCalls: 3 },
      false,
      () => ({ provider: 'mock' })
    );

    expect(agent.call).toHaveBeenCalledWith('node.sum', { value: 2 });
    expect(toolOutputs[0]).toEqual({ error: 'boom', tool: 'node__sum' });
    expect(result.text).toBe('handled tool failure');
    expect(result.trace.calls).toEqual([
      expect.objectContaining({
        toolName: 'node__sum',
        arguments: { value: 2 },
        error: 'boom'
      })
    ]);
    expect(result.trace.totalTurns).toBe(1);
  });

  it('enforces maxToolCalls inside the observable wrapper', async () => {
    const agent = {
      call: vi.fn().mockResolvedValue({ ok: true })
    } as any;
    const toolOutputs: unknown[] = [];

    generateTextMock.mockImplementationOnce(async (options: any) => {
      toolOutputs.push(await options.tools.node__sum.execute({ value: 1 }));
      toolOutputs.push(await options.tools.node__sum.execute({ value: 2 }));
      options.onStepFinish?.();
      return {
        text: '',
        steps: [{ toolCalls: [{ toolName: 'node__sum' }, { toolName: 'node__sum' }] }]
      };
    });

    const result = await executeToolCallLoop(
      agent,
      'prompt',
      {
        node__sum: {
          description: 'sum',
          inputSchema: { type: 'object', properties: {} }
        }
      } as any,
      { maxTurns: 2, maxToolCalls: 1 },
      false,
      () => ({ provider: 'mock' })
    );

    expect(agent.call).toHaveBeenCalledTimes(1);
    expect(toolOutputs).toEqual([
      { ok: true },
      { error: 'Tool call limit reached. Please provide a final response.' }
    ]);
    expect(result.trace.totalToolCalls).toBe(2);
    expect(result.trace.calls[1]).toMatchObject({
      toolName: 'node__sum',
      error: 'Tool call limit reached'
    });
    expect(result.text).toBe('');
  });
});
