import { afterEach, describe, expect, it, vi } from 'vitest';

const {
  generateTextMock,
  toolMock,
  jsonSchemaMock,
  stepCountIsMock
} = vi.hoisted(() => ({
  generateTextMock: vi.fn(),
  toolMock: vi.fn((definition: Record<string, unknown>) => definition),
  jsonSchemaMock: vi.fn((schema: unknown) => schema),
  stepCountIsMock: vi.fn((count: number) => ({ type: 'step-count', count }))
}));

vi.mock('ai', () => ({
  generateText: generateTextMock,
  tool: toolMock,
  jsonSchema: jsonSchemaMock,
  stepCountIs: stepCountIsMock
}));

import {
  buildToolConfig,
  capabilitiesToTools,
  capabilityToMetadataTool,
  capabilityToTool,
  executeToolCallLoop
} from '../src/ai/ToolCalling.js';
import type {
  AgentCapability,
  DiscoveryResult,
  ReasonerCapability,
  SkillCapability
} from '../src/types/agent.js';

function makeReasoner(invocationTarget: string, inputSchema?: object): ReasonerCapability {
  return {
    id: invocationTarget.split(':').at(-1) ?? invocationTarget,
    description: `reasoner ${invocationTarget}`,
    tags: ['reasoner'],
    inputSchema,
    invocationTarget
  };
}

function makeSkill(invocationTarget: string, inputSchema?: object): SkillCapability {
  return {
    id: invocationTarget.split(':').at(-1) ?? invocationTarget,
    description: `skill ${invocationTarget}`,
    tags: ['skill'],
    inputSchema,
    invocationTarget
  };
}

function makeAgentCapability(
  reasoners: ReasonerCapability[],
  skills: SkillCapability[]
): AgentCapability {
  return {
    agentId: 'node',
    baseUrl: 'http://localhost:8001',
    version: '1.0.0',
    healthStatus: 'healthy',
    reasoners,
    skills
  };
}

describe('ToolCalling extra coverage', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    generateTextMock.mockReset();
    toolMock.mockClear();
    jsonSchemaMock.mockClear();
    stepCountIsMock.mockClear();
  });

  it('creates executable tools for reasoners and skills', async () => {
    const agent = {
      call: vi.fn().mockResolvedValue({ ok: true })
    };

    const reasoner = makeReasoner('worker:summarize', { properties: { text: { type: 'string' } } });
    const reasonerTool = capabilityToTool(reasoner, agent as never);
    const reasonerExecute = (reasonerTool as unknown as { execute: (args: object) => Promise<unknown> }).execute;
    expect(jsonSchemaMock).toHaveBeenLastCalledWith({
      type: 'object',
      properties: {
        properties: { text: { type: 'string' } }
      }
    });
    await expect(reasonerExecute({ text: 'hi' })).resolves.toEqual({ ok: true });
    expect(agent.call).toHaveBeenCalledWith('worker.summarize', { text: 'hi' });

    const skill = makeSkill('worker:skill:get_weather');
    const skillTool = capabilityToTool(skill, agent as never);
    const skillExecute = (skillTool as unknown as { execute: (args: object) => Promise<unknown> }).execute;
    await skillExecute({ city: 'SF' });
    expect(agent.call).toHaveBeenCalledWith('worker.get_weather', { city: 'SF' });
    expect(jsonSchemaMock).toHaveBeenLastCalledWith({ type: 'object', properties: {} });
  });

  it('creates metadata-only tools and converts mixed capability arrays', () => {
    const agent = { call: vi.fn() };
    const metadataTool = capabilityToMetadataTool(makeSkill('worker:skill:lookup', { type: 'string' }), agent as never);
    expect(jsonSchemaMock).toHaveBeenCalledWith({ type: 'object', properties: {} });
    expect(metadataTool).toBeDefined();

    const tools = capabilitiesToTools(
      [
        makeAgentCapability([makeReasoner('worker:sum')], [makeSkill('worker:skill:lookup')]),
        makeReasoner('solo:echo'),
        makeSkill('solo:skill:run')
      ],
      agent as never
    );

    expect(Object.keys(tools)).toEqual([
      'worker__sum',
      'worker__skill__lookup',
      'solo__echo',
      'solo__skill__run'
    ]);
  });

  it('builds tool config from discover, config, discovery result, capability arrays, and raw maps', async () => {
    const discover = vi.fn()
      .mockResolvedValueOnce({
        json: {
          capabilities: [makeAgentCapability([makeReasoner('math:add')], [makeSkill('math:skill:mul')])]
        }
      })
      .mockResolvedValueOnce({
        json: {
          capabilities: [makeAgentCapability([makeReasoner('math:add')], [])]
        }
      });
    const agent = { discover, call: vi.fn() };

    const discovered = await buildToolConfig('discover', agent as never);
    expect(Object.keys(discovered.tools)).toEqual(['math__add', 'math__skill__mul']);
    expect(discovered.needsLazyHydration).toBe(false);
    expect(discover).toHaveBeenNthCalledWith(1, expect.objectContaining({
      includeInputSchema: true,
      includeDescriptions: true
    }));

    const lazy = await buildToolConfig({
      tags: ['math'],
      schemaHydration: 'lazy',
      maxCandidateTools: 1,
      maxTurns: 7,
      maxToolCalls: 9
    }, agent as never);
    expect(lazy.needsLazyHydration).toBe(true);
    expect(lazy.config.maxTurns).toBe(7);
    expect(Object.keys(lazy.tools)).toEqual(['math__add']);
    expect(discover).toHaveBeenNthCalledWith(2, expect.objectContaining({
      tags: ['math'],
      includeInputSchema: false
    }));

    const discoveryResult: DiscoveryResult = {
      format: 'json',
      raw: '{}',
      json: {
        discoveredAt: '2026-01-01T00:00:00Z',
        totalAgents: 1,
        totalReasoners: 1,
        totalSkills: 0,
        pagination: { limit: 10, offset: 0, hasMore: false },
        capabilities: [makeAgentCapability([makeReasoner('node:echo')], [])]
      }
    };
    const fromDiscovery = await buildToolConfig(discoveryResult, agent as never);
    expect(Object.keys(fromDiscovery.tools)).toEqual(['node__echo']);

    const emptyDiscovery = await buildToolConfig({ format: 'json', raw: '{}' } as DiscoveryResult, agent as never);
    expect(emptyDiscovery.tools).toEqual({});

    const fromArray = await buildToolConfig([makeReasoner('solo:run')], agent as never);
    expect(Object.keys(fromArray.tools)).toEqual(['solo__run']);

    const rawMap = { passthrough: { description: 'raw tool' } };
    const fromRaw = await buildToolConfig(rawMap as never, agent as never);
    expect(fromRaw.tools).toBe(rawMap);
  });

  it('rejects invalid tool config input', async () => {
    await expect(buildToolConfig(123 as never, { discover: vi.fn(), call: vi.fn() } as never)).rejects.toThrow(
      'Invalid tools parameter'
    );
  });

  it('hydrates only LLM-selected tools during lazy execution', async () => {
    const discover = vi.fn().mockResolvedValue({
      json: {
        capabilities: [
          makeAgentCapability(
            [makeReasoner('worker:sum'), makeReasoner('worker:diff')],
            [makeSkill('worker:skill:weather')]
          )
        ]
      }
    });
    const agent = {
      discover,
      call: vi.fn().mockResolvedValue({ total: 3 })
    };

    generateTextMock
      .mockResolvedValueOnce({
        text: '',
        steps: [
          {
            toolCalls: [
              { toolName: 'worker__sum' },
              { toolName: 'worker__skill__weather' }
            ]
          }
        ]
      })
      .mockImplementationOnce(async (options: {
        tools: Record<string, { execute: (args: object) => Promise<unknown> }>;
        onStepFinish?: () => void;
      }) => {
        expect(Object.keys(options.tools)).toEqual(['worker__sum']);
        const result = await options.tools.worker__sum.execute({ a: 1, b: 2 });
        options.onStepFinish?.();
        return {
          text: `answer ${JSON.stringify(result)}`,
          steps: [{ toolCalls: [{ toolName: 'worker__sum' }] }]
        };
      });

    const result = await executeToolCallLoop(
      agent as never,
      'prompt',
      {
        worker__sum: { description: 'sum', inputSchema: {} },
        worker__diff: { description: 'diff', inputSchema: {} },
        worker__skill__weather: { description: 'weather', inputSchema: {} }
      } as never,
      { maxHydratedTools: 1 },
      true,
      () => ({ id: 'model' })
    );

    expect(discover).toHaveBeenCalledTimes(1);
    expect(stepCountIsMock).toHaveBeenNthCalledWith(1, 1);
    expect(stepCountIsMock).toHaveBeenNthCalledWith(2, 10);
    expect(agent.call).toHaveBeenCalledWith('worker.sum', { a: 1, b: 2 });
    expect(result.text).toContain('answer');
    expect(result.trace.calls).toEqual([
      expect.objectContaining({
        toolName: 'worker__sum',
        result: { total: 3 },
        turn: 0
      })
    ]);
  });

  it('records observable tool failures and limit overflows', async () => {
    const agent = {
      call: vi.fn()
        .mockResolvedValueOnce({ ok: 1 })
        .mockRejectedValueOnce(new Error('boom'))
    };

    generateTextMock.mockImplementationOnce(async (options: {
      tools: Record<string, { execute: (args: object) => Promise<unknown> }>;
      onStepFinish?: () => void;
    }) => {
      const first = await options.tools.worker__sum.execute({ value: 1 });
      const second = await options.tools.worker__sum.execute({ value: 2 });
      const third = await options.tools.worker__sum.execute({ value: 3 });
      expect(first).toEqual({ ok: 1 });
      expect(second).toEqual({ error: 'boom', tool: 'worker__sum' });
      expect(third).toEqual({ error: 'Tool call limit reached. Please provide a final response.' });
      options.onStepFinish?.();
      return {
        text: 'done',
        steps: [{ toolCalls: [{ toolName: 'worker__sum' }] }]
      };
    });

    const result = await executeToolCallLoop(
      agent as never,
      'prompt',
      {
        worker__sum: { description: 'sum', inputSchema: {} }
      } as never,
      { maxTurns: 2, maxToolCalls: 2 },
      false,
      () => ({ id: 'model' })
    );

    expect(agent.call).toHaveBeenCalledTimes(2);
    expect(result.trace.totalTurns).toBe(1);
    expect(result.trace.totalToolCalls).toBe(3);
    expect(result.trace.calls).toEqual([
      expect.objectContaining({ toolName: 'worker__sum', result: { ok: 1 } }),
      expect.objectContaining({ toolName: 'worker__sum', error: 'boom' }),
      expect.objectContaining({ toolName: 'worker__sum', error: 'Tool call limit reached' })
    ]);
    expect(result.trace.finalResponse).toBe('done');
  });
});
