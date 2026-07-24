import { describe, it, expect } from 'vitest';
import {
  capabilitiesToTools,
  type ToolCallConfig,
  type ToolCallTrace,
  type ToolCallRecord
} from '../src/ai/ToolCalling.js';

// We can't test the internal invocationTargetToCallTarget directly since it's not exported,
// but we test it indirectly through capabilitiesToTools and the types.

describe('ToolCalling Types', () => {
  it('ToolCallConfig has correct defaults', () => {
    const config: ToolCallConfig = {};
    expect(config.maxTurns).toBeUndefined();
    expect(config.maxToolCalls).toBeUndefined();
    expect(config.schemaHydration).toBeUndefined();
  });

  it('ToolCallConfig accepts all fields', () => {
    const config: ToolCallConfig = {
      maxTurns: 8,
      maxToolCalls: 12,
      maxCandidateTools: 30,
      maxHydratedTools: 8,
      schemaHydration: 'lazy',
      fallbackBroadening: true,
      tags: ['weather'],
      agentIds: ['worker-1'],
      healthStatus: 'healthy'
    };
    expect(config.maxTurns).toBe(8);
    expect(config.schemaHydration).toBe('lazy');
    expect(config.tags).toEqual(['weather']);
  });

  it('ToolCallTrace records calls', () => {
    const record: ToolCallRecord = {
      toolName: 'worker.get_weather',
      arguments: { city: 'Tokyo' },
      result: { temp_f: 81 },
      latencyMs: 42,
      turn: 0
    };
    const trace: ToolCallTrace = {
      calls: [record],
      totalTurns: 1,
      totalToolCalls: 1,
      finalResponse: 'The weather in Tokyo is sunny at 81F.'
    };
    expect(trace.calls).toHaveLength(1);
    expect(trace.calls[0].error).toBeUndefined();
  });

  it('ToolCallRecord can record errors', () => {
    const record: ToolCallRecord = {
      toolName: 'worker.get_weather',
      arguments: { city: 'Atlantis' },
      error: 'City not found',
      latencyMs: 5,
      turn: 0
    };
    expect(record.error).toBe('City not found');
    expect(record.result).toBeUndefined();
  });
});

describe('capabilitiesToTools', () => {
  // These tests need a mock Agent, which is complex to set up.
  // We test the type signatures and basic structure instead.

  it('returns empty object for empty capabilities', () => {
    // capabilitiesToTools needs an Agent instance, so we'll test with a minimal mock
    const mockAgent = {
      call: async () => ({})
    } as any;

    const tools = capabilitiesToTools([], mockAgent);
    expect(Object.keys(tools)).toHaveLength(0);
  });

  it('converts AgentCapability with reasoners and skills', () => {
    const mockAgent = {
      call: async () => ({})
    } as any;

    const caps = [{
      agentId: 'worker',
      baseUrl: 'http://localhost:8001',
      version: '1.0.0',
      healthStatus: 'healthy',
      reasoners: [{
        id: 'summarize',
        tags: ['text'],
        inputSchema: { type: 'object', properties: { text: { type: 'string' } } },
        invocationTarget: 'worker:summarize'
      }],
      skills: [{
        id: 'get_weather',
        tags: ['weather'],
        inputSchema: { type: 'object', properties: { city: { type: 'string' } } },
        invocationTarget: 'worker:skill:get_weather'
      }]
    }];

    const tools = capabilitiesToTools(caps, mockAgent);
    expect(Object.keys(tools)).toHaveLength(2);
    expect(tools['worker__summarize']).toBeDefined();
    expect(tools['worker__skill__get_weather']).toBeDefined();
  });

  it('handles metadata-only mode', () => {
    const mockAgent = {
      call: async () => ({})
    } as any;

    const caps = [{
      agentId: 'worker',
      baseUrl: 'http://localhost:8001',
      version: '1.0.0',
      healthStatus: 'healthy',
      reasoners: [{
        id: 'summarize',
        tags: ['text'],
        inputSchema: { type: 'object', properties: { text: { type: 'string' } } },
        invocationTarget: 'worker:summarize'
      }],
      skills: []
    }];

    const tools = capabilitiesToTools(caps, mockAgent, true);
    expect(Object.keys(tools)).toHaveLength(1);
    expect(tools['worker__summarize']).toBeDefined();
  });
});
