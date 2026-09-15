import { describe, expect, it } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import { buildSessionDefinition } from '../src/session.js';
import type { TurnDetection } from '../src/sessionTurnDetection.js';

describe('session turn detection', () => {
  it.each(['webrtc', 'websocket'])('registers and serializes %s options preserving false and zero', (transport) => {
    const agent = new Agent({ nodeId: 'support', devMode: true });
    const config: TurnDetection = {
      type: 'server_vad', threshold: 0, prefix_padding_ms: 0, silence_duration_ms: 750,
      create_response: false, interrupt_response: false
    };
    agent.session('voice', { provider: 'OpenAI', transport, turn_detection: config }, async () => ({}));
    const expected = { ...config };
    config.threshold = 1;
    expect(JSON.parse(JSON.stringify(agent.sessionDefinitions()))[0].turn_detection).toEqual(expected);
  });

  it('uses semantic defaults without server-only fields', () => {
    expect(buildSessionDefinition('voice', {
      provider: 'openai', transport: 'webrtc', turn_detection: { type: 'semantic_vad', eagerness: 'low' }
    }).turn_detection).toEqual({ type: 'semantic_vad', eagerness: 'low', create_response: true, interrupt_response: true });
  });

  it.each([
    {}, { type: 'client_vad' }, { type: 'server_vad', threshold: 2 },
    { type: 'server_vad', threshold: NaN }, { type: 'server_vad', threshold: true },
    { type: 'server_vad', silence_duration_ms: -1 }, { type: 'server_vad', prefix_padding_ms: 1.5 },
    { type: 'server_vad', create_response: 'false' }, { type: 'server_vad', interrupt_response: null },
    { type: 'semantic_vad', threshold: 0.5 }, { type: 'server_vad', eagerness: 'low' },
    { type: 'semantic_vad', eagerness: 'urgent' }, { type: 'server_vad', unknown: true }, []
  ])('rejects invalid runtime input %j', (config) => {
    expect(() => buildSessionDefinition('voice', {
      provider: 'openai', transport: 'webrtc', turn_detection: config as TurnDetection
    })).toThrow(/turn_detection/);
  });

  it('rejects VAD for openrouter without changing its defaults', () => {
    expect(() => buildSessionDefinition('voice', {
      provider: 'openrouter', transport: 'audio_turns', turn_detection: { type: 'server_vad' }
    })).toThrow(/turn_detection requires/);
    expect(buildSessionDefinition('voice', {
      provider: 'openrouter', transport: 'audio_turns'
    }).turn_detection).toBeUndefined();
  });
});
