import { describe, expect, it } from 'vitest';
import { SessionTransportError, validateSessionTransport } from '../src/sessionTransport.js';

describe('session transport validation', () => {
  it('accepts explicit supported pairs', () => {
    expect(validateSessionTransport('OpenAI', 'WebRTC')).toEqual({
      provider: 'openai',
      transport: 'webrtc'
    });
  });

  it('rejects provider and transport mismatches', () => {
    expect(() => validateSessionTransport('openrouter', 'webrtc')).toThrow(SessionTransportError);
    expect(() => validateSessionTransport('openrouter', 'webrtc')).toThrow(
      /Supported transports: audio_turns/
    );
    expect(() => validateSessionTransport('openrouter', 'webrtc')).toThrow(
      /does not infer or switch providers/
    );
  });

  it('requires explicit transport', () => {
    expect(() => validateSessionTransport('openai', '')).toThrow(/transport is required/);
  });

  it('rejects unknown providers', () => {
    expect(() => validateSessionTransport('custom', 'webrtc')).toThrow(
      /Unknown session provider 'custom'/
    );
  });
});
