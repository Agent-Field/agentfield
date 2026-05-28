import { describe, it, expect } from 'vitest';

import { httpAgent, httpsAgent } from '../src/utils/httpAgents.js';

describe('http agents', () => {
  it('configures the shared http agent with connection pooling limits', () => {
    expect(httpAgent.keepAlive).toBe(true);
    expect(httpAgent.maxSockets).toBe(10);
    expect(httpAgent.maxTotalSockets).toBe(50);
    expect(httpAgent.maxFreeSockets).toBe(5);
  });

  it('configures the shared https agent with connection pooling limits', () => {
    expect(httpsAgent.keepAlive).toBe(true);
    expect(httpsAgent.maxSockets).toBe(10);
    expect(httpsAgent.maxTotalSockets).toBe(50);
    expect(httpsAgent.maxFreeSockets).toBe(5);
  });
});
