import { describe, it, expect, vi, afterEach } from 'vitest';

import { CodexProvider } from '../src/harness/providers/codex.js';
import { GeminiProvider } from '../src/harness/providers/gemini.js';
import * as cli from '../src/harness/cli.js';

afterEach(() => {
  vi.restoreAllMocks();
});

describe.each([
  ['codex', CodexProvider],
  ['gemini', GeminiProvider],
])('%s provider system prompt', (_name, Provider) => {
  it('prepends systemPrompt to the prompt', async () => {
    const run = vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: '', stderr: '', exitCode: 0 });

    await new Provider().execute('hello', { systemPrompt: 'Be terse.' });
    expect(run.mock.calls[0]?.[0].at(-1)).toBe('Be terse.\n\nTask:\nhello');

    await new Provider().execute('hello', {});
    expect(run.mock.calls[1]?.[0].at(-1)).toBe('hello');
  });
});
