import { describe, it, expect, vi, afterEach } from 'vitest';

import { parseJsonl, extractFinalText } from '../src/harness/cli.js';
import { CodexProvider } from '../src/harness/providers/codex.js';
import { buildProvider } from '../src/harness/providers/factory.js';
import * as cli from '../src/harness/cli.js';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('harness cli', () => {
  it('parseJsonl parses valid lines and skips invalid ones', () => {
    const events = parseJsonl('{"type":"thread.started"}\ninvalid\n{"type":"result","result":"ok"}\n');

    expect(events).toEqual([
      { type: 'thread.started' },
      { type: 'result', result: 'ok' },
    ]);
  });

  it('parseJsonl returns empty list for empty input', () => {
    expect(parseJsonl('')).toEqual([]);
    expect(parseJsonl('\n\n')).toEqual([]);
  });

  it('extractFinalText returns latest matching text', () => {
    const text = extractFinalText([
      { type: 'assistant', content: 'first' },
      { type: 'item.completed', item: { type: 'agent_message', text: 'codex text' } },
      { type: 'result', result: 'final text' },
    ]);

    expect(text).toBe('final text');
  });

  it('extractFinalText returns undefined for empty input', () => {
    expect(extractFinalText([])).toBeUndefined();
  });
});

describe('codex provider', () => {
  it('builds command with cwd/full-auto and maps results', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: '{"type":"thread.started","thread_id":"thread-1"}\n{"type":"turn.completed","text":"done"}\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new CodexProvider('/usr/local/bin/codex');
    const result = await provider.execute('hello', {
      cwd: '/tmp/work',
      permissionMode: 'auto',
      env: { A: '1' },
    });

    expect(cli.runCli).toHaveBeenCalledWith(
      ['/usr/local/bin/codex', 'exec', '--json', '-C', '/tmp/work', '--full-auto', 'hello'],
      { cwd: '/tmp/work', env: { A: '1' } }
    );
    expect(result.isError).toBe(false);
    expect(result.result).toBe('done');
    expect(result.metrics.numTurns).toBe(1);
    expect(result.metrics.sessionId).toBe('thread-1');
    expect(result.messages).toHaveLength(2);
  });

  it('returns helpful message when binary is not found', async () => {
    vi.spyOn(cli, 'runCli').mockRejectedValue(new Error('spawn codex ENOENT'));

    const provider = new CodexProvider('codex-missing');
    const result = await provider.execute('hello', {});

    expect(result.isError).toBe(true);
    expect(result.errorMessage).toContain("Codex binary not found at 'codex-missing'");
  });

  it('returns stderr when non-zero exit has no result', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: '{"type":"thread.started","thread_id":"thread-1"}\n',
      stderr: 'boom',
      exitCode: 2,
    });

    const provider = new CodexProvider('codex');
    const result = await provider.execute('hello', {});

    expect(result.isError).toBe(true);
    expect(result.result).toBeUndefined();
    expect(result.errorMessage).toBe('boom');
  });
});

describe('provider factory', () => {
  it('routes codex to CodexProvider and passes codexBin', async () => {
    const provider = await buildProvider({ provider: 'codex', codexBin: '/opt/codex' });

    expect(provider).toBeInstanceOf(CodexProvider);
  });
});
