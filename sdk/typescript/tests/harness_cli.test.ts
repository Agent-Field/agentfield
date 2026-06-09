import { EventEmitter } from 'node:events';
import type { ChildProcessWithoutNullStreams } from 'node:child_process';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { extractFinalText, parseJsonl, runCli } from '../src/harness/cli.js';

type SpawnImpl = typeof import('node:child_process').spawn;

const { spawnMock } = vi.hoisted(() => ({
  spawnMock: vi.fn<SpawnImpl>()
}));

vi.mock('node:child_process', () => ({
  spawn: spawnMock
}));

class MockStream extends EventEmitter {
  pushChunk(chunk: string) {
    this.emit('data', chunk);
  }
}

type MockChild = EventEmitter &
  Pick<ChildProcessWithoutNullStreams, 'stdout' | 'stderr' | 'kill'>;

const createProcess = (): MockChild => {
  const proc = new EventEmitter() as MockChild;
  proc.stdout = new MockStream() as ChildProcessWithoutNullStreams['stdout'];
  proc.stderr = new MockStream() as ChildProcessWithoutNullStreams['stderr'];
  proc.kill = vi.fn();
  return proc;
};

describe('harness cli utilities', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('runs a CLI command and captures stdout, stderr, exit code, cwd, and env', async () => {
    const proc = createProcess();
    spawnMock.mockReturnValueOnce(proc as unknown as ReturnType<SpawnImpl>);

    const pending = runCli(['node', 'script.js'], {
      cwd: '/tmp/work',
      env: { CUSTOM_ENV: 'yes' }
    });

    expect(spawnMock).toHaveBeenCalledWith('node', ['script.js'], {
      env: expect.objectContaining({ CUSTOM_ENV: 'yes' }),
      cwd: '/tmp/work',
      stdio: ['pipe', 'pipe', 'pipe']
    });

    (proc.stdout as unknown as MockStream).pushChunk('hello ');
    (proc.stdout as unknown as MockStream).pushChunk('world');
    (proc.stderr as unknown as MockStream).pushChunk('warn');
    proc.emit('close', 3);

    await expect(pending).resolves.toEqual({
      stdout: 'hello world',
      stderr: 'warn',
      exitCode: 3
    });
  });

  it('rejects on child process errors and on timeouts', async () => {
    const errorProc = createProcess();
    spawnMock.mockReturnValueOnce(errorProc as unknown as ReturnType<SpawnImpl>);
    const errorPending = runCli(['bad-bin']);
    const failure = new Error('spawn failed');
    errorProc.emit('error', failure);
    await expect(errorPending).rejects.toBe(failure);

    vi.useFakeTimers();
    const timeoutProc = createProcess();
    spawnMock.mockReturnValueOnce(timeoutProc as unknown as ReturnType<SpawnImpl>);
    const timeoutPending = runCli(['slow-bin'], { timeout: 25 });
    const timeoutExpectation = expect(timeoutPending).rejects.toThrow('CLI timed out after 25ms');
    await vi.advanceTimersByTimeAsync(25);
    await timeoutExpectation;
    expect(timeoutProc.kill).toHaveBeenCalledTimes(1);
  });

  it('parses JSONL and extracts the last final text from supported event types', () => {
    expect(
      parseJsonl('{"type":"result","text":"one"}\nnot-json\n{"type":"assistant","content":"two"}\n \n')
    ).toEqual([
      { type: 'result', text: 'one' },
      { type: 'assistant', content: 'two' }
    ]);

    expect(
      extractFinalText([
        {
          type: 'item.completed',
          item: {
            type: 'agent_message',
            text: 'from-item'
          }
        },
        {
          type: 'result',
          result: 'from-result'
        },
        {
          type: 'turn.completed',
          text: 'from-turn'
        },
        {
          type: 'message',
          content: 'from-message'
        },
        {
          type: 'assistant',
          content: 'from-assistant'
        }
      ])
    ).toBe('from-assistant');

    expect(extractFinalText([{ type: 'item.completed', item: { type: 'other', text: 'ignored' } }])).toBeUndefined();
  });
});
