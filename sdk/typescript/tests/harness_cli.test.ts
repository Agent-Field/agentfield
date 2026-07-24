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
      stdio: ['ignore', 'pipe', 'pipe']
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

  it('adds OpenRouter attribution env defaults and preserves caller overrides', async () => {
    const proc = createProcess();
    spawnMock.mockReturnValueOnce(proc as unknown as ReturnType<SpawnImpl>);

    const pending = runCli(['node', 'script.js'], {
      env: { OR_APP_NAME: 'Caller App' }
    });

    expect(spawnMock).toHaveBeenCalledWith('node', ['script.js'], {
      env: expect.objectContaining({
        AGENTFIELD_OPENROUTER_SITE_URL: 'https://agentfield.ai',
        AGENTFIELD_OPENROUTER_APP_NAME: 'Caller App',
        OR_SITE_URL: 'https://agentfield.ai',
        OR_APP_NAME: 'Caller App'
      }),
      cwd: undefined,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    proc.emit('close', 0);
    await expect(pending).resolves.toMatchObject({ exitCode: 0 });
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

  it('aborts a stalled child via the idle watchdog before the wall-clock cap', async () => {
    vi.useFakeTimers();
    const proc = createProcess();
    spawnMock.mockReturnValueOnce(proc as unknown as ReturnType<SpawnImpl>);

    // Generous wall-clock cap; the 1s idle window should fire first.
    const pending = runCli(['staller'], { timeout: 60000, idleSeconds: 1 });
    const expectation = expect(pending).rejects.toThrow('CLI made no progress for 1s');

    // One line of output, then the child stalls (no more data events).
    (proc.stdout as unknown as MockStream).pushChunk('started\n');

    // Advance past the idle window without any further output.
    await vi.advanceTimersByTimeAsync(1200);
    await expectation;
    expect(proc.kill).toHaveBeenCalledTimes(1);
  });

  it('resets the idle window on each chunk so an active child is not killed', async () => {
    vi.useFakeTimers();
    const proc = createProcess();
    spawnMock.mockReturnValueOnce(proc as unknown as ReturnType<SpawnImpl>);

    const pending = runCli(['active'], { idleSeconds: 1 });

    // Emit a chunk every 600ms: under the 1s idle window, so no kill.
    for (let i = 0; i < 4; i += 1) {
      (proc.stdout as unknown as MockStream).pushChunk(`chunk-${i} `);
      await vi.advanceTimersByTimeAsync(600);
    }
    proc.emit('close', 0);

    await expect(pending).resolves.toMatchObject({ exitCode: 0 });
    expect(proc.kill).not.toHaveBeenCalled();
  });

  it('reads the idle window from AGENTFIELD_HARNESS_IDLE_SECONDS', async () => {
    vi.useFakeTimers();
    const prev = process.env.AGENTFIELD_HARNESS_IDLE_SECONDS;
    process.env.AGENTFIELD_HARNESS_IDLE_SECONDS = '1';
    try {
      const proc = createProcess();
      spawnMock.mockReturnValueOnce(proc as unknown as ReturnType<SpawnImpl>);

      const pending = runCli(['staller'], { timeout: 60000 });
      const expectation = expect(pending).rejects.toThrow('CLI made no progress for 1s');
      await vi.advanceTimersByTimeAsync(1200);
      await expectation;
      expect(proc.kill).toHaveBeenCalledTimes(1);
    } finally {
      if (prev === undefined) {
        delete process.env.AGENTFIELD_HARNESS_IDLE_SECONDS;
      } else {
        process.env.AGENTFIELD_HARNESS_IDLE_SECONDS = prev;
      }
    }
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
