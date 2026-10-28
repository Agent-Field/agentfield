import fs from 'node:fs';
import { createHash } from 'node:crypto';
import http from 'node:http';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

let directory: string;
let binDirectory: string;

function fakeBinary(binary: string, legacy = false, delay = 0): string {
  fs.mkdirSync(path.dirname(binary), { recursive: true });
  const result = legacy
    ? { text: 'legacy answer', stop: 'done', turns: 2, usage: { prompt_tokens: 5, completion_tokens: 3, cost: 0.02 } }
    : { ok: true, stop: 'done', answer: 'codeaf answer', files: [], error: '', spend_usd: 0,
        tokens: { in: 5, out: 3 }, seconds: 1, model: 'reported/model', steps: 2, calls: 3 };
  fs.writeFileSync(binary, `#!${process.execPath}
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => input += chunk);
process.stdin.on('end', () => {
  const result = ${JSON.stringify(result)};
  result.observed = {
    binary: ${JSON.stringify(binary)}, argv: process.argv.slice(2), input,
    home: process.env.CODEAF_HOME, update: process.env.CODEAF_NO_UPDATE_CHECK,
    telemetry: process.env.CODEAF_TELEMETRY,
    panels: [process.env.CODEAF_MODELS ?? null, process.env.AFORGE_MODELS ?? null],
    reasoning: process.env.CODEAF_EXEC_REASONING,
  };
  const output = input.match(/(\\S*\\.agentfield_output\\.json)/);
  if (output) require('fs').writeFileSync(output[1], JSON.stringify({ answer: 'schema answer' }));
  setTimeout(() => console.log(JSON.stringify(result)), ${delay});
});
`, { mode: 0o755 });
  return binary;
}

function fakeOutcome(stdout: string, exitCode: number): string {
  const binary = path.join(directory, 'outcome');
  fs.writeFileSync(binary, `#!${process.execPath}
process.stdin.resume();
process.stdin.on('end', () => {
  process.stdout.write(${JSON.stringify(stdout)});
  process.stderr.write('fixture diagnostic');
  ${exitCode < 0 ? "process.kill(process.pid, 'SIGTERM');" : `process.exitCode = ${exitCode};`}
});
`, { mode: 0o755 });
  return binary;
}

async function managedBinary(location = 0): Promise<string> {
  const { codeafManagedPaths } = await import('../src/harness/codeafInstall.js');
  const binary = fakeBinary(codeafManagedPaths()[location]);
  fs.writeFileSync(path.join(path.dirname(binary), '.codeaf.version'), 'v0.8.2\n');
  return binary;
}

beforeEach(() => {
  vi.resetModules();
  directory = fs.mkdtempSync(path.join(os.tmpdir(), 'agentfield-codeaf-resolution-'));
  binDirectory = path.join(directory, 'path');
  fs.mkdirSync(binDirectory);
  for (const name of [
    'AGENTFIELD_HARNESS_PROVIDER', 'CODEAF_BIN', 'AFORGE_BIN', 'AGENTFIELD_CODEAF_COMMAND',
    'AGENTFIELD_AFORGE_COMMAND', 'AGENTFIELD_HARNESS_TIMEOUT_SECONDS', 'CODEAF_HOME',
    'CODEAF_NO_UPDATE_CHECK', 'CODEAF_TELEMETRY', 'CODEAF_EXEC_REASONING', 'CODEAF_MODELS',
    'AFORGE_MODELS', 'AFORGE_MODEL', 'CODEAF_MODEL', 'CI', 'DO_NOT_TRACK', 'AGENTFIELD_TELEMETRY_ENABLED', 'HTTPS_PROXY', 'https_proxy',
  ]) vi.stubEnv(name, undefined);
  vi.stubEnv('AGENTFIELD_HOME', path.join(directory, 'home'));
  vi.stubEnv('HOME', path.join(directory, 'user-home'));
  vi.stubEnv('OPENROUTER_API_KEY', 'test-codeaf-key');
  vi.stubEnv('PATH', binDirectory);
  vi.stubEnv('AGENTFIELD_SKIP_CODEAF', 'true');
  vi.stubEnv('AGENTFIELD_SKIP_AFORGE', '');
  vi.spyOn(os, 'tmpdir').mockReturnValue(path.join(directory, 'temp'));
  vi.spyOn(console, 'info').mockImplementation(() => undefined);
  vi.spyOn(console, 'warn').mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  vi.doUnmock('../src/harness/codeafRelease.js');
  vi.unstubAllEnvs();
  fs.rmSync(directory, { recursive: true, force: true });
});

// Executable Node scripts exercise the real subprocess transport on Unix.
describe.skipIf(process.platform === 'win32')('V1–V6 CodeAF dispatch with fake CLIs', () => {
  it('C1 preserves the home install error and hint through retry throttling', async () => {
    vi.stubEnv('AGENTFIELD_SKIP_CODEAF', '');
    fs.writeFileSync(process.env.AGENTFIELD_HOME!, 'not a directory');
    const { CodeafProvider } = await import('../src/harness/providers/codeaf.js');
    for (let attempt = 0; attempt < 2; attempt += 1) {
      await expect(new CodeafProvider().execute('task', {})).rejects.toThrow('cannot install CodeAF into');
      await expect(new CodeafProvider().execute('task', {})).rejects.toThrow('af codeaf ensure');
    }
    expect(fs.existsSync(path.join(directory, 'temp'))).toBe(false);
  });

  it('C2 executes CODEAF_BIN with the default state path even when mkdir fails', async () => {
    vi.stubEnv('CODEAF_BIN', fakeBinary(path.join(directory, 'explicit-codeaf')));
    fs.writeFileSync(process.env.AGENTFIELD_HOME!, 'not a directory');
    const { CodeafProvider } = await import('../src/harness/providers/codeaf.js');
    const result = await new CodeafProvider().execute('task', {});
    expect(result.isError).toBe(false);
    expect(result.messages[0].observed).toHaveProperty('home', path.join(process.env.AGENTFIELD_HOME!, 'codeaf'));
    expect(fs.existsSync(path.join(directory, 'temp'))).toBe(false);
  });

  it.each(['HTTPS_PROXY', 'https_proxy'])('C3 skips downloads with %s and falls through to PATH, legacy or unavailable', async (key) => {
    vi.stubEnv('AGENTFIELD_SKIP_CODEAF', '');
    vi.stubEnv(key, 'http://proxy.invalid');
    const get = vi.spyOn(http, 'get').mockImplementation(() => { throw new Error('unexpected network request'); });
    const secureGet = vi.spyOn(https, 'get').mockImplementation(() => { throw new Error('unexpected network request'); });
    const { CodeafProvider } = await import('../src/harness/providers/codeaf.js');
    const provider = new CodeafProvider();
    await expect(provider.execute('task', {})).rejects.toThrow('CodeAF download skipped: HTTPS_PROXY is set and the TypeScript SDK does not use proxies; run `af codeaf ensure`');
    const legacy = fakeBinary(path.join(binDirectory, 'aforge'), true);
    expect(await provider.execute('task', {})).toMatchObject({ provider: 'aforge' });
    expect(get).not.toHaveBeenCalled();
    expect(secureGet).not.toHaveBeenCalled();
    fs.unlinkSync(legacy);
    vi.resetModules();
    fakeBinary(path.join(binDirectory, 'codeaf'));
    const { CodeafProvider: FreshProvider } = await import('../src/harness/providers/codeaf.js');
    expect(await new FreshProvider().execute('task', {})).toMatchObject({ provider: 'codeaf' });
    expect(get).not.toHaveBeenCalled();
    expect(secureGet).not.toHaveBeenCalled();
  });

  it('V1 uses CodeAF with no provider configuration and returns its canonical metrics', async () => {
    await managedBinary();
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const result = await new HarnessRunner().run('task', { maxRetries: 0 });
    expect(result).toMatchObject({ provider: 'codeaf', result: 'codeaf answer', costUsd: 0,
      inputTokens: 5, outputTokens: 3, numTurns: 2, model: 'reported/model', failureType: 'none' });
    expect(result.messages[0].observed).toMatchObject({ input: 'task', argv: [
      'exec', '--json', '--dir', '.', '--model', '~deepseek/deepseek-v4-flash-latest', '--timeout', expect.any(String),
    ] });
  });

  it('V3 resolves an unpinned aforge alias through CodeAF and logs the note once', async () => {
    await managedBinary();
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const runner = new HarnessRunner({ provider: 'aforge' });
    const results = await Promise.all([runner.run('first'), runner.run('second')]);
    expect(results.map((result) => result.provider)).toEqual(['codeaf', 'codeaf']);
    expect(console.info).toHaveBeenCalledExactlyOnceWith("provider 'aforge' now runs CodeAF; set AFORGE_BIN to pin the legacy binary");
  });

  it.each(['config', 'env'])('V4 honors a legacy %s pin and the unchanged protocol', async (pin) => {
    const legacy = fakeBinary(path.join(directory, 'legacy'), true);
    vi.stubEnv('CODEAF_BIN', fakeBinary(path.join(directory, 'other-codeaf')));
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const runner = new HarnessRunner(pin === 'config'
      ? { provider: 'aforge', aforgeBin: legacy } : { provider: 'aforge' });
    if (pin === 'env') vi.stubEnv('AFORGE_BIN', legacy);
    const result = await runner.run('task', { maxRetries: 0 });
    expect(result).toMatchObject({ provider: 'aforge', result: 'legacy answer', costUsd: 0.02, inputTokens: 5 });
    const observed = result.messages[0].observed as { argv: string[] };
    expect(observed.argv).toContain('-w');
    expect(observed.argv).toContain('1795');
    expect(console.info).not.toHaveBeenCalled();
  });

  it('V4 reads AFORGE_BIN at first execution, after construction', async () => {
    const { buildProvider } = await import('../src/harness/providers/factory.js');
    const provider = await buildProvider({ provider: 'aforge' });
    vi.stubEnv('AFORGE_BIN', fakeBinary(path.join(directory, 'legacy'), true));
    expect(await provider.execute('task', {})).toMatchObject({ provider: 'aforge', result: 'legacy answer' });
  });

  it('V5 prefers codeafBin over CODEAF_BIN, then CODEAF_BIN over managed/PATH', async () => {
    await managedBinary();
    fakeBinary(path.join(binDirectory, 'codeaf'));
    const configured = fakeBinary(path.join(directory, 'configured'));
    const fromEnv = fakeBinary(path.join(directory, 'from-env'));
    vi.stubEnv('CODEAF_BIN', fromEnv);
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const runner = new HarnessRunner({ codeafBin: configured });
    const first = await runner.run('task');
    const second = await new HarnessRunner().run('task');
    expect(first.messages[0].observed).toHaveProperty('binary', configured);
    expect(second.messages[0].observed).toHaveProperty('binary', fromEnv);
  });

  it('V5 reads CODEAF_BIN lazily and codeaf ignores an unrelated AFORGE_BIN pin', async () => {
    const { buildProvider } = await import('../src/harness/providers/factory.js');
    const provider = await buildProvider({ provider: 'codeaf' });
    vi.stubEnv('AFORGE_BIN', fakeBinary(path.join(directory, 'legacy'), true));
    vi.stubEnv('CODEAF_BIN', fakeBinary(path.join(directory, 'from-env')));
    expect(await provider.execute('task', {})).toMatchObject({ provider: 'codeaf', result: 'codeaf answer' });
  });

  it.each([0])('V5 finds managed location %s ahead of PATH without installing', async (location) => {
    const managed = await managedBinary(location);
    fakeBinary(path.join(binDirectory, 'codeaf'));
    const install = await import('../src/harness/codeafInstall.js');
    const ensure = vi.spyOn(install, 'ensureCodeafInstalled');
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const result = await new HarnessRunner().run('task');
    expect(result.messages[0].observed).toHaveProperty('binary', managed);
    expect(ensure).not.toHaveBeenCalled();
  });

  it('V5–V6 installs before considering PATH, shares resolution across concurrent executes, and caches it', async () => {
    fakeBinary(path.join(binDirectory, 'codeaf'));
    const installed = fakeBinary(path.join(directory, 'downloaded'));
    const install = await import('../src/harness/codeafInstall.js');
    const ensure = vi.spyOn(install, 'ensureCodeafInstalled').mockImplementation(async () => {
      await new Promise((resolve) => setTimeout(resolve, 5));
      return installed;
    });
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const runner = new HarnessRunner();
    const results = await Promise.all(Array.from({ length: 12 }, () => runner.run('task')));
    expect(results.every((result) => (result.messages[0].observed as { binary: string }).binary === installed)).toBe(true);
    await runner.run('after');
    expect(ensure).toHaveBeenCalledTimes(1);
  });

  it('V5 falls back to PATH after a failed download', async () => {
    const onPath = fakeBinary(path.join(binDirectory, 'codeaf'));
    const install = await import('../src/harness/codeafInstall.js');
    vi.spyOn(install, 'ensureCodeafInstalled').mockRejectedValue(new Error('fixture network error'));
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const result = await new HarnessRunner().run('task');
    expect(result).toHaveProperty('provider', 'codeaf');
    expect(result.messages[0].observed).toHaveProperty('binary', onPath);
    expect(console.warn).not.toHaveBeenCalled();
  });

  it.each(['AGENTFIELD_SKIP_CODEAF', 'AGENTFIELD_SKIP_AFORGE'])('V5–V6 falls back to PATH when %s skips downloading', async (key) => {
    vi.stubEnv('AGENTFIELD_SKIP_CODEAF', '');
    vi.stubEnv(key, 'YeS');
    fakeBinary(path.join(binDirectory, 'codeaf'));
    const { HarnessRunner } = await import('../src/harness/runner.js');
    expect(await new HarnessRunner().run('task')).toHaveProperty('provider', 'codeaf');
    expect(console.info).not.toHaveBeenCalled();
  });

  it.each(['home', 'path'])('V5 falls back to legacy aforge on %s and warns once', async (location) => {
    const legacy = fakeBinary(location === 'home'
      ? path.join(process.env.AGENTFIELD_HOME!, 'bin', 'aforge') : path.join(binDirectory, 'aforge'), true);
    if (location === 'home') fakeBinary(path.join(binDirectory, 'aforge'), true);
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const runner = new HarnessRunner();
    const first = await runner.run('task');
    await runner.run('task');
    expect(first).toMatchObject({ provider: 'aforge', result: 'legacy answer' });
    expect(first.messages[0].observed).toHaveProperty('binary', legacy);
    expect(console.warn).toHaveBeenCalledExactlyOnceWith(
      `CodeAF is not installed; falling back to the legacy aforge binary at ${legacy}. Run \`af codeaf ensure\`.`
    );
  });

  it('V5 reports a typed codeaf-unavailable error with installer hint and download failure', async () => {
    const install = await import('../src/harness/codeafInstall.js');
    const ensure = vi.spyOn(install, 'ensureCodeafInstalled').mockRejectedValue(new Error('fixture checksum failed'));
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const promise = new HarnessRunner().run('task', { maxRetries: 0 });
    await expect(promise).rejects.toMatchObject({ provider: 'codeaf', installCommand: 'af codeaf ensure' });
    await expect(promise).rejects.toThrow('CODEAF_BIN');
    await expect(promise).rejects.toThrow('fixture checksum failed');
    await expect(new HarnessRunner().run('task', { maxRetries: 0 })).rejects.toThrow('fixture checksum failed');
    expect(ensure).toHaveBeenCalledTimes(2);
  });

  it('V5 skips the download and reports unavailable when no fallback exists', async () => {
    const { HarnessRunner } = await import('../src/harness/runner.js');
    await expect(new HarnessRunner().run('task', { maxRetries: 0 })).rejects.toThrow('af codeaf ensure');
    expect(console.info).not.toHaveBeenCalled();
  });

  it.each(['home', 'path', 'legacy', 'override'])('A1 recovers from failed resolution via %s without resetting caches', async (repair) => {
    const { CodeafProvider } = await import('../src/harness/providers/codeaf.js');
    const provider = new CodeafProvider();
    await expect(provider.execute('task', {})).rejects.toThrow('af codeaf ensure');
    let binary: string;
    if (repair === 'home') binary = await managedBinary();
    else if (repair === 'path') binary = fakeBinary(path.join(binDirectory, 'codeaf'));
    else if (repair === 'legacy') binary = fakeBinary(path.join(binDirectory, 'aforge'), true);
    else {
      binary = fakeBinary(path.join(directory, 'override'));
      vi.stubEnv('CODEAF_BIN', binary);
    }
    for (const instance of [provider, new CodeafProvider()]) {
      const result = await instance.execute('task', {});
      expect(result.isError).toBe(false);
      expect(result.messages[0].observed).toHaveProperty('binary', binary);
    }
  });

  it('A1 retries a failed real download after 60 seconds and shares the retry', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.stubEnv('AGENTFIELD_SKIP_CODEAF', '');
    const source = fakeBinary(path.join(directory, 'source'));
    const bytes = fs.readFileSync(source);
    const digest = createHash('sha256').update(bytes).digest('hex');
    vi.doMock('../src/harness/codeafRelease.js', async (importOriginal) => {
      const actual = await importOriginal<typeof import('../src/harness/codeafRelease.js')>();
      return { ...actual, CODEAF_SHA256: Object.fromEntries(Object.keys(actual.CODEAF_SHA256).map((asset) => [asset, digest])) };
    });
    let requests = 0;
    let healthy = false;
    const server = http.createServer((_request, response) => {
      requests += 1;
      if (healthy) response.end(bytes);
      else response.writeHead(503).end();
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', `http://127.0.0.1:${(server.address() as { port: number }).port}`);
    for (const name of ['HTTP_PROXY', 'http_proxy', 'NO_PROXY', 'no_proxy']) vi.stubEnv(name, undefined);
    try {
      const { CodeafProvider } = await import('../src/harness/providers/codeaf.js');
      const provider = new CodeafProvider();
      await expect(provider.execute('task', {})).rejects.toThrow('HTTP 503');
      healthy = true;
      await expect(new CodeafProvider().execute('task', {})).rejects.toThrow('af codeaf ensure');
      expect(requests).toBe(1);
      vi.setSystemTime(Date.now() + 60_000);
      const results = await Promise.all(Array.from({ length: 8 }, () => provider.execute('task', {})));
      expect(results.every((result) => result.result === 'codeaf answer')).toBe(true);
      expect(requests).toBe(2);
    } finally {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  it.each(['process', 'overlay', 'empty'])('A2 preserves explicit %s CODEAF_HOME as-is', async (source) => {
    vi.stubEnv('CODEAF_BIN', fakeBinary(path.join(directory, 'explicit-codeaf')));
    fs.writeFileSync(process.env.AGENTFIELD_HOME!, 'home is a file');
    const explicit = source === 'empty' ? '' : path.join(directory, 'explicit');
    if (source === 'process') vi.stubEnv('CODEAF_HOME', explicit);
    const { CodeafProvider } = await import('../src/harness/providers/codeaf.js');
    const result = await new CodeafProvider().execute('task', source === 'process' ? {} : { env: { CODEAF_HOME: explicit } });
    expect(result.isError).toBe(false);
    expect(result.messages[0].observed).toHaveProperty('home', explicit);
    expect(fs.existsSync(path.join(directory, 'temp'))).toBe(false);
  });

  it('V8 removes inherited model panels in the real child and honors overlays', async () => {
    await managedBinary();
    vi.stubEnv('CODEAF_MODELS', 'wrong');
    vi.stubEnv('AFORGE_MODELS', 'wrong');
    vi.stubEnv('CI', 'true');
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const runner = new HarnessRunner();
    const defaults = await runner.run('task');
    expect(defaults.messages[0].observed).toMatchObject({
      panels: [null, null], update: '1', telemetry: 'off', home: path.join(process.env.AGENTFIELD_HOME!, 'codeaf'),
    });
    const overlay = await runner.run('task', { env: {
      CODEAF_HOME: directory, CODEAF_NO_UPDATE_CHECK: '0', CODEAF_TELEMETRY: 'on',
      CODEAF_MODELS: 'new', AFORGE_MODELS: 'old', CODEAF_EXEC_REASONING: 'off',
    } });
    expect(overlay.messages[0].observed).toMatchObject({
      panels: ['new', 'old'], home: directory, update: '0', telemetry: 'on', reasoning: 'off',
    });
  });

  it('preserves the existing schema pipeline and actual provider in its result', async () => {
    await managedBinary();
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const result = await new HarnessRunner({ provider: 'aforge' }).run('task', {
      cwd: directory, schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
    });
    expect(result).toMatchObject({ provider: 'codeaf', parsed: { answer: 'schema answer' }, isError: false, costUsd: 0 });
    expect(fs.readdirSync(directory).filter((name) => name.startsWith('.agentfield-out-'))).toEqual([]);
  });

  it.each([
    ['done', true, 'answer', 0, 'none'],
    ['deadline', false, 'partial', 3, 'timeout'],
    ['deadline', false, 'partial', 124, 'timeout'],
    ['unchecked', false, 'answer', 2, 'none'],
    ['budget', false, 'partial', 3, 'none'],
    ['turn-cap', false, 'partial', 3, 'none'],
    ['budget', false, '', 3, 'no_output'],
    ['turn-cap', false, '', 3, 'no_output'],
    ['question', false, 'partial', 4, 'crash'],
    ['incomplete', false, 'partial', 2, 'crash'],
    ['error', false, 'partial', 1, 'crash'],
    ['price', false, 'partial', 3, 'crash'],
    ['unknown', true, 'answer', 0, 'none'],
    ['unknown', true, '', 0, 'crash'],
    ['unknown', true, 'answer', 1, 'crash'],
    ['done', true, 'answer', -15, 'crash'],
  ])('V10 real CLI: stop=%s ok=%s answer=%s exit=%s gives %s', async (stop, ok, answer, exitCode, failureType) => {
    const binary = fakeOutcome(JSON.stringify({ ok, stop, answer, blocked_on: 'Which project?' }), exitCode);
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const result = await new HarnessRunner({ codeafBin: binary }).run('task', { maxRetries: 0 });
    expect(result).toMatchObject({ provider: 'codeaf', failureType, isError: failureType !== 'none' });
    expect(result.result).toBe(failureType === 'none' ? answer : undefined);
    if (stop === 'question') expect(result.errorMessage).toContain('Which project?');
  });

  it.each([
    ['', 0, 'no_output'], ['', 1, 'crash'], ['not JSON', 0, 'crash'],
    ['{"text":"legacy","usage":{"cost":1}}', 0, 'crash'],
  ])('V9–V10 real CLI: missing envelope %s exit=%s gives %s', async (stdout, exitCode, failureType) => {
    const binary = fakeOutcome(stdout, exitCode);
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const result = await new HarnessRunner({ codeafBin: binary }).run('task', { maxRetries: 0 });
    expect(result).toMatchObject({ provider: 'codeaf', failureType, isError: true });
  });

  it('V10 classifies a real outer timeout even when the process has produced no output', async () => {
    vi.stubEnv('AGENTFIELD_HARNESS_TIMEOUT_SECONDS', '1');
    const binary = fakeBinary(path.join(directory, 'slow'), false, 3000);
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const result = await new HarnessRunner({ codeafBin: binary }).run('task', { maxRetries: 0 });
    expect(result).toMatchObject({ provider: 'codeaf', failureType: 'timeout', isError: true });
    expect(result.result).toBeUndefined();
  });
});

describe('V12 CodeAF semaphore', () => {
  it('limits all provider instances in the process to the configured concurrency', async () => {
    vi.stubEnv('CODEAF_MAX_CONCURRENT', '2');
    const cli = await import('../src/harness/cli.js');
    let active = 0;
    let maximum = 0;
    vi.spyOn(cli, 'runCli').mockImplementation(async () => {
      active += 1;
      maximum = Math.max(maximum, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active -= 1;
      return { stdout: '{"ok":true,"stop":"done","answer":"ok"}', stderr: '', exitCode: 0 };
    });
    const { CodeafProvider } = await import('../src/harness/providers/codeaf.js');
    const results = await Promise.all(Array.from({ length: 11 }, () => new CodeafProvider('/fixture/codeaf').execute('task', {})));
    expect(results.every((result) => result.result === 'ok')).toBe(true);
    expect(maximum).toBe(2);
  });
});

describe.skipIf(process.platform === 'win32')('F1/F3/F5 compatibility contracts', () => {
  it.each(['env', 'config', 'call'])('F1a-c retains an implicit %s pin and honors explicit CodeAF', async (pin) => {
    const legacy = fakeBinary(path.join(directory, 'legacy'), true);
    const canonical = await managedBinary();
    if (pin === 'env') vi.stubEnv('AFORGE_BIN', legacy);
    const { HarnessRunner } = await import('../src/harness/runner.js');
    const { buildProvider } = await import('../src/harness/providers/factory.js');
    const { AforgeProvider } = await import('../src/harness/providers/aforge.js');
    const config = pin === 'config' ? { aforgeBin: legacy } : {};
    const options = pin === 'call' ? { aforgeBin: legacy } : {};
    expect((await new HarnessRunner(config).run('p', options)).provider).toBe('aforge');
    if (pin !== 'call') expect(await buildProvider(config)).toBeInstanceOf(AforgeProvider);
    const result = await new HarnessRunner({ ...config, provider: 'codeaf' }).run('p', options);
    expect(result.provider).toBe('codeaf');
    expect(result.messages[0].observed).toHaveProperty('binary', canonical);
    vi.stubEnv('AGENTFIELD_HARNESS_PROVIDER', 'codeaf');
    expect((await new HarnessRunner(config).run('p', options)).provider).toBe('codeaf');
  });

  it('F3a/b bounds all waiters without canceling the shared download', async () => {
    vi.stubEnv('AGENTFIELD_SKIP_CODEAF', '');
    vi.stubEnv('AGENTFIELD_HARNESS_TIMEOUT_SECONDS', '1');
    const source = fakeBinary(path.join(directory, 'source'));
    const bytes = fs.readFileSync(source);
    const digest = createHash('sha256').update(bytes).digest('hex');
    vi.doMock('../src/harness/codeafRelease.js', async (importOriginal) => {
      const actual = await importOriginal<typeof import('../src/harness/codeafRelease.js')>();
      return { ...actual, CODEAF_SHA256: Object.fromEntries(Object.keys(actual.CODEAF_SHA256).map((asset) => [asset, digest])) };
    });
    let release!: () => void;
    const blocked = new Promise<void>((resolve) => { release = resolve; });
    let requests = 0;
    const server = http.createServer(async (_request, response) => {
      requests += 1;
      await blocked;
      response.end(bytes);
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    vi.stubEnv('AGENTFIELD_CODEAF_BASE_URL', `http://127.0.0.1:${(server.address() as { port: number }).port}`);
    try {
      const { CodeafProvider } = await import('../src/harness/providers/codeaf.js');
      const started = performance.now();
      const results = await Promise.all([new CodeafProvider().execute('p', {}), new CodeafProvider().execute('p', {})]);
      expect(performance.now() - started).toBeLessThan(1500);
      expect(results.every((r) => r.failureType === 'timeout' && r.errorMessage?.includes('provisioning CodeAF'))).toBe(true);
      expect(requests).toBe(1);
      // Explicit binaries do not wait on the still-running installation.
      expect((await new CodeafProvider(source).execute('p', {})).isError).toBe(false);
      release();
      expect((await new CodeafProvider().execute('p', {})).result).toBe('codeaf answer');
      expect(requests).toBe(1);
    } finally {
      release();
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  it('F3c passes the remaining budget to the subprocess', async () => {
    const installer = await import('../src/harness/codeafInstall.js');
    const source = fakeBinary(path.join(directory, 'source'));
    vi.spyOn(installer, 'ensureCodeafInstalled').mockImplementation(async () => {
      await new Promise((resolve) => setTimeout(resolve, 150));
      return source;
    });
    vi.stubEnv('AGENTFIELD_HARNESS_TIMEOUT_SECONDS', '10');
    const transport = await import('../src/harness/cli.js');
    const run = vi.spyOn(transport, 'runCli').mockResolvedValue({ stdout: '{"ok":true,"stop":"done"}', stderr: '', exitCode: 0 });
    const { CodeafProvider } = await import('../src/harness/providers/codeaf.js');
    await new CodeafProvider().execute('p', {});
    const [cmd, options] = run.mock.calls[0];
    // Provisioning took >= 150ms of the 10s budget; only the rest is passed on.
    expect(options!.timeout).toBeGreaterThan(5000);
    expect(options!.timeout).toBeLessThanOrEqual(9855);
    // The inner timeout is taken first, so it can only exceed remaining - 5.
    const inner = Number.parseFloat(cmd[cmd.indexOf('--timeout') + 1]);
    expect(inner + 0.001).toBeGreaterThanOrEqual(options!.timeout! / 1000 - 5);
    expect(inner).toBeLessThanOrEqual(4.856);
  });

  it.each(['managed', 'path', 'legacy', 'missing', 'pin'])('F5a-d doctor %s alias is read-only', async (location) => {
    const target = location === 'managed' ? await managedBinary()
      : location === 'missing' ? '' : fakeBinary(path.join(binDirectory, location === 'legacy' ? 'aforge' : 'codeaf'), location === 'legacy');
    if (location === 'pin') vi.stubEnv('AFORGE_BIN', path.join(directory, 'missing'));
    const { harnessDoctor } = await import('../src/harness/availability.js');
    const versionProbe = vi.fn().mockResolvedValue(location === 'legacy' ? 'aforge v0.1.0' : 'codeaf v0.8.2');
    const [report] = await harnessDoctor(['aforge'], { versionProbe });
    expect(report.usable).toBe(!['missing', 'pin'].includes(location));
    if (['managed', 'path'].includes(location)) {
      expect(report.version).toBe('codeaf v0.8.2');
      expect(versionProbe).toHaveBeenCalledWith([target, 'version']);
    }
    if (!report.usable) expect(report.installCommand).toBe('af aforge ensure');
  });

  it.each(['R4a', 'R4b', 'R4c'])('%s doctor honors an authoritative CodeAF override', async (contract) => {
    const legacy = fakeBinary(path.join(binDirectory, 'aforge'), true);
    const target = path.join(directory, 'override');
    if (contract !== 'R4b') vi.stubEnv('CODEAF_BIN', target);
    if (contract === 'R4c') fakeBinary(target);
    const { harnessDoctor } = await import('../src/harness/availability.js');
    const version = contract === 'R4b' ? 'aforge v0.1.0' : 'codeaf v0.8.2';
    const versionProbe = vi.fn().mockResolvedValue(version);
    const [report] = await harnessDoctor(['aforge'], { versionProbe });
    expect(report.usable).toBe(contract !== 'R4a');
    expect(report.installed).toBe(report.usable);
    if (contract === 'R4a') {
      expect(report.binary).toBeNull();
      expect(report.issues).toEqual(['binary_not_found']);
      expect(versionProbe).not.toHaveBeenCalled();
    } else {
      expect(report.version).toBe(version);
      expect(versionProbe).toHaveBeenCalledExactlyOnceWith([contract === 'R4b' ? legacy : target, 'version']);
    }
  });
});
