import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import { HarnessProviderNotConfigured, HarnessProviderUnavailable } from '../src/index.js';
import { buildProvider } from '../src/harness/providers/factory.js';
import { CodeafProvider, resetCodeafAuthWarnings } from '../src/harness/providers/codeaf.js';
import * as install from '../src/harness/codeafInstall.js';

interface AuthCase {
  name: string;
  allow: boolean;
  env: Record<string, string>;
  overlay: Record<string, string>;
  files: Record<string, string | null>;
}
const contract: { refusal: string; warning: string; cases: AuthCase[] } = JSON.parse(
  fs.readFileSync(new URL('../../codeaf-auth-contract.json', import.meta.url), 'utf8')
);
let directory: string;

beforeEach(() => {
  directory = fs.mkdtempSync(path.join(os.tmpdir(), 'codeaf-auth-'));
  for (const key of Object.keys(process.env)) {
    if (/^(CODEAF_|AFORGE_|AGENTFIELD_)/.test(key) || ['OPENROUTER_API_KEY', 'OPENAI_API_KEY'].includes(key)) {
      vi.stubEnv(key, undefined);
    }
  }
  vi.stubEnv('HOME', path.join(directory, 'home'));
  vi.stubEnv('PATH', directory);
  vi.stubEnv('AGENTFIELD_HOME', path.join(directory, 'agentfield'));
  vi.stubEnv('CODEAF_TELEMETRY', 'off');
  vi.spyOn(install, 'ensureCodeafInstalled').mockImplementation(() => { throw new Error('installer called'); });
  vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  vi.spyOn(console, 'info').mockImplementation(() => undefined);
  resetCodeafAuthWarnings();
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  fs.rmSync(directory, { recursive: true, force: true });
});

function fakeBinary(): { binary: string; marker: string } {
  const binary = path.join(directory, 'fake-codeaf');
  const marker = path.join(directory, 'ran');
  fs.writeFileSync(binary, `#!${process.execPath}\nprocess.stdin.resume();\nprocess.stdin.on('end', () => {\nrequire('node:fs').writeFileSync(${JSON.stringify(marker)}, 'ran');\nconsole.log('{"ok":true,"stop":"done","answer":"answer"}');\n});\n`);
  fs.chmodSync(binary, 0o755);
  return { binary, marker };
}

function expectRefusal(error: unknown): void {
  expect(error).toBeInstanceOf(HarnessProviderNotConfigured);
  expect(error).toBeInstanceOf(HarnessProviderUnavailable);
  expect(error).toMatchObject({
    provider: 'codeaf', message: contract.refusal, missingAuthEnv: ['OPENROUTER_API_KEY'],
    installCommand: 'af codeaf ensure',
  });
}

describe('K1 K3 K4 K7 shared key ladder', () => {
  it.each(contract.cases)('$name', async (testCase) => {
    const paths: Record<string, string> = {
      $STATE: path.join(directory, 'agentfield', 'codeaf'),
      $PROFILE: path.join(directory, 'profile'), $HOME: path.join(directory, 'home'),
    };
    const expand = (value: string): string => Object.entries(paths).reduce(
      (result, [token, replacement]) => result.replaceAll(token, replacement), value
    );
    for (const [key, value] of Object.entries(testCase.env)) vi.stubEnv(key, expand(value));
    for (const [filename, contents] of Object.entries(testCase.files)) {
      const target = expand(filename);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      if (contents === null) fs.mkdirSync(target);
      else fs.writeFileSync(target, contents);
    }
    const { binary, marker } = fakeBinary();
    const provider = new CodeafProvider(binary);
    const options = { env: Object.fromEntries(Object.entries(testCase.overlay).map(([k, v]) => [k, expand(v)])) };
    if (testCase.allow) {
      expect((await provider.execute('task', options)).result).toBe('answer');
      expect(fs.existsSync(marker)).toBe(true);
      expect(console.warn).not.toHaveBeenCalled();
    } else {
      const error = await provider.execute('task', options).catch((err: unknown) => err);
      expectRefusal(error);
      expect(error).toMatchObject({ binary });
      expect(fs.existsSync(marker)).toBe(false);
      expect(console.warn).toHaveBeenCalledExactlyOnceWith(contract.warning);
    }
    expect(install.ensureCodeafInstalled).not.toHaveBeenCalled();
  });
});

for (const provider of [undefined, 'codeaf', 'aforge']) {
  for (const pin of ['automatic', 'env', 'option']) {
    it(`K1 K2 refusal before resolution: ${provider ?? 'default'} ${pin}`, async () => {
      const { binary, marker } = fakeBinary();
      if (pin === 'env') vi.stubEnv('CODEAF_BIN', binary);
      const harness = await buildProvider({ provider, codeafBin: pin === 'option' ? binary : undefined });
      for (let call = 0; call < 2; call += 1) {
        expectRefusal(await harness.execute('task', {}).catch((err: unknown) => err));
      }
      expect(fs.existsSync(marker)).toBe(false);
      expect(install.ensureCodeafInstalled).not.toHaveBeenCalled();
      expect(console.warn).toHaveBeenCalledExactlyOnceWith(contract.warning);
    });
  }
}

it.each(['env', 'option'])('K5 legacy %s pin runs without a key', async (pin) => {
  const { binary, marker } = fakeBinary();
  if (pin === 'env') vi.stubEnv('AFORGE_BIN', binary);
  const provider = await buildProvider({ provider: 'aforge', aforgeBin: pin === 'option' ? binary : undefined });
  await provider.execute('task', {});
  expect(fs.existsSync(marker)).toBe(true);
  expect(console.warn).not.toHaveBeenCalled();
});

it.each([
  { provider: undefined, configured: true, key: '', legacy: false, warn: true },
  { provider: 'codeaf', configured: true, key: '', legacy: false, warn: true },
  { provider: 'aforge', configured: true, key: '', legacy: false, warn: true },
  { provider: 'codeaf', configured: true, key: 'key', legacy: false, warn: false },
  { provider: 'claude-code', configured: true, key: '', legacy: false, warn: false },
  { provider: undefined, configured: false, key: '', legacy: false, warn: false },
  { provider: 'aforge', configured: true, key: '', legacy: true, warn: false },
])('K6 Agent construction $provider configured=$configured key=$key legacy=$legacy', (testCase) => {
  vi.stubEnv('OPENROUTER_API_KEY', testCase.key);
  if (testCase.legacy) vi.stubEnv('AFORGE_BIN', '/fake/legacy');
  for (let count = 0; count < 2; count += 1) {
    new Agent({ nodeId: 'auth-test', harnessConfig: testCase.configured ? { provider: testCase.provider } : undefined });
  }
  if (testCase.warn) expect(console.warn).toHaveBeenCalledExactlyOnceWith(contract.warning);
  else expect(console.warn).not.toHaveBeenCalled();
});

it.each(['codeaf', 'aforge', 'claude-code'])('K6 environment provider %s', (provider) => {
  vi.stubEnv('AGENTFIELD_HARNESS_PROVIDER', provider);
  new Agent({ nodeId: 'auth-test', harnessConfig: {} });
  if (provider === 'claude-code') expect(console.warn).not.toHaveBeenCalled();
  else expect(console.warn).toHaveBeenCalledExactlyOnceWith(contract.warning);
});
