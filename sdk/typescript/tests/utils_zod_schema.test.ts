import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import * as z4 from 'zod/v4';

import { resetZod4ConverterCache, zod4ToJsonSchema } from '../src/utils/zod-schema.js';

// Contract: a zod 4 schema is converted by the APPLICATION's own zod copy when
// one is resolvable from the working directory, because that is the copy that
// built the schema instance (a different copy converts the structure but drops
// registry-backed metadata such as descriptions). The SDK's own zod is the
// fallback.
describe('zod4ToJsonSchema converter resolution', () => {
  const tempRoots: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();
    resetZod4ConverterCache();
    for (const root of tempRoots.splice(0)) {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('prefers the zod copy resolvable from the working directory', () => {
    const root = mkdtempSync(path.join(tmpdir(), 'agentfield-zod4-'));
    tempRoots.push(root);
    const fakeZod = path.join(root, 'node_modules', 'zod');
    mkdirSync(fakeZod, { recursive: true });
    writeFileSync(path.join(root, 'package.json'), JSON.stringify({ name: 'app', version: '0.0.0' }));
    writeFileSync(path.join(fakeZod, 'package.json'), JSON.stringify({ name: 'zod', version: '4.99.0', main: 'index.js' }));
    writeFileSync(
      path.join(fakeZod, 'index.js'),
      "module.exports = { toJSONSchema: () => ({ type: 'object', properties: {}, 'x-converter': 'app-copy' }) };",
    );
    vi.spyOn(process, 'cwd').mockReturnValue(root);
    resetZod4ConverterCache();

    const out = zod4ToJsonSchema(z4.object({ name: z4.string() }));

    expect(out['x-converter']).toBe('app-copy');
  });

  it('falls back to the SDK-relative zod when the working directory has none', () => {
    const root = mkdtempSync(path.join(tmpdir(), 'agentfield-zod4-empty-'));
    tempRoots.push(root);
    writeFileSync(path.join(root, 'package.json'), JSON.stringify({ name: 'app', version: '0.0.0' }));
    vi.spyOn(process, 'cwd').mockReturnValue(root);
    resetZod4ConverterCache();

    // Structure only: zod 3.25's `zod/v4` shim keeps separate metadata
    // registries for its ESM and CJS builds, so `.describe()` fidelity across
    // that boundary is not part of this contract (zod 4 proper shares one
    // registry between both builds — verified end to end).
    const out = zod4ToJsonSchema(z4.object({ name: z4.string() }));

    expect(out).toMatchObject({
      type: 'object',
      properties: { name: { type: 'string' } },
      required: ['name'],
    });
  });
});
