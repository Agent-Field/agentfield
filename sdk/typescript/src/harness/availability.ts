import fs from 'node:fs';
import path from 'node:path';

type Environment = Readonly<Record<string, string | undefined>>;

interface ProviderSpec {
  binary: string | null;
  versionArgs: readonly string[];
  installCommand: string;
  authEnvVars: readonly string[];
}

const PROVIDER_SPECS = {
  aforge: {
    binary: 'aforge',
    versionArgs: ['version'],
    installCommand: 'af aforge ensure',
    authEnvVars: ['OPENROUTER_API_KEY'],
  },
  'claude-code': {
    binary: null,
    versionArgs: [],
    installCommand: 'npm install @anthropic-ai/claude-agent-sdk',
    authEnvVars: ['ANTHROPIC_API_KEY'],
  },
  codex: {
    binary: 'codex',
    versionArgs: ['--version'],
    installCommand: 'npm install -g @openai/codex',
    authEnvVars: ['OPENAI_API_KEY'],
  },
  gemini: {
    binary: 'gemini',
    versionArgs: ['--version'],
    installCommand: 'npm install -g @google/gemini-cli',
    authEnvVars: ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
  },
  omp: {
    binary: 'omp',
    versionArgs: ['--version'],
    installCommand: 'curl -fsSL https://omp.sh/install | sh',
    authEnvVars: [
      'OPENROUTER_API_KEY',
      'ANTHROPIC_API_KEY',
      'OPENAI_API_KEY',
      'GEMINI_API_KEY',
      'GOOGLE_API_KEY',
    ],
  },
  opencode: {
    binary: 'opencode',
    versionArgs: ['--version'],
    installCommand: 'curl -fsSL https://opencode.ai/install | bash',
    authEnvVars: [],
  },
  pi: {
    binary: 'pi',
    versionArgs: ['--version'],
    installCommand: 'npm install -g --ignore-scripts @earendil-works/pi-coding-agent',
    authEnvVars: [
      'OPENROUTER_API_KEY',
      'ANTHROPIC_API_KEY',
      'OPENAI_API_KEY',
      'GEMINI_API_KEY',
      'GOOGLE_API_KEY',
    ],
  },
} as const satisfies Record<string, ProviderSpec>;

export type HarnessProviderName = keyof typeof PROVIDER_SPECS;

export const SUPPORTED_PROVIDER_NAMES = Object.freeze(
  Object.keys(PROVIDER_SPECS).sort() as HarnessProviderName[]
);

export interface ProviderHealth {
  provider: HarnessProviderName;
  binary: string | null;
  installed: boolean;
  version: string | null;
  auth: 'configured' | 'unknown';
  usable: boolean;
  installCommand: string;
  authEnvVars: readonly string[];
  issues: readonly string[];
}

export class HarnessProviderUnavailable extends Error {
  public readonly provider: string;
  public readonly binary: string;
  public readonly installCommand: string;
  public readonly missingAuthEnv: readonly string[];

  public constructor(
    provider: string,
    options: {
      binary: string;
      installCommand: string;
      missingAuthEnv?: readonly string[];
    }
  ) {
    const missingAuthEnv = [...(options.missingAuthEnv ?? [])];
    let message =
      `Harness provider '${provider}' is unavailable: binary '${options.binary}' was not found. ` +
      `Install it with: ${options.installCommand}`;
    if (missingAuthEnv.length > 0) {
      message += `. Configure one of: ${missingAuthEnv.join(', ')}`;
    }
    super(message);
    this.name = 'HarnessProviderUnavailable';
    this.provider = provider;
    this.binary = options.binary;
    this.installCommand = options.installCommand;
    this.missingAuthEnv = missingAuthEnv;
  }
}

export type BinaryResolver = (binary: string, env: Environment) => string | undefined;
/** Receives the resolved binary path followed by the provider's version arguments. */
export type VersionProbe = (command: string[]) => Promise<string>;
export type WrapperProbe = (provider: 'claude-code') => Promise<boolean>;

export interface HarnessDoctorOptions {
  /** Environment values used for the offline check (binary discovery and the authentication signal). */
  env?: Environment;
  resolveBinary?: BinaryResolver;
  versionProbe?: VersionProbe;
  wrapperProbe?: WrapperProbe;
}

function executableExtensions(binary: string, env: Environment): string[] {
  if (process.platform !== 'win32' || path.extname(binary)) {
    return [''];
  }
  const pathExt = env.PATHEXT ?? env.PathExt ?? '.COM;.EXE;.BAT;.CMD';
  return pathExt.split(';').filter(Boolean).map((extension) => extension.toLowerCase());
}

function isExecutableFile(candidate: string): boolean {
  try {
    if (!fs.statSync(candidate).isFile()) {
      return false;
    }
    fs.accessSync(candidate, process.platform === 'win32' ? fs.constants.F_OK : fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/** Resolve an executable without invoking it, mirroring `shutil.which`. */
export function findExecutable(binary: string, env: Environment = process.env): string | undefined {
  const extensions = executableExtensions(binary, env);
  const hasPath = path.isAbsolute(binary) || binary.includes('/') || binary.includes('\\');
  const directories = hasPath
    ? ['']
    : (env.PATH ?? env.Path ?? env.path ?? '').split(path.delimiter).filter(Boolean);

  for (const directory of directories) {
    for (const extension of extensions) {
      const candidate = `${hasPath ? binary : path.join(directory, binary)}${extension}`;
      if (isExecutableFile(candidate)) {
        return path.resolve(candidate);
      }
    }
  }
  return undefined;
}

export function ensureCliAvailable(
  provider: string,
  binary: string,
  resolveBinary: BinaryResolver = findExecutable,
  env: Environment = process.env
): string {
  const resolved = resolveBinary(binary, env);
  if (resolved) {
    return resolved;
  }
  throw providerUnavailable(provider, binary);
}

export function providerUnavailable(provider: string, binary: string): HarnessProviderUnavailable {
  const spec = PROVIDER_SPECS[provider as HarnessProviderName];
  return new HarnessProviderUnavailable(provider, {
    binary,
    installCommand: spec?.installCommand ?? 'install the configured harness provider binary',
  });
}

const WINDOWS_BATCH_EXTENSIONS = new Set(['.bat', '.cmd']);

function isWindowsBatchFile(command: string): boolean {
  return process.platform === 'win32'
    && WINDOWS_BATCH_EXTENSIONS.has(path.extname(command).toLowerCase());
}

const BATCH_PROBE_ENV_PREFIX = 'AGENTFIELD_PROBE_ARG_';

/**
 * Build a `cmd.exe` invocation for a `.cmd`/`.bat` shim. Node refuses to spawn
 * batch files without a shell (CVE-2024-27980), so they have to go through
 * cmd.exe. The `/c` line is a fixed template of `%VAR%` references; the path
 * and arguments travel in the child's environment.
 *
 * cmd.exe parses percent expansion BEFORE it splits the line or interprets
 * metacharacters: each `%VAR%` is replaced first, and only afterwards are the
 * resulting characters treated as separators or operators. That is why values
 * reach the command line safely here — the template wraps every reference in
 * double quotes, so after expansion each token is already a quoted argument,
 * and `&`, `^`, `(`, `%`, `!` and spaces inside it stay literal. `/s` makes
 * cmd.exe strip only the outer quote pair of the fixed template, again leaving
 * each `"%VAR%"` expansion quoted instead of unwrapping the first and last
 * quote on the line. `/v:off` disables delayed `!` expansion so `!` in a value
 * stays literal even when delayed expansion is otherwise enabled. `/d` skips
 * AutoRun registry scripts.
 *
 * This is not a claim that environment data can never be interpreted: the
 * guarantee rests on the quote/control-character invariant below. The guard in
 * `defaultVersionProbe` runs before any value is copied into the child's
 * environment, so no `%VAR%` can ever expand to a double quote or a control
 * character, and therefore no expansion can close a quote early or inject
 * command syntax. Passing the path as a bare argv element instead breaks on
 * `&` and `(x86)` because cmd.exe without `/s` drops the quotes Node adds.
 */
function batchProbeInvocation(command: string[]): {
  file: string;
  args: string[];
  env: NodeJS.ProcessEnv;
} {
  const env: NodeJS.ProcessEnv = { ...process.env };
  const tokens = command.map((value, index) => {
    const name = `${BATCH_PROBE_ENV_PREFIX}${index}`;
    env[name] = value;
    return `"%${name}%"`;
  });
  return {
    file: process.env.ComSpec ?? 'cmd.exe',
    // /s removes the outer quote pair of the fixed template, so each "%VAR%"
    // expansion stays quoted.
    // /v:off disables delayed ! expansion; /d disables AutoRun.
    args: ['/d', '/v:off', '/s', '/c', `"${tokens.join(' ')}"`],
    env,
  };
}

async function defaultVersionProbe(command: string[]): Promise<string> {
  const { execFile } = await import('node:child_process');

  return new Promise((resolve, reject) => {
    const onProbeOutput = (error: Error | null, stdout: string, stderr: string): void => {
      if (error) {
        reject(error);
        return;
      }
      const output = String(stdout || stderr).trim();
      resolve(output ? output.split(/\r?\n/, 1)[0] : 'unknown');
    };

    if (isWindowsBatchFile(command[0])) {
      // Guard the cmd.exe branch only: every value is copied into the child's
      // environment, so reject a double quote or any control character before
      // insertion. The direct branch below is left unchanged.
      for (const value of command) {
        for (let index = 0; index < value.length; index += 1) {
          const codePoint = value.charCodeAt(index);
          if (codePoint === 0x22 || codePoint < 0x20) {
            reject(new TypeError(
              `Windows batch probe values must not contain double quotes or control characters; ` +
              `offending code point ${codePoint} in ${JSON.stringify(value)}`
            ));
            return;
          }
        }
      }

      const { file, args, env } = batchProbeInvocation(command);
      execFile(
        file,
        args,
        { timeout: 2_000, windowsHide: true, windowsVerbatimArguments: true, env },
        onProbeOutput
      );
      return;
    }

    execFile(
      command[0],
      command.slice(1),
      { timeout: 2_000, windowsHide: true, windowsVerbatimArguments: false },
      onProbeOutput
    );
  });
}

async function defaultWrapperProbe(): Promise<boolean> {
  try {
    await import('@anthropic-ai/claude-agent-sdk');
    return true;
  } catch {
    return false;
  }
}

function authStatus(spec: ProviderSpec, env: Environment): 'configured' | 'unknown' {
  return spec.authEnvVars.some((name) => Boolean(env[name])) ? 'configured' : 'unknown';
}

export async function harnessDoctor(
  providers?: readonly string[],
  options: HarnessDoctorOptions = {}
): Promise<ProviderHealth[]> {
  const selected = providers && providers.length > 0
    ? [...providers]
    : [...SUPPORTED_PROVIDER_NAMES];
  const unknown = selected.filter((provider) => !(provider in PROVIDER_SPECS)).sort();
  if (unknown.length > 0) {
    throw new Error(
      `Unknown harness provider: "${unknown[0]}". Supported providers: ${SUPPORTED_PROVIDER_NAMES.join(', ')}`
    );
  }

  const env = options.env ?? process.env;
  const resolveBinary = options.resolveBinary ?? findExecutable;
  const versionProbe = options.versionProbe ?? defaultVersionProbe;
  const wrapperProbe = options.wrapperProbe ?? defaultWrapperProbe;
  const reports: ProviderHealth[] = [];

  for (const providerValue of selected) {
    const provider = providerValue as HarnessProviderName;
    const spec = PROVIDER_SPECS[provider];
    if (provider === 'claude-code') {
      const installed = await wrapperProbe(provider);
      reports.push({
        provider,
        binary: null,
        installed,
        version: null,
        auth: authStatus(spec, env),
        usable: installed,
        installCommand: spec.installCommand,
        authEnvVars: [...spec.authEnvVars],
        issues: installed ? [] : ['wrapper_not_installed'],
      });
      continue;
    }

    const binaryName = spec.binary as string;
    const binary = resolveBinary(binaryName, env) ?? null;
    const issues: string[] = [];
    let version: string | null = null;
    if (binary === null) {
      issues.push('binary_not_found');
    } else {
      try {
        const output = (await versionProbe([binary, ...spec.versionArgs])).trim();
        version = output ? output.split(/\r?\n/, 1)[0] : 'unknown';
      } catch {
        issues.push('version_probe_failed');
      }
    }

    reports.push({
      provider,
      binary,
      installed: binary !== null,
      version,
      auth: authStatus(spec, env),
      usable: binary !== null && issues.length === 0,
      installCommand: spec.installCommand,
      authEnvVars: [...spec.authEnvVars],
      issues,
    });
  }
  return reports;
}
