import type { HarnessProvider } from './base.js';
import { resolveRoot } from './base.js';
import type { RawResult } from '../types.js';
import { createRawResult, createMetrics } from '../types.js';
import { runCli } from '../cli.js';
import { resolveModelAndVariant } from '../modelVariant.js';
import {
  isOpenRouterRequest,
  openRouterAttributionHeaders,
} from '../../ai/openrouterAttribution.js';

const ANSI_PATTERN = /\x1B\[[0-?]*[ -/]*[@-~]/g;
const STDERR_ERROR_PATTERNS = [
  /^Error:/m,
  /\bModel not found\b/,
  /\bAuthenticationError\b/,
  /\bUnauthorized\b/,
  /\bAPIError\b/,
];

const openCodeConfigSchema = 'https://opencode.ai/config.json';
const openCodeAgentName = 'agentfield-harness';
const openCodeDefaultSteps = 500;
const openCodeInlineSystemPromptEnv = 'AGENTFIELD_OPENCODE_INLINE_SYSTEM_PROMPT';
const openCodeStepsEnv = 'AGENTFIELD_OPENCODE_STEPS';
const agentFieldWorkerInstruction =
  'You are an AgentField-launched worker. Complete the assigned prompt directly. ' +
  'Do not invoke AgentField orchestration, the `af` CLI, `swe-planner.plan`, ' +
  'or delegate work back to AgentField.';
const trueEnvValues = new Set(['1', 'true', 'yes', 'on']);

type OpenCodeConfig = Record<string, unknown>;

function openCodePermissions(): OpenCodeConfig {
  return {
    '*': 'allow',
    skill: { 'agentfield*': 'deny' },
    question: 'deny',
    task: 'deny',
  };
}

function agentSystemPrompt(options: Record<string, unknown>): string {
  // HarnessRunner passes camelCase option keys (`systemPrompt`); the snake_case
  // spelling is the Python-matching alias other providers also accept. Reading
  // only `system_prompt` meant a system prompt set through the public
  // TypeScript API never reached opencode at all.
  const systemPrompt = options.systemPrompt ?? options.system_prompt;
  const callerPrompt = typeof systemPrompt === 'string' ? systemPrompt.trim() : '';
  return callerPrompt
    ? `${callerPrompt}\n\n${agentFieldWorkerInstruction}`
    : agentFieldWorkerInstruction;
}

function optionEnvValue(options: Record<string, unknown>, name: string): string | undefined {
  const env = options.env;
  if (
    typeof env === 'object' &&
    env !== null &&
    Object.prototype.hasOwnProperty.call(env, name)
  ) {
    const value = (env as Record<string, unknown>)[name];
    return typeof value === 'string' ? value : undefined;
  }
  return process.env[name];
}

function inlineSystemPromptEnabled(options: Record<string, unknown>): boolean {
  const value = optionEnvValue(options, openCodeInlineSystemPromptEnv);
  return typeof value === 'string' && trueEnvValues.has(value.trim().toLowerCase());
}

function openCodeSteps(options: Record<string, unknown>): number {
  const value = optionEnvValue(options, openCodeStepsEnv);
  if (typeof value === 'string' && /^\+?\d+$/.test(value.trim())) {
    const steps = Number(value.trim());
    if (Number.isSafeInteger(steps) && steps > 0) {
      return steps;
    }
  }
  return openCodeDefaultSteps;
}

function isConfigObject(value: unknown): value is OpenCodeConfig {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function deepMergeConfig(base: OpenCodeConfig, overlay: OpenCodeConfig): OpenCodeConfig {
  const merged: OpenCodeConfig = { ...base };
  for (const [key, value] of Object.entries(overlay)) {
    const existing = merged[key];
    merged[key] = isConfigObject(existing) && isConfigObject(value)
      ? deepMergeConfig(existing, value)
      : value;
  }
  return merged;
}

function normalizeAgentFieldHarnessConfig(
  config: OpenCodeConfig,
  stripSystemPrompt: boolean,
): void {
  const agents = config.agent;
  if (!isConfigObject(agents)) {
    return;
  }
  const selectedAgent = agents[openCodeAgentName];
  if (!isConfigObject(selectedAgent)) {
    return;
  }

  if (stripSystemPrompt) {
    delete selectedAgent.prompt;
  }

  const permission = selectedAgent.permission;
  if (!isConfigObject(permission)) {
    return;
  }

  const ordered: OpenCodeConfig = {};
  if ('*' in permission) {
    ordered['*'] = permission['*'];
  }
  for (const [key, value] of Object.entries(permission)) {
    if (!['*', 'skill', 'question', 'task'].includes(key)) {
      ordered[key] = value;
    }
  }
  if ('skill' in permission) {
    const skill = permission.skill;
    if (isConfigObject(skill)) {
      const orderedSkill: OpenCodeConfig = {};
      for (const [key, value] of Object.entries(skill)) {
        if (key !== 'agentfield*') {
          orderedSkill[key] = value;
        }
      }
      if ('agentfield*' in skill) {
        orderedSkill['agentfield*'] = skill['agentfield*'];
      }
      ordered.skill = orderedSkill;
    } else {
      ordered.skill = skill;
    }
  }
  if ('question' in permission) {
    ordered.question = permission.question;
  }
  if ('task' in permission) {
    ordered.task = permission.task;
  }
  selectedAgent.permission = ordered;
}

function mergeOpenCodeConfigContent(
  existingContent: string | undefined,
  overlayContent: string,
  stripSystemPrompt = false,
): string {
  if (!existingContent?.trim()) {
    return overlayContent;
  }

  let existing: unknown;
  let overlay: unknown;
  try {
    existing = JSON.parse(existingContent);
    overlay = JSON.parse(overlayContent);
  } catch (error) {
    throw new Error(
      'OPENCODE_CONFIG_CONTENT must be valid JSON when supplied to the AgentField OpenCode provider',
      { cause: error },
    );
  }
  if (!isConfigObject(existing)) {
    throw new Error(
      'OPENCODE_CONFIG_CONTENT must contain a JSON object when supplied to the AgentField OpenCode provider',
    );
  }
  if (!isConfigObject(overlay)) {
    throw new Error('AgentField OpenCode overlay must contain a JSON object');
  }

  const merged = deepMergeConfig(existing, overlay);
  normalizeAgentFieldHarnessConfig(merged, stripSystemPrompt);
  return JSON.stringify(merged);
}

function inlineOpenCodePrompt(prompt: string, options: Record<string, unknown>): string {
  return `SYSTEM INSTRUCTIONS:\n${agentSystemPrompt(options)}\n\n---\n\nUSER REQUEST:\n${prompt}`;
}

function buildOpenCodeConfigContent(
  options: Record<string, unknown>,
  modelValue: string | undefined,
  variantValue: string | undefined,
  includeSystemPrompt = true,
): string {
  const agent: OpenCodeConfig = {
    mode: 'primary',
    steps: openCodeSteps(options),
    permission: openCodePermissions(),
  };
  if (includeSystemPrompt) {
    agent.prompt = agentSystemPrompt(options);
  }
  if (modelValue) {
    agent.model = modelValue;
  }
  if (variantValue) {
    agent.reasoningEffort = variantValue;
  }

  return JSON.stringify({
    $schema: openCodeConfigSchema,
    default_agent: openCodeAgentName,
    agent: { [openCodeAgentName]: agent },
  });
}

function extractOpenCodeError(stderr: string): string {
  const lines = stderr.split(/\r?\n/);
  for (let index = 0; index < lines.length; index += 1) {
    if (STDERR_ERROR_PATTERNS.some((pattern) => pattern.test(lines[index]))) {
      return lines.slice(Math.max(0, index - 1), index + 5).join('\n').trim().slice(0, 1000);
    }
  }
  return stderr.slice(0, 1000);
}

export class OpenCodeProvider implements HarnessProvider {
  private readonly bin: string;

  constructor(binPath = 'opencode') {
    this.bin = binPath;
  }

  async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    // opencode v1.4+ uses the `run` subcommand. Prior `-c <dir> -p <prompt>`
    // syntax is broken on v1.14: `-c` now means `--continue` (a boolean) and
    // there is no top-level `-p` flag, so opencode prints help to stdout and
    // exits 0 — the SDK then captures the help screen as the LLM response.
    // See agentfield#582.
    const cmd = [this.bin, 'run'];

    const inlineSystemPrompt = inlineSystemPromptEnabled(options);
    cmd.push('--agent', openCodeAgentName);

    // Use --dir for project directory.
    const root = resolveRoot(options);
    if (root) {
      cmd.push('--dir', root);
    }

    const env: Record<string, string> = { ...(options.env as Record<string, string>) };

    // Pass model via -m flag on the run subcommand (not env var). A
    // "#variant" suffix on the model (or an explicit options.variant) maps
    // to --variant — opencode's provider-specific reasoning effort (e.g.
    // high, max, minimal).
    const { model: modelValue, variant: variantValue } = resolveModelAndVariant(options);
    if (modelValue) {
      cmd.push('-m', modelValue);
    }
    if (variantValue) {
      cmd.push('--variant', variantValue);
    }

    // The attribution overlay keys off the base model — a "#variant" suffix
    // would otherwise leak into the config's model slug.
    if (
      modelValue &&
      isOpenRouterRequest({ model: modelValue }) &&
      !env.OPENCODE_CONFIG_CONTENT &&
      !process.env.OPENCODE_CONFIG_CONTENT
    ) {
      const modelSlug = modelValue.slice('openrouter/'.length);
      const headers = openRouterAttributionHeaders({ env: { ...process.env, ...env } });
      if (modelSlug && Object.keys(headers).length > 0) {
        env.OPENCODE_CONFIG_CONTENT = JSON.stringify({
          provider: {
            openrouter: {
              models: {
                [modelSlug]: { headers },
              },
            },
          },
        });
      }
    }

    const existingConfig = Object.prototype.hasOwnProperty.call(env, 'OPENCODE_CONFIG_CONTENT')
      ? env.OPENCODE_CONFIG_CONTENT
      : process.env.OPENCODE_CONFIG_CONTENT;
    env.OPENCODE_CONFIG_CONTENT = mergeOpenCodeConfigContent(
      existingConfig,
      buildOpenCodeConfigContent(options, modelValue, variantValue, !inlineSystemPrompt),
      inlineSystemPrompt,
    );

    const effectivePrompt = inlineSystemPrompt
      ? inlineOpenCodePrompt(prompt, options)
      : prompt;

    // Prompt is the positional `message` arg to `opencode run`.
    cmd.push(effectivePrompt);

    const startApi = Date.now();
    try {
      const { stdout, stderr, exitCode } = await runCli(cmd, { env });

      const resultText = stdout.trim() || undefined;
      const cleanStderr = stderr.trim().replace(ANSI_PATTERN, '');
      let isError = false;
      let errorMessage: string | undefined;

      if (exitCode < 0) {
        isError = true;
        errorMessage = cleanStderr
          ? `Process killed by signal ${-exitCode}. stderr: ${cleanStderr.slice(0, 500)}`
          : `Process killed by signal ${-exitCode}.`;
      } else if (exitCode !== 0 && !resultText) {
        isError = true;
        errorMessage = cleanStderr
          ? extractOpenCodeError(cleanStderr)
          : `Process exited with code ${exitCode} and produced no output.`;
      } else if (
        !resultText &&
        cleanStderr &&
        STDERR_ERROR_PATTERNS.some((pattern) => pattern.test(cleanStderr))
      ) {
        isError = true;
        errorMessage = extractOpenCodeError(cleanStderr);
      }

      return createRawResult({
        result: resultText,
        messages: [],
        metrics: createMetrics({
          durationApiMs: Date.now() - startApi,
          numTurns: resultText ? 1 : 0,
          sessionId: '',
          model: modelValue,
        }),
        isError,
        errorMessage,
        failureType: isError ? 'crash' : 'none',
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('ENOENT')) {
        return createRawResult({
          isError: true,
          errorMessage: `OpenCode binary not found at '${this.bin}'. Install: https://github.com/opencode-ai/opencode`,
          metrics: createMetrics({ durationApiMs: Date.now() - startApi }),
        });
      }
      return createRawResult({
        isError: true,
        errorMessage: msg,
        metrics: createMetrics({ durationApiMs: Date.now() - startApi }),
      });
    }
  }
}
