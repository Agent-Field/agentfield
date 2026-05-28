import type { HarnessProvider } from './base.js';
import type { RawResult } from '../types.js';
import { createMetrics, createRawResult } from '../types.js';

type QueryInput = {
  prompt: string;
  options: Record<string, unknown>;
};

type ClaudeSdkModule = {
  query: (input: QueryInput) => AsyncIterable<unknown>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function getString(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key];
  return typeof value === 'string' ? value : undefined;
}

function getNumber(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key];
  return typeof value === 'number' ? value : undefined;
}

export class ClaudeCodeProvider implements HarnessProvider {
  public async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    let sdk: ClaudeSdkModule;
    try {
      const mod = await import('@anthropic-ai/claude-agent-sdk');
      sdk = mod as ClaudeSdkModule;
    } catch {
      throw new Error(
        "@anthropic-ai/claude-agent-sdk is required for the 'claude-code' provider. " +
          'Install it with: npm install @anthropic-ai/claude-agent-sdk'
      );
    }

    const agentOptions: Record<string, unknown> = {};
    if (options.model !== undefined) agentOptions.model = options.model;
    if (options.cwd !== undefined) agentOptions.cwd = options.cwd;
    if (options.maxTurns !== undefined) agentOptions.max_turns = options.maxTurns;
    if (options.tools !== undefined) agentOptions.allowed_tools = options.tools;
    if (options.systemPrompt !== undefined) agentOptions.system_prompt = options.systemPrompt;
    if (options.maxBudgetUsd !== undefined) agentOptions.max_budget_usd = options.maxBudgetUsd;
    if (options.permissionMode !== undefined) {
      const modeMap: Record<string, string> = { auto: 'bypassPermissions', plan: 'plan' };
      const raw = String(options.permissionMode);
      agentOptions.permission_mode = modeMap[raw] ?? raw;
    }
    if (options.env !== undefined) agentOptions.env = options.env;

    const messages: Array<Record<string, unknown>> = [];
    let resultText: string | undefined;
    let totalCost: number | undefined;
    let numTurns = 0;
    let sessionId = '';
    const startApi = Date.now();

    try {
      for await (const msg of sdk.query({ prompt, options: agentOptions })) {
        const msgObj = isRecord(msg) ? msg : { raw: String(msg) };
        messages.push(msgObj);

        const msgType = getString(msgObj, 'type') ?? '';
        if (msgType === 'result') {
          const resultValue = msgObj.result ?? msgObj.text;
          resultText = typeof resultValue === 'string' ? resultValue : resultValue == null ? '' : String(resultValue);

          const sid = getString(msgObj, 'session_id');
          if (sid !== undefined) {
            sessionId = sid;
          }

          const costUsd = getNumber(msgObj, 'cost_usd');
          const totalCostUsd = getNumber(msgObj, 'total_cost_usd');
          if (costUsd !== undefined) {
            totalCost = costUsd;
          } else if (totalCostUsd !== undefined) {
            totalCost = totalCostUsd;
          }

          const turns = getNumber(msgObj, 'num_turns');
          numTurns = turns === undefined ? messages.length : Math.trunc(turns);
        } else if (msgType === 'assistant' && resultText === undefined) {
          let content: unknown = msgObj.content;
          if (content === undefined && isRecord(msgObj.message)) {
            content = msgObj.message.content;
          }

          if (typeof content === 'string') {
            resultText = content;
          } else if (Array.isArray(content)) {
            for (const block of content) {
              if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
                resultText = block.text;
              }
            }
          }
        }
      }

      return createRawResult({
        result: resultText,
        messages,
        metrics: createMetrics({
          durationApiMs: Date.now() - startApi,
          numTurns,
          totalCostUsd: totalCost,
          sessionId,
        }),
        isError: false,
      });
    } catch (error: unknown) {
      return createRawResult({
        result: undefined,
        messages,
        metrics: createMetrics({ durationApiMs: Date.now() - startApi, sessionId }),
        isError: true,
        errorMessage: error instanceof Error ? error.message : String(error),
      });
    }
  }
}
