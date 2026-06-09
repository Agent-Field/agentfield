import { spawn } from 'node:child_process';

export interface CliResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

export function runCli(
  cmd: string[],
  options?: { env?: Record<string, string>; cwd?: string; timeout?: number }
): Promise<CliResult> {
  return new Promise((resolve, reject) => {
    const [bin, ...args] = cmd;
    const proc = spawn(bin, args, {
      env: { ...process.env, ...options?.env },
      cwd: options?.cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data: Uint8Array | string) => {
      stdout += data.toString();
    });
    proc.stderr.on('data', (data: Uint8Array | string) => {
      stderr += data.toString();
    });

    const timer = options?.timeout
      ? setTimeout(() => {
          proc.kill();
          reject(new Error(`CLI timed out after ${options.timeout}ms`));
        }, options.timeout)
      : undefined;

    proc.on('close', (code) => {
      if (timer) {
        clearTimeout(timer);
      }
      resolve({ stdout, stderr, exitCode: code ?? 0 });
    });

    proc.on('error', (err) => {
      if (timer) {
        clearTimeout(timer);
      }
      reject(err);
    });
  });
}

export function parseJsonl(text: string): Array<Record<string, unknown>> {
  const events: Array<Record<string, unknown>> = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      events.push(JSON.parse(trimmed) as Record<string, unknown>);
    } catch {
      continue;
    }
  }
  return events;
}

export function extractFinalText(events: Array<Record<string, unknown>>): string | undefined {
  let result: string | undefined;
  for (const event of events) {
    const type = event.type;
    if (type === 'item.completed') {
      const item = event.item;
      if (typeof item === 'object' && item !== null) {
        const itemType = (item as Record<string, unknown>).type;
        const itemText = (item as Record<string, unknown>).text;
        if (itemType === 'agent_message' && typeof itemText === 'string') {
          result = itemText;
        }
      }
    } else if (type === 'result') {
      const candidate = event.result ?? event.text;
      if (typeof candidate === 'string') {
        result = candidate;
      }
    } else if (type === 'turn.completed' && typeof event.text === 'string') {
      result = event.text;
    } else if ((type === 'message' || type === 'assistant') && typeof event.content === 'string') {
      result = event.content;
    }
  }
  return result;
}
