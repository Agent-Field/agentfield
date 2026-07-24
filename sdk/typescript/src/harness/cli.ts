import { spawn } from 'node:child_process';
import { applyOpenRouterAttributionEnv } from '../ai/openrouterAttribution.js';

export interface CliResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

const DEFAULT_IDLE_SECONDS = 120;

/**
 * Resolve the no-progress watchdog window in milliseconds.
 *
 * Precedence: explicit `idleSeconds`, then env
 * `AGENTFIELD_HARNESS_IDLE_SECONDS`, then `DEFAULT_IDLE_SECONDS` (120s).
 * A value <= 0 disables the watchdog and returns `undefined`.
 */
function resolveIdleMs(idleSeconds?: number): number | undefined {
  let seconds = idleSeconds;
  if (seconds === undefined) {
    const raw = process.env.AGENTFIELD_HARNESS_IDLE_SECONDS;
    const parsed = raw !== undefined ? Number(raw) : NaN;
    seconds = Number.isFinite(parsed) ? parsed : DEFAULT_IDLE_SECONDS;
  }
  return seconds > 0 ? seconds * 1000 : undefined;
}

export function runCli(
  cmd: string[],
  options?: {
    env?: Record<string, string>;
    cwd?: string;
    timeout?: number;
    idleSeconds?: number;
  }
): Promise<CliResult> {
  return new Promise((resolve, reject) => {
    const [bin, ...args] = cmd;
    const env = { ...process.env, ...options?.env };
    applyOpenRouterAttributionEnv(env);
    // 'ignore' on stdin gives the child an immediate EOF instead of an open
    // pipe that never closes (a hang risk if the child probes stdin).
    const proc = spawn(bin, args, {
      env,
      cwd: options?.cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    let lastActivity = Date.now();

    // Both stdout and stderr are drained concurrently via their own 'data'
    // listeners, so a full stderr pipe cannot deadlock the read of stdout.
    proc.stdout.on('data', (data: Uint8Array | string) => {
      stdout += data.toString();
      lastActivity = Date.now();
    });
    proc.stderr.on('data', (data: Uint8Array | string) => {
      stderr += data.toString();
      lastActivity = Date.now();
    });

    // Wall-clock cap: outer bound, unchanged behavior.
    const timer = options?.timeout
      ? setTimeout(() => {
          if (settled) {
            return;
          }
          settled = true;
          cleanup();
          proc.kill('SIGKILL');
          reject(new Error(`CLI timed out after ${options.timeout}ms`));
        }, options.timeout)
      : undefined;

    // No-progress (idle) watchdog: if no chunk arrives for idleMs, kill the
    // child and reject, rather than waiting for the full wall-clock cap.
    const idleMs = resolveIdleMs(options?.idleSeconds);
    const idleTimer = idleMs
      ? setInterval(() => {
          if (settled) {
            return;
          }
          if (Date.now() - lastActivity >= idleMs) {
            settled = true;
            cleanup();
            proc.kill('SIGKILL');
            reject(
              new Error(
                `CLI made no progress for ${Math.round(idleMs / 1000)}s`
              )
            );
          }
        }, Math.min(idleMs, 1000))
      : undefined;

    function cleanup(): void {
      if (timer) {
        clearTimeout(timer);
      }
      if (idleTimer) {
        clearInterval(idleTimer);
      }
    }

    proc.on('close', (code) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve({ stdout, stderr, exitCode: code ?? 0 });
    });

    proc.on('error', (err) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
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
