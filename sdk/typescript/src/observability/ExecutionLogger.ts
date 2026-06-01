import type { WriteStream } from 'node:tty';

export type ExecutionLogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface ExecutionLogContext {
  executionId?: string;
  runId?: string;
  workflowId?: string;
  rootWorkflowId?: string;
  parentExecutionId?: string;
  sessionId?: string;
  actorId?: string;
  agentNodeId?: string;
  reasonerId?: string;
  callerDid?: string;
  targetDid?: string;
  agentNodeDid?: string;
}

export interface ExecutionLogAttributes {
  [key: string]: unknown;
}

export interface ExecutionLogEntry extends ExecutionLogContext {
  v: 1;
  ts: string;
  level: ExecutionLogLevel;
  source: string;
  message: string;
  eventType?: string;
  systemGenerated?: boolean;
  attributes?: ExecutionLogAttributes;
}

export interface ExecutionLogWireEntry {
  v: 1;
  ts: string;
  execution_id?: string;
  run_id?: string;
  workflow_id?: string;
  root_workflow_id?: string;
  parent_execution_id?: string;
  session_id?: string;
  actor_id?: string;
  agent_node_id?: string;
  reasoner_id?: string;
  caller_did?: string;
  target_did?: string;
  agent_node_did?: string;
  level: ExecutionLogLevel;
  source: string;
  event_type?: string;
  message: string;
  attributes?: ExecutionLogAttributes;
  system_generated?: boolean;
}

export interface ExecutionLogBatchPayload {
  entries: ExecutionLogWireEntry[];
}

export type ExecutionLogTransportPayload = ExecutionLogWireEntry | ExecutionLogBatchPayload;

export function isExecutionLogBatchPayload(
  payload: ExecutionLogTransportPayload
): payload is ExecutionLogBatchPayload {
  return 'entries' in payload;
}

export interface ExecutionLogEmitOptions {
  eventType?: string;
  source?: string;
  systemGenerated?: boolean;
}

export interface ExecutionLogTransport {
  emit(payload: ExecutionLogTransportPayload): void | Promise<void>;
}

export interface ExecutionLoggerOptions {
  contextProvider?: () => ExecutionLogContext | undefined;
  transport?: ExecutionLogTransport;
  mirrorToStdout?: boolean;
  stdout?: Pick<WriteStream, 'write'>;
  source?: string;
}

function safeJsonStringify(value: unknown): string {
  const seen = new WeakSet<object>();
  return JSON.stringify(value, (_key, current) => {
    if (typeof current === 'bigint') {
      return current.toString();
    }
    if (typeof current === 'object' && current !== null) {
      if (seen.has(current)) {
        return '[Circular]';
      }
      seen.add(current);
    }
    return current;
  });
}

function mergeAttributes(
  existing?: ExecutionLogAttributes,
  next?: ExecutionLogAttributes
): ExecutionLogAttributes | undefined {
  if (!existing && !next) {
    return undefined;
  }
  return {
    ...(existing ?? {}),
    ...(next ?? {})
  };
}

export function normalizeExecutionLogEntry(entry: ExecutionLogEntry): ExecutionLogWireEntry {
  return {
    v: entry.v,
    ts: entry.ts,
    execution_id: entry.executionId,
    run_id: entry.runId,
    workflow_id: entry.workflowId,
    root_workflow_id: entry.rootWorkflowId,
    parent_execution_id: entry.parentExecutionId,
    session_id: entry.sessionId,
    actor_id: entry.actorId,
    agent_node_id: entry.agentNodeId,
    reasoner_id: entry.reasonerId,
    caller_did: entry.callerDid,
    target_did: entry.targetDid,
    agent_node_did: entry.agentNodeDid,
    level: entry.level,
    source: entry.source,
    event_type: entry.eventType,
    message: entry.message,
    attributes: entry.attributes,
    system_generated: entry.systemGenerated
  };
}

export function serializeExecutionLogEntry(entry: ExecutionLogEntry): string {
  return safeJsonStringify(normalizeExecutionLogEntry(entry));
}

export class ExecutionLogger {
  private readonly contextProvider?: () => ExecutionLogContext | undefined;
  private readonly transport?: ExecutionLogTransport;
  private readonly mirrorToStdout: boolean;
  private readonly stdout?: Pick<WriteStream, 'write'>;
  private readonly defaultSource: string;

  constructor(options: ExecutionLoggerOptions = {}) {
    this.contextProvider = options.contextProvider;
    this.transport = options.transport;
    this.mirrorToStdout = options.mirrorToStdout ?? true;
    this.stdout = options.stdout ?? (typeof process !== 'undefined' ? process.stdout : undefined);
    this.defaultSource = options.source ?? 'sdk.logger';
  }

  log(
    level: ExecutionLogLevel,
    message: string,
    attributes?: ExecutionLogAttributes,
    options: ExecutionLogEmitOptions = {}
  ): ExecutionLogEntry {
    const context = this.contextProvider?.() ?? {};
    const entry: ExecutionLogEntry = {
      v: 1,
      ts: new Date().toISOString(),
      level,
      source: options.source ?? this.defaultSource,
      message,
      ...context,
      ...(options.eventType ? { eventType: options.eventType } : {}),
      ...(options.systemGenerated ? { systemGenerated: true } : {}),
      ...(attributes ? { attributes: mergeAttributes(undefined, attributes) } : {})
    };

    this.emit(entry);
    return entry;
  }

  debug(
    message: string,
    attributes?: ExecutionLogAttributes,
    options?: ExecutionLogEmitOptions
  ): ExecutionLogEntry {
    return this.log('debug', message, attributes, options);
  }

  info(
    message: string,
    attributes?: ExecutionLogAttributes,
    options?: ExecutionLogEmitOptions
  ): ExecutionLogEntry {
    return this.log('info', message, attributes, options);
  }

  warn(
    message: string,
    attributes?: ExecutionLogAttributes,
    options?: ExecutionLogEmitOptions
  ): ExecutionLogEntry {
    return this.log('warn', message, attributes, options);
  }

  error(
    message: string,
    attributes?: ExecutionLogAttributes,
    options?: ExecutionLogEmitOptions
  ): ExecutionLogEntry {
    return this.log('error', message, attributes, options);
  }

  system(
    eventType: string,
    message: string,
    attributes?: ExecutionLogAttributes
  ): ExecutionLogEntry {
    return this.log('info', message, attributes, {
      eventType,
      source: 'sdk.runtime',
      systemGenerated: true
    });
  }

  private emit(entry: ExecutionLogEntry): void {
    const wire = normalizeExecutionLogEntry(entry);
    const line = safeJsonStringify(wire) + '\n';

    if (this.mirrorToStdout && this.stdout?.write) {
      this.stdout.write(line);
    }

    if (this.transport && wire.execution_id) {
      try {
        const result = this.transport.emit(wire);
        if (result && typeof (result as Promise<void>).catch === 'function') {
          void Promise.resolve(result).catch(() => {});
        }
      } catch {
        // Logging must never break execution flow.
      }
    }
  }
}

export function createExecutionLogger(options: ExecutionLoggerOptions = {}): ExecutionLogger {
  return new ExecutionLogger(options);
}
