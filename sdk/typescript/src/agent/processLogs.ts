type LogEntry = {
  v: number;
  seq: number;
  ts: string;
  stream: string;
  line: string;
  level?: string;
  source?: string;
  truncated?: boolean;
};

function logsEnabled(): boolean {
  const v = (process.env.AGENTFIELD_LOGS_ENABLED ?? 'true').trim().toLowerCase();
  return !['0', 'false', 'no', 'off'].includes(v);
}

function maxBufferBytes(): number {
  const raw = parseInt(process.env.AGENTFIELD_LOG_BUFFER_BYTES ?? '4194304', 10);
  return Number.isFinite(raw) && raw >= 1024 ? raw : 4194304;
}

function maxLineBytes(): number {
  const raw = parseInt(process.env.AGENTFIELD_LOG_MAX_LINE_BYTES ?? '16384', 10);
  return Number.isFinite(raw) && raw >= 256 ? raw : 16384;
}

function maxTailLines(): number {
  const raw = parseInt(process.env.AGENTFIELD_LOG_MAX_TAIL_LINES ?? '50000', 10);
  return Number.isFinite(raw) && raw >= 1 ? raw : 50000;
}

function internalBearerOk(authHeader: string | undefined): boolean {
  const want = (process.env.AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN ?? '').trim();
  if (!want) return true;
  if (!authHeader?.toLowerCase().startsWith('bearer ')) return false;
  return authHeader.slice(7).trim() === want;
}

export class ProcessLogRing {
  private seq = 0;
  private entries: LogEntry[] = [];
  private approxBytes = 0;
  private readonly maxBytes: number;

  constructor() {
    this.maxBytes = maxBufferBytes();
  }

  append(stream: string, line: string, truncated: boolean): void {
    const ts = new Date().toISOString();
    this.seq += 1;
    const sl = stream.toLowerCase();
    const level = sl === 'stderr' ? 'error' : sl === 'stdout' ? 'info' : 'log';
    const e: LogEntry = {
      v: 1,
      seq: this.seq,
      ts,
      stream,
      line,
      level,
      source: 'process',
      truncated,
    };
    this.entries.push(e);
    this.approxBytes += line.length + 64;
    while (this.approxBytes > this.maxBytes && this.entries.length > 1) {
      const old = this.entries.shift()!;
      this.approxBytes -= old.line.length + 64;
    }
  }

  tail(n: number): LogEntry[] {
    if (n <= 0) return [];
    return this.entries.length <= n ? [...this.entries] : this.entries.slice(-n);
  }

  snapshotAfter(sinceSeq: number, limit: number | null): LogEntry[] {
    const buf = this.entries.filter((e) => e.seq > sinceSeq);
    if (limit != null && limit > 0 && buf.length > limit) {
      return buf.slice(-limit);
    }
    return buf;
  }

}

let captureInstalled = false;

export function installStdioLogCapture(ring: ProcessLogRing): void {
  if (captureInstalled || !logsEnabled()) return;
  captureInstalled = true;
  const maxLB = maxLineBytes();

  const hook = (stream: NodeJS.WriteStream, name: string) => {
    const orig = stream.write.bind(stream);
    let buf = '';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (stream as any).write = (chunk: any, ...args: any[]) => {
      const s =
        typeof chunk === 'string'
          ? chunk
          : Buffer.isBuffer(chunk)
            ? chunk.toString('utf8')
            : String(chunk);
      buf += s;
      let idx: number;
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx);
        buf = buf.slice(idx + 1);
        let out = line;
        let trunc = false;
        if (Buffer.byteLength(out, 'utf8') > maxLB) {
          out = Buffer.from(out, 'utf8').subarray(0, maxLB).toString('utf8');
          trunc = true;
        }
        ring.append(name, out, trunc);
      }
      return orig(chunk, ...args);
    };
  };

  hook(process.stdout, 'stdout');
  hook(process.stderr, 'stderr');
}

export function registerAgentfieldLogsRoute(
  app: import('express').Express,
  ring: ProcessLogRing
): void {
  app.get('/agentfield/v1/logs', (req, res) => {
    if (!logsEnabled()) {
      return res.status(404).json({
        error: 'logs_disabled',
        message: 'Process logs API is disabled'
      });
    }
    const auth = req.headers.authorization;
    if (!internalBearerOk(auth)) {
      return res.status(401).json({
        error: 'unauthorized',
        message: 'Valid Authorization Bearer required'
      });
    }
    let tailLines = parseInt(String(req.query.tail_lines ?? '0'), 10);
    const sinceSeq = parseInt(String(req.query.since_seq ?? '0'), 10);
    const follow = ['1', 'true', 'yes'].includes(String(req.query.follow ?? '').toLowerCase());
    const cap = maxTailLines();
    if (tailLines > cap) {
      return res.status(413).json({
        error: 'tail_too_large',
        message: `tail_lines exceeds max ${cap}`
      });
    }
    if (tailLines <= 0 && sinceSeq <= 0 && !follow) tailLines = 200;

    let initial: LogEntry[];
    if (sinceSeq > 0) {
      initial = ring.snapshotAfter(sinceSeq, tailLines > 0 ? tailLines : null);
    } else {
      const n = tailLines > 0 ? tailLines : 200;
      initial = ring.tail(n);
    }

    res.setHeader('Content-Type', 'application/x-ndjson');
    res.setHeader('Cache-Control', 'no-store');
    res.status(200);

    const writeLines = (entries: LogEntry[]) => {
      for (const e of entries) {
        res.write(`${JSON.stringify(e)}\n`);
      }
      (res as import('express').Response & { flush?: () => void }).flush?.();
    };

    writeLines(initial);
    let lastSeq = sinceSeq;
    if (initial.length) lastSeq = initial[initial.length - 1]!.seq;

    if (!follow) {
      return res.end();
    }

    const iv = setInterval(() => {
      const newer = ring.snapshotAfter(lastSeq, null);
      if (newer.length) {
        writeLines(newer);
        lastSeq = newer[newer.length - 1]!.seq;
      }
    }, 400);

    req.on('close', () => {
      clearInterval(iv);
    });
  });
}
