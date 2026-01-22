import WebSocket from 'ws';
import type { MemoryChangeEvent } from './MemoryInterface.js';

export type MemoryEventHandler = (event: MemoryChangeEvent) => Promise<void> | void;

export class MemoryEventClient {
  private readonly url: string;
  private ws?: WebSocket;
  private handlers: MemoryEventHandler[] = [];
  private reconnectDelay = 1000;
  private closed = false;
  private reconnectPending = false;
  private reconnectTimer?: ReturnType<typeof setTimeout>;
  private readonly headers: Record<string, string>;

  constructor(baseUrl: string, headers?: Record<string, string | number | boolean | undefined>) {
    this.url = `${baseUrl.replace(/^http/, 'ws')}/api/v1/memory/events/ws`;
    this.headers = this.buildForwardHeaders(headers ?? {});
  }

  start() {
    if (this.ws) return;
    this.connect();
  }

  onEvent(handler: MemoryEventHandler) {
    this.handlers.push(handler);
  }

  stop() {
    this.closed = true;
    this.cleanup();
  }

  private cleanup() {
    // Clear any pending reconnect timer
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = undefined;
    }
    if (this.ws) {
      // Remove all listeners to prevent reconnect triggers during cleanup
      this.ws.removeAllListeners();
      // Terminate forcefully to ensure socket is closed
      this.ws.terminate();
      this.ws = undefined;
    }
  }

  private connect() {
    // Clean up any existing connection first
    this.cleanup();
    this.reconnectPending = false;

    this.ws = new WebSocket(this.url, { headers: this.headers });

    this.ws.on('open', () => {
      this.reconnectDelay = 1000;
    });

    this.ws.on('message', async (raw) => {
      try {
        const parsed = JSON.parse(raw.toString()) as MemoryChangeEvent;
        for (const handler of this.handlers) {
          await handler(parsed);
        }
      } catch (err) {
        // swallow parsing errors to keep connection alive
        console.error('Failed to handle memory event', err);
      }
    });

    // Use a single handler for both close and error to prevent duplicate reconnects
    const handleDisconnect = () => this.scheduleReconnect();
    this.ws.on('close', handleDisconnect);
    this.ws.on('error', handleDisconnect);
  }

  private scheduleReconnect() {
    // Prevent duplicate reconnect scheduling
    if (this.closed || this.reconnectPending) return;
    this.reconnectPending = true;

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined;
      if (this.closed) return;
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
      this.connect();
    }, this.reconnectDelay);
  }

  private buildForwardHeaders(headers: Record<string, any>): Record<string, string> {
    const allowed = new Set(['authorization', 'cookie']);
    const sanitized: Record<string, string> = {};
    Object.entries(headers).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      const lower = key.toLowerCase();
      if (lower.startsWith('x-') || allowed.has(lower)) {
        sanitized[key] = typeof value === 'string' ? value : String(value);
      }
    });
    return sanitized;
  }
}
