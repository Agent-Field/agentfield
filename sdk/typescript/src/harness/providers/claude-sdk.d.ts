declare module '@anthropic-ai/claude-agent-sdk' {
  export function query(input: {
    prompt: string;
    options: Record<string, unknown>;
  }): AsyncIterable<unknown>;
}
