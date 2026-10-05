/** Classify only the API route; never retain a private endpoint in telemetry. */
export function routingProvider(provider?: string | null, endpoint?: string | null): string {
  if (endpoint !== undefined && endpoint !== null) {
    try {
      const url = new URL(endpoint);
      if (!['https:', 'http:'].includes(url.protocol)) return 'unknown';
      const host = url.hostname.toLowerCase();
      if (host === 'openrouter.ai') return 'openrouter';
      if (host === 'api.openai.com') return 'openai';
      if (host === 'api.anthropic.com') return 'anthropic';
      if (host === 'generativelanguage.googleapis.com') return 'google';
      return 'other';
    } catch { return 'unknown'; }
  }
  const normalized = provider === 'gemini' ? 'google' : provider?.toLowerCase();
  return ['openrouter', 'openai', 'anthropic', 'google', 'bedrock', 'azure', 'ollama'].includes(normalized ?? '')
    ? normalized! : 'unknown';
}
