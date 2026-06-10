const DEFAULT_OPENROUTER_SITE_URL = 'https://agentfield.ai';
const DEFAULT_OPENROUTER_APP_NAME = 'AgentField AI';

type EnvMap = Record<string, string | undefined>;

export interface OpenRouterAttributionOptions {
  siteUrl?: string;
  appName?: string;
  headers?: Record<string, string | undefined>;
  env?: EnvMap;
}

export function isOpenRouterRequest(options: {
  provider?: string;
  model?: string;
  baseUrl?: string;
}): boolean {
  const provider = clean(options.provider)?.toLowerCase();
  if (provider === 'openrouter') {
    return true;
  }

  const model = clean(options.model)?.toLowerCase();
  if (model?.startsWith('openrouter/')) {
    return true;
  }

  const baseUrl = clean(options.baseUrl)?.toLowerCase();
  return Boolean(baseUrl?.includes('openrouter.ai'));
}

export function openRouterAttributionEnabled(env: EnvMap = process.env): boolean {
  const raw = clean(env.AGENTFIELD_OPENROUTER_ATTRIBUTION);
  if (!raw) {
    return true;
  }
  return !['0', 'false', 'no', 'off'].includes(raw.toLowerCase());
}

export function resolveOpenRouterAttribution(
  options: OpenRouterAttributionOptions = {}
): { siteUrl: string; appName: string } | undefined {
  const env = options.env ?? process.env;
  if (!openRouterAttributionEnabled(env)) {
    return undefined;
  }

  const siteUrl =
    clean(options.siteUrl) ??
    clean(env.AGENTFIELD_OPENROUTER_SITE_URL) ??
    clean(env.OR_SITE_URL) ??
    DEFAULT_OPENROUTER_SITE_URL;

  const appName =
    clean(options.appName) ??
    clean(env.AGENTFIELD_OPENROUTER_APP_NAME) ??
    clean(env.OR_APP_NAME) ??
    DEFAULT_OPENROUTER_APP_NAME;

  return { siteUrl, appName };
}

export function openRouterAttributionHeaders(
  options: OpenRouterAttributionOptions = {}
): Record<string, string> {
  const resolved = resolveOpenRouterAttribution(options);
  if (!resolved) {
    return {};
  }
  return {
    'HTTP-Referer': resolved.siteUrl,
    'X-OpenRouter-Title': resolved.appName,
    'X-Title': resolved.appName,
  };
}

export function mergeOpenRouterAttributionHeaders(
  existing: Record<string, string | undefined> = {},
  options: OpenRouterAttributionOptions = {}
): Record<string, string> {
  const merged: Record<string, string> = {};
  const lowerKeys = new Set<string>();

  for (const [key, value] of Object.entries(existing)) {
    const cleaned = clean(value);
    if (!cleaned) {
      continue;
    }
    merged[key] = cleaned;
    lowerKeys.add(key.toLowerCase());
  }

  for (const [key, value] of Object.entries(openRouterAttributionHeaders(options))) {
    if (!lowerKeys.has(key.toLowerCase())) {
      merged[key] = value;
    }
  }

  return merged;
}

export function openRouterAttributionEnv(env: EnvMap = process.env): Record<string, string> {
  const resolved = resolveOpenRouterAttribution({ env });
  if (!resolved) {
    return {};
  }
  return {
    AGENTFIELD_OPENROUTER_SITE_URL: resolved.siteUrl,
    AGENTFIELD_OPENROUTER_APP_NAME: resolved.appName,
    OR_SITE_URL: resolved.siteUrl,
    OR_APP_NAME: resolved.appName,
  };
}

export function applyOpenRouterAttributionEnv(env: Record<string, string | undefined>): void {
  if (!openRouterAttributionEnabled(env)) {
    for (const key of [
      'AGENTFIELD_OPENROUTER_SITE_URL',
      'AGENTFIELD_OPENROUTER_APP_NAME',
      'OR_SITE_URL',
      'OR_APP_NAME',
    ]) {
      delete env[key];
    }
    return;
  }

  const attribution = openRouterAttributionEnv(env);
  for (const [key, value] of Object.entries(attribution)) {
    if (clean(env[key]) == null) {
      env[key] = value;
    }
  }
}

function clean(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}
