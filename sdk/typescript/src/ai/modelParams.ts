// Models whose default mode rejects a custom temperature. Mirrors
// rejectsTemperature in the Go SDK (sdk/go/ai/model_params.go) and
// _model_rejects_temperature in the Python SDK. GPT-5.1/5.2/5.4 accept
// temperature, so this is not every gpt-5* id. This SDK does not expose
// reasoning_effort.

const TEMPERATURE_REJECTING_FAMILIES = [
  'gpt-5-mini',
  'gpt-5-nano',
  'gpt-5-pro',
  'gpt-5.2-pro',
  'gpt-5.4-pro',
  'gpt-5.5',
  'gpt-5.6',
  'gpt-5-codex',
  'gpt-5.1-codex',
  'gpt-5.2-codex',
  'gpt-5.3-codex',
  'claude-opus-4-7',
  'claude-opus-4-8',
  'claude-opus-5',
  'claude-sonnet-5',
  'claude-haiku-5',
  'claude-fable-5'
] as const;

export function rejectsTemperature(model: string | undefined | null): boolean {
  if (!model) return false;
  let id = model.trim().toLowerCase();
  const slash = id.lastIndexOf('/');
  if (slash >= 0) id = id.slice(slash + 1);
  if (id.length >= 2 && id[0] === 'o' && id[1] >= '0' && id[1] <= '9') {
    return true;
  }
  if (id === 'gpt-5' || id.startsWith('gpt-5-202')) {
    return true;
  }
  for (const family of TEMPERATURE_REJECTING_FAMILIES) {
    if (id === family || id.startsWith(`${family}-`)) return true;
  }
  return false;
}

/** Temperature to forward, or undefined when the model rejects a custom value. */
export function temperatureForModel(
  model: string | undefined | null,
  temperature: number | undefined
): number | undefined {
  if (temperature === undefined || rejectsTemperature(model)) return undefined;
  return temperature;
}
