/**
 * Pull human-facing fields out of reasoner/harness `input_data` so compare / step
 * UIs can show prose + badges first and full JSON second.
 */

/** Long-text fields we surface above the raw JSON (first match wins per key). */
export const REASONER_PROSE_KEY_ORDER = [
  "start_tip",
  "startTip",
  "goal",
  "prompt",
  "task",
  "instruction",
  "instructions",
  "objective",
  "user_message",
  "userMessage",
  "message",
  "query",
  "question",
  "description",
  "context",
  "brief",
] as const;

/** Short scalar fields shown as badges / metadata chips. */
export const REASONER_META_KEY_ORDER = [
  "ai_provider",
  "provider",
  "model",
  "model_id",
  "modelId",
  "temperature",
  "max_tokens",
  "maxTokens",
  "artifacts_dir",
  "artifactsDir",
  "workspace",
  "session_hint",
  "sessionHint",
] as const;

const PROSE_LABELS: Record<string, string> = {
  start_tip: "Start tip",
  startTip: "Start tip",
  goal: "Goal",
  prompt: "Prompt",
  task: "Task",
  instruction: "Instruction",
  instructions: "Instructions",
  objective: "Objective",
  user_message: "User message",
  userMessage: "User message",
  message: "Message",
  query: "Query",
  question: "Question",
  description: "Description",
  context: "Context",
  brief: "Brief",
};

const META_LABELS: Record<string, string> = {
  ai_provider: "Provider",
  provider: "Provider",
  model: "Model",
  model_id: "Model",
  modelId: "Model",
  temperature: "Temp",
  max_tokens: "Max tokens",
  maxTokens: "Max tokens",
  artifacts_dir: "Artifacts",
  artifactsDir: "Artifacts",
  workspace: "Workspace",
  session_hint: "Session",
  sessionHint: "Session",
};

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function formatScalar(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") {
    const t = v.trim();
    return t.length ? t : null;
  }
  if (typeof v === "number" && Number.isFinite(v)) return String(v);
  if (typeof v === "boolean") return v ? "true" : "false";
  return null;
}

export type ProseField = { key: string; label: string; text: string };
export type MetaChip = { key: string; label: string; value: string };

export function extractReasonerInputLayers(input: unknown): {
  prose: ProseField[];
  meta: MetaChip[];
  extractedKeys: Set<string>;
} {
  if (!isPlainObject(input)) {
    return { prose: [], meta: [], extractedKeys: new Set() };
  }

  const extractedKeys = new Set<string>();
  const prose: ProseField[] = [];

  for (const key of REASONER_PROSE_KEY_ORDER) {
    if (!(key in input)) continue;
    const raw = input[key];
    let text: string | null = null;
    if (typeof raw === "string") {
      text = raw.trim() || null;
    } else if (isPlainObject(raw) || Array.isArray(raw)) {
      try {
        text = JSON.stringify(raw, null, 2);
      } catch {
        text = String(raw);
      }
    } else {
      text = formatScalar(raw);
    }
    if (text) {
      prose.push({
        key,
        label: PROSE_LABELS[key] ?? key.replace(/_/g, " "),
        text,
      });
      extractedKeys.add(key);
    }
  }

  const meta: MetaChip[] = [];
  for (const key of REASONER_META_KEY_ORDER) {
    if (!(key in input)) continue;
    const s = formatScalar(input[key]);
    if (s) {
      meta.push({
        key,
        label: META_LABELS[key] ?? key.replace(/_/g, " "),
        value: s,
      });
      extractedKeys.add(key);
    }
  }

  return { prose, meta, extractedKeys };
}

/** Best-effort token / usage string from common SDK shapes on output. */
export function formatOutputUsageHint(output: unknown): string | null {
  if (!isPlainObject(output)) return null;

  const tryUsage = (u: unknown): string | null => {
    if (!isPlainObject(u)) return null;
    const total: number | undefined =
      typeof u.total_tokens === "number" ? u.total_tokens :
      typeof u.total === "number" ? u.total :
      typeof u.tokens === "number" ? u.tokens :
      undefined;
    const inT =
      typeof u.prompt_tokens === "number"
        ? u.prompt_tokens
        : typeof u.input_tokens === "number"
          ? u.input_tokens
          : undefined;
    const outT =
      typeof u.completion_tokens === "number"
        ? u.completion_tokens
        : typeof u.output_tokens === "number"
          ? u.output_tokens
          : undefined;

    if (total != null && total > 0) {
      if (inT != null && outT != null) {
        return `${total.toLocaleString()} tok (${inT.toLocaleString()} in / ${outT.toLocaleString()} out)`;
      }
      return `${total.toLocaleString()} tokens`;
    }
    if (inT != null && outT != null) {
      return `${(inT + outT).toLocaleString()} tok (${inT.toLocaleString()} in / ${outT.toLocaleString()} out)`;
    }
    return null;
  };

  const direct =
    tryUsage(output.usage) ||
    tryUsage(output.token_usage) ||
    tryUsage(output.metrics);
  if (direct) return direct;

  const nested = output.result ?? output.data ?? output.response;
  if (isPlainObject(nested)) {
    return (
      tryUsage(nested.usage) ||
      tryUsage(nested.token_usage) ||
      tryUsage(nested.metrics) ||
      null
    );
  }
  return null;
}
