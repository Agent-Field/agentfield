const ERROR_MESSAGE_KEYS = ["message", "detail", "error", "reason", "msg"] as const;
const MAX_ERROR_DEPTH = 4;

function messageFromValue(value: unknown, depth = 0): string | null {
  if (depth > MAX_ERROR_DEPTH || value == null) return null;

  if (typeof value === "string") {
    const message = value.trim();
    return message || null;
  }

  if (value instanceof Error) return messageFromValue(value.message, depth + 1);

  if (Array.isArray(value)) {
    const messages = value
      .map((entry) => messageFromValue(entry, depth + 1))
      .filter((entry): entry is string => Boolean(entry));
    return messages.length > 0 ? messages.join("; ") : null;
  }

  if (typeof value !== "object") return String(value);

  const record = value as Record<string, unknown>;
  for (const key of ERROR_MESSAGE_KEYS) {
    const message = messageFromValue(record[key], depth + 1);
    if (message) return message;
  }

  return null;
}

export function getErrorMessage(value: unknown, fallback: string): string {
  return messageFromValue(value) ?? fallback;
}
