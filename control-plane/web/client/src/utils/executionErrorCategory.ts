export type ExecutionErrorCategory =
  | "llm_unavailable"
  | "concurrency_limit"
  | "agent_timeout"
  | "agent_unreachable"
  | "agent_error"
  | "bad_response"
  | "internal_error";

export interface ExecutionErrorCategoryMeta {
  category: string;
  label: string;
  description: string;
  badgeClassName: string;
  diagnosticsLabel?: string;
  diagnosticsPath?: string;
}

const ERROR_CATEGORY_META: Record<ExecutionErrorCategory, Omit<ExecutionErrorCategoryMeta, "category">> = {
  llm_unavailable: {
    label: "LLM unavailable",
    description: "LLM backend circuit breaker is open.",
    badgeClassName: "bg-red-500/10 text-red-700 border-red-500/30 dark:text-red-300",
    diagnosticsLabel: "LLM health",
    diagnosticsPath: "/dashboard",
  },
  concurrency_limit: {
    label: "Concurrency limit",
    description: "Agent is at max concurrent executions.",
    badgeClassName: "bg-yellow-500/10 text-yellow-700 border-yellow-500/30 dark:text-yellow-300",
    diagnosticsLabel: "Queue status",
    diagnosticsPath: "/dashboard",
  },
  agent_timeout: {
    label: "Agent timeout",
    description: "Agent did not respond before timeout.",
    badgeClassName: "bg-orange-500/10 text-orange-700 border-orange-500/30 dark:text-orange-300",
  },
  agent_unreachable: {
    label: "Agent unreachable",
    description: "Agent appears offline or unreachable.",
    badgeClassName: "bg-red-500/10 text-red-700 border-red-500/30 dark:text-red-300",
    diagnosticsLabel: "Node status",
    diagnosticsPath: "/agents",
  },
  agent_error: {
    label: "Agent error",
    description: "Agent returned an application error.",
    badgeClassName: "bg-red-500/10 text-red-700 border-red-500/30 dark:text-red-300",
  },
  bad_response: {
    label: "Bad response",
    description: "Agent returned an invalid response payload.",
    badgeClassName: "bg-red-500/10 text-red-700 border-red-500/30 dark:text-red-300",
  },
  internal_error: {
    label: "Internal error",
    description: "Control plane encountered an internal error.",
    badgeClassName: "bg-red-500/10 text-red-700 border-red-500/30 dark:text-red-300",
  },
};

export function getExecutionErrorCategoryMeta(
  category?: string | null,
): ExecutionErrorCategoryMeta | null {
  if (!category) {
    return null;
  }
  const normalized = category.trim().toLowerCase() as ExecutionErrorCategory;
  const known = ERROR_CATEGORY_META[normalized];
  if (!known) {
    return {
      category: normalized,
      label: normalized.replace(/_/g, " "),
      description: "Execution failed with an uncategorized error.",
      badgeClassName: "bg-red-500/10 text-red-700 border-red-500/30 dark:text-red-300",
    };
  }
  return { category: normalized, ...known };
}
