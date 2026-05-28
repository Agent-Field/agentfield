import type { WorkflowTimelineNode } from "../types/workflows";

export interface WorkflowWebhookSummary {
  nodesWithWebhook: number;
  pendingNodes: number;
  totalDeliveries: number;
  successDeliveries: number;
  failedDeliveries: number;
  lastStatus?: string;
  lastSentAt?: string;
  lastError?: string;
  lastHttpStatus?: number;
}

const EMPTY_SUMMARY: WorkflowWebhookSummary = {
  nodesWithWebhook: 0,
  pendingNodes: 0,
  totalDeliveries: 0,
  successDeliveries: 0,
  failedDeliveries: 0,
};

export function summarizeWorkflowWebhook(
  timeline?: WorkflowTimelineNode[] | null,
): WorkflowWebhookSummary {
  if (!timeline || timeline.length === 0) {
    return { ...EMPTY_SUMMARY };
  }

  let latestTimestamp = 0;
  const summary: WorkflowWebhookSummary = { ...EMPTY_SUMMARY };

  for (const node of timeline) {
    const eventCount = node.webhook_event_count ?? 0;
    const successCount = node.webhook_success_count ?? 0;
    const failureCount = node.webhook_failure_count ?? 0;
    const hasWebhook = Boolean(
      node.webhook_registered ||
        eventCount > 0 ||
        successCount > 0 ||
        failureCount > 0,
    );

    if (!hasWebhook) {
      continue;
    }

    summary.nodesWithWebhook += 1;
    summary.totalDeliveries += eventCount;
    summary.successDeliveries += successCount;
    summary.failedDeliveries += failureCount;

    if (
      node.webhook_registered &&
      eventCount === 0 &&
      successCount === 0 &&
      failureCount === 0
    ) {
      summary.pendingNodes += 1;
    }

    if (node.webhook_last_sent_at) {
      const timestamp = Date.parse(node.webhook_last_sent_at);
      if (!Number.isNaN(timestamp) && timestamp >= latestTimestamp) {
        latestTimestamp = timestamp;
        summary.lastStatus = node.webhook_last_status;
        summary.lastSentAt = node.webhook_last_sent_at;
        summary.lastError = node.webhook_last_error;
        summary.lastHttpStatus = node.webhook_last_http_status;
      }
    }
  }

  return summary;
}

export function formatWebhookStatusLabel(status?: string | null): string {
  if (!status) return "registered";
  const normalized = status.toLowerCase();
  switch (normalized) {
    case "succeeded":
    case "delivered":
    case "success":
      return "delivered";
    case "failed":
      return "failed";
    case "pending":
    case "queued":
      return "pending";
    default:
      return normalized;
  }
}
