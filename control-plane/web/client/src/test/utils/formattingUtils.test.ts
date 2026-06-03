import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  extractReasonerInputLayers,
  formatOutputUsageHint,
} from "@/utils/reasonerCompareExtract";
import {
  formatCompactDate,
  formatCompactRelativeTime,
  formatRelativeTime,
} from "@/utils/dateFormat";
import {
  formatWebhookStatusLabel,
  summarizeWorkflowWebhook,
} from "@/utils/webhook";

describe("formatting utilities", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-07T12:00:00Z"));
  });

  it("extracts reasoner prose and meta layers from input payloads", () => {
    expect(extractReasonerInputLayers(null)).toEqual({
      prose: [],
      meta: [],
      extractedKeys: new Set(),
    });

    const layers = extractReasonerInputLayers({
      prompt: "Summarize the run",
      description: { text: "Nested JSON" },
      model: "gpt-5.4",
      temperature: 0.2,
      workspace: "agentfield",
      ignored: "left in raw JSON",
    });

    expect(layers.prose).toEqual([
      { key: "prompt", label: "Prompt", text: "Summarize the run" },
      {
        key: "description",
        label: "Description",
        text: JSON.stringify({ text: "Nested JSON" }, null, 2),
      },
    ]);
    expect(layers.meta).toEqual([
      { key: "model", label: "Model", value: "gpt-5.4" },
      { key: "temperature", label: "Temp", value: "0.2" },
      { key: "workspace", label: "Workspace", value: "agentfield" },
    ]);
    expect(layers.extractedKeys.has("ignored")).toBe(false);
  });

  it("formats output usage hints from direct and nested usage payloads", () => {
    expect(formatOutputUsageHint(null)).toBeNull();
    expect(
      formatOutputUsageHint({
        usage: {
          total_tokens: 1500,
          prompt_tokens: 500,
          completion_tokens: 1000,
        },
      })
    ).toBe("1,500 tok (500 in / 1,000 out)");
    expect(
      formatOutputUsageHint({
        response: {
          metrics: {
            input_tokens: 100,
            output_tokens: 50,
          },
        },
      })
    ).toBe("150 tok (100 in / 50 out)");
  });

  it("formats relative and compact dates", () => {
    expect(formatRelativeTime("0001-01-01T00:00:00Z")).toBe("—");
    expect(formatRelativeTime("2026-04-07T11:59:50Z")).toBe("< 1 min ago");
    expect(formatRelativeTime("2026-04-07T11:55:00Z")).toBe("5 mins ago");
    expect(formatRelativeTime("2026-04-07T10:00:00Z")).toBe("2 hours ago");
    expect(formatRelativeTime("2026-04-06T06:00:00Z")).toBe(
      `Yesterday, ${new Date("2026-04-06T06:00:00Z").toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      })}`
    );
    expect(formatRelativeTime("2026-04-02T12:00:00Z")).toBe(
      new Date("2026-04-02T12:00:00Z").toLocaleDateString("en-US", {
        weekday: "short",
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      })
    );
    expect(formatRelativeTime("2025-12-31T12:00:00Z")).toBe(
      new Date("2025-12-31T12:00:00Z").toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    );

    expect(formatCompactRelativeTime(undefined)).toBe("—");
    expect(formatCompactRelativeTime("invalid")).toBe("—");
    expect(formatCompactRelativeTime("2026-04-07T12:00:05Z")).toBe("now");
    expect(formatCompactRelativeTime("2026-04-07T11:59:58Z")).toBe("now");
    expect(formatCompactRelativeTime("2026-04-07T11:59:40Z")).toBe("20s ago");
    expect(formatCompactRelativeTime("2026-04-07T11:45:00Z")).toBe("15m ago");
    expect(formatCompactRelativeTime("2026-04-07T08:00:00Z")).toBe("4h ago");
    expect(formatCompactRelativeTime("2026-04-01T12:00:00Z")).toBe("6d ago");
    expect(formatCompactRelativeTime("2024-04-01T12:00:00Z")).toBe(">1y ago");

    expect(formatCompactDate("2026-04-07T09:30:00Z")).toBe(
      new Date("2026-04-07T09:30:00Z").toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      })
    );
    expect(formatCompactDate("2025-04-07T09:30:00Z")).toBe(
      new Date("2025-04-07T09:30:00Z").toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    );
  });

  it("summarizes webhook activity and normalizes labels", () => {
    expect(summarizeWorkflowWebhook()).toEqual({
      nodesWithWebhook: 0,
      pendingNodes: 0,
      totalDeliveries: 0,
      successDeliveries: 0,
      failedDeliveries: 0,
    });

    expect(
      summarizeWorkflowWebhook([
        {
          workflow_id: "wf-1",
          execution_id: "exec-1",
          agent_node_id: "node-1",
          reasoner_id: "reasoner-1",
          status: "running",
          started_at: "2026-04-07T11:00:00Z",
          workflow_depth: 0,
          webhook_registered: true,
        },
        {
          workflow_id: "wf-2",
          execution_id: "exec-2",
          agent_node_id: "node-2",
          reasoner_id: "reasoner-2",
          status: "failed",
          started_at: "2026-04-07T10:00:00Z",
          workflow_depth: 1,
          webhook_registered: true,
          webhook_event_count: 3,
          webhook_success_count: 2,
          webhook_failure_count: 1,
          webhook_last_status: "failed",
          webhook_last_sent_at: "2026-04-07T11:30:00Z",
          webhook_last_error: "bad gateway",
          webhook_last_http_status: 502,
        },
      ])
    ).toEqual({
      nodesWithWebhook: 2,
      pendingNodes: 1,
      totalDeliveries: 3,
      successDeliveries: 2,
      failedDeliveries: 1,
      lastStatus: "failed",
      lastSentAt: "2026-04-07T11:30:00Z",
      lastError: "bad gateway",
      lastHttpStatus: 502,
    });

    expect(formatWebhookStatusLabel(undefined)).toBe("registered");
    expect(formatWebhookStatusLabel("Succeeded")).toBe("delivered");
    expect(formatWebhookStatusLabel("queued")).toBe("pending");
    expect(formatWebhookStatusLabel("other")).toBe("other");
  });
});
