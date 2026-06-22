import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { NewTriggerDialog } from "@/components/triggers/NewTriggerDialog";
import { SourceIcon } from "@/components/triggers/SourceIcon";

describe("Databricks trigger UI", () => {
  it("renders the Databricks source icon glyph", () => {
    const { container } = render(<SourceIcon source="databricks" />);

    expect(container.querySelector("svg")).not.toBeNull();
    expect(container.querySelectorAll("path").length).toBeGreaterThanOrEqual(1);
  });

  it("shows Databricks trigger guidance and defaults", () => {
    render(
      <NewTriggerDialog
        open
        onOpenChange={vi.fn()}
        onCreated={vi.fn()}
        defaultSourceName="databricks"
        sources={[
          {
            name: "databricks",
            kind: "http",
            secret_required: true,
            config_schema: {
              type: "object",
              properties: {
                auth_mode: { type: "string" },
              },
            },
          },
        ]}
      />,
    );

    expect(
      screen.getByText(
        "Bind a Databricks notification destination to a reasoner. The control plane verifies the webhook secret and dispatches each normalized Databricks event to the selected node.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("handle_databricks_event"),
    ).toBeInTheDocument();
    expect(screen.getByPlaceholderText("TERMINATED, FAILED")).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("DATABRICKS_TRIGGER_SECRET"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Use this value as the Databricks notification destination basic-auth password or bearer token.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByDisplayValue(/"mode": "webhook_notification"/)).toBeInTheDocument();
  });
});
