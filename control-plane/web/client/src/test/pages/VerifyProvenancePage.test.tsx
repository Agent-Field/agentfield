// @ts-nocheck
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";

import { VerifyProvenancePage } from "@/pages/VerifyProvenancePage";

const state = vi.hoisted(() => ({
  verifyProvenanceAudit: vi.fn<(payload: unknown) => Promise<any>>(),
}));

vi.mock("@/services/vcApi", () => ({
  verifyProvenanceAudit: (payload: unknown) => state.verifyProvenanceAudit(payload),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/label", () => ({
  Label: ({
    children,
    ...props
  }: React.PropsWithChildren<React.LabelHTMLAttributes<HTMLLabelElement>>) => (
    <label {...props}>{children}</label>
  ),
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/separator", () => ({
  Separator: () => <div>separator</div>,
}));

vi.mock("@/components/ui/table", () => ({
  Table: ({ children }: React.PropsWithChildren) => <table>{children}</table>,
  TableHeader: ({ children }: React.PropsWithChildren) => <thead>{children}</thead>,
  TableBody: ({ children }: React.PropsWithChildren) => <tbody>{children}</tbody>,
  TableRow: ({ children }: React.PropsWithChildren) => <tr>{children}</tr>,
  TableHead: ({ children }: React.PropsWithChildren) => <th>{children}</th>,
  TableCell: ({ children }: React.PropsWithChildren) => <td>{children}</td>,
}));

vi.mock("@/components/ui/hover-card", () => ({
  HoverCard: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  HoverCardTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  HoverCardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("lucide-react", () => {
  const Icon = () => <span>icon</span>;
  return {
    Upload: Icon,
    FileCheck2: Icon,
    AlertTriangle: Icon,
    Info: Icon,
  };
});

function renderPage() {
  return render(
    <MemoryRouter>
      <VerifyProvenancePage />
    </MemoryRouter>
  );
}

describe("VerifyProvenancePage", () => {
  beforeEach(() => {
    state.verifyProvenanceAudit.mockReset();
  });

  it("renders loading while verifying and then shows populated audit results", async () => {
    let resolveAudit: ((value: any) => void) | null = null;
    state.verifyProvenanceAudit.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveAudit = resolve;
        })
    );

    renderPage();

    fireEvent.change(screen.getByLabelText("JSON"), {
      target: { value: '{"workflow_id":"wf-1"}' },
    });
    fireEvent.click(screen.getByRole("button", { name: /run audit/i }));

    await waitFor(() => {
      expect(state.verifyProvenanceAudit).toHaveBeenCalledWith({ workflow_id: "wf-1" });
    });
    expect(screen.getAllByText("Running audit…").length).toBeGreaterThan(0);

    resolveAudit?.({
      valid: true,
      type: "workflow_bundle",
      message: "verified",
      workflow_id: "wf-1",
      component_results: [
        {
          execution_id: "execution-12345678",
          vc_id: "vc-1",
          valid: true,
          signature_valid: true,
        },
      ],
      did_resolutions: [
        {
          did: "did:web:test",
          success: true,
          resolved_from: "bundle",
        },
      ],
      comprehensive: {
        overall_score: 99.5,
        security_analysis: { tamper_evidence: [] },
        critical_issues: [],
      },
    });

    expect(await screen.findByText("Audit passed")).toBeInTheDocument();
    expect(screen.getByText("99.5 / 100")).toBeInTheDocument();
    expect(screen.getByText("workflow_bundle")).toBeInTheDocument();
    expect(screen.getByText("…12345678")).toBeInTheDocument();
    expect(screen.getByText("did:web:test")).toBeInTheDocument();
  });
});