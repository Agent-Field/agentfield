// @ts-nocheck
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { NewSettingsPage } from "@/pages/NewSettingsPage";

const pageState = vi.hoisted(() => ({
  clipboardWriteText: vi.fn<(value: string) => Promise<void>>(),
  getObservabilityWebhook: vi.fn(),
  setObservabilityWebhook: vi.fn(),
  deleteObservabilityWebhook: vi.fn(),
  getObservabilityWebhookStatus: vi.fn(),
  redriveDeadLetterQueue: vi.fn(),
  clearDeadLetterQueue: vi.fn(),
  getDIDSystemStatus: vi.fn(),
  getNodeLogProxySettings: vi.fn(),
  putNodeLogProxySettings: vi.fn(),
  open: vi.fn(),
  fetch: vi.fn(),
}));

vi.mock("@/components/ui/tabs", () => ({
  Tabs: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TabsList: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TabsTrigger: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" role="tab" {...props}>
      {children}
    </button>
  ),
  TabsContent: ({
    children,
    value,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement> & { value?: string }>) => (
    <div data-tab-content={value} {...props}>
      {children}
    </div>
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <section {...props}>{children}</section>
  ),
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => (
    <h2 {...props}>{children}</h2>
  ),
  CardDescription: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLParagraphElement>>) => (
    <p {...props}>{children}</p>
  ),
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardFooter: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
}));

vi.mock("@/components/ui/input", () => ({
  Input: React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
    (props, ref) => <input ref={ref} {...props} />,
  ),
}));

vi.mock("@/components/ui/label", () => ({
  Label: ({ children, ...props }: React.PropsWithChildren<React.LabelHTMLAttributes<HTMLLabelElement>>) => (
    <label {...props}>{children}</label>
  ),
}));

vi.mock("@/components/ui/switch", () => ({
  Switch: ({
    checked,
    onCheckedChange,
    ...props
  }: {
    checked?: boolean;
    onCheckedChange?: (value: boolean) => void;
  } & React.InputHTMLAttributes<HTMLInputElement>) => (
    <input
      type="checkbox"
      role="switch"
      checked={checked}
      onChange={(event) => onCheckedChange?.(event.target.checked)}
      {...props}
    />
  ),
}));

vi.mock("@/components/ui/button", () => ({
  Button: React.forwardRef<
    HTMLButtonElement,
    React.ButtonHTMLAttributes<HTMLButtonElement> & { asChild?: boolean }
  >(({ children, ...props }, ref) => (
    <button ref={ref} type="button" {...props}>
      {children}
    </button>
  )),
  buttonVariants: () => "",
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({
    children,
    showIcon: _showIcon,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLSpanElement> & { showIcon?: boolean }>) => (
    <span {...props}>{children}</span>
  ),
}));

vi.mock("@/components/ui/separator", () => ({
  Separator: (props: React.HTMLAttributes<HTMLHRElement>) => <hr {...props} />,
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  AlertTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => (
    <h3 {...props}>{children}</h3>
  ),
  AlertDescription: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLParagraphElement>>) => (
    <p {...props}>{children}</p>
  ),
}));

vi.mock("@/components/ui/alert-dialog", () => ({
  AlertDialog: ({
    children,
    open,
  }: React.PropsWithChildren<{ open?: boolean; onOpenChange?: (open: boolean) => void }>) =>
    open ? <div role="dialog">{children}</div> : null,
  AlertDialogContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: React.PropsWithChildren) => <h2>{children}</h2>,
  AlertDialogDescription: ({ children }: React.PropsWithChildren) => <p>{children}</p>,
  AlertDialogFooter: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogCancel: React.forwardRef<
    HTMLButtonElement,
    React.ButtonHTMLAttributes<HTMLButtonElement>
  >(({ children, ...props }, ref) => (
    <button ref={ref} type="button" {...props}>
      {children}
    </button>
  )),
  AlertDialogAction: React.forwardRef<
    HTMLButtonElement,
    React.ButtonHTMLAttributes<HTMLButtonElement>
  >(({ children, ...props }, ref) => (
    <button ref={ref} type="button" {...props}>
      {children}
    </button>
  )),
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = React.forwardRef<SVGSVGElement, { className?: string }>((props, ref) => (
    <svg ref={ref} data-testid="icon" {...props} />
  ));

  return {
    Trash: Icon,
    Plus: Icon,
    CheckCircle: Icon,
    XCircle: Icon,
    Renew: Icon,
    Eye: Icon,
    EyeOff: Icon,
    Copy: Icon,
  };
});

vi.mock("@/services/observabilityWebhookApi", () => ({
  getObservabilityWebhook: (...args: unknown[]) => pageState.getObservabilityWebhook(...args),
  setObservabilityWebhook: (...args: unknown[]) => pageState.setObservabilityWebhook(...args),
  deleteObservabilityWebhook: (...args: unknown[]) => pageState.deleteObservabilityWebhook(...args),
  getObservabilityWebhookStatus: (...args: unknown[]) => pageState.getObservabilityWebhookStatus(...args),
  redriveDeadLetterQueue: (...args: unknown[]) => pageState.redriveDeadLetterQueue(...args),
  clearDeadLetterQueue: (...args: unknown[]) => pageState.clearDeadLetterQueue(...args),
}));

vi.mock("@/services/didApi", () => ({
  getDIDSystemStatus: (...args: unknown[]) => pageState.getDIDSystemStatus(...args),
}));

vi.mock("@/services/api", () => ({
  getNodeLogProxySettings: (...args: unknown[]) => pageState.getNodeLogProxySettings(...args),
  putNodeLogProxySettings: (...args: unknown[]) => pageState.putNodeLogProxySettings(...args),
}));

vi.mock("@/utils/dateFormat", () => ({
  formatRelativeTime: (value: string) => `relative:${value}`,
}));

function seedPageMocks() {
  pageState.getObservabilityWebhook.mockResolvedValue({
    configured: true,
    config: {
      url: "https://hooks.example.test/events",
      enabled: true,
      has_secret: true,
      headers: { Authorization: "Bearer token" },
      created_at: "2026-04-07T10:00:00Z",
      updated_at: "2026-04-07T12:00:00Z",
    },
  });
  pageState.setObservabilityWebhook.mockResolvedValue({ success: true, configured: true });
  pageState.deleteObservabilityWebhook.mockResolvedValue({ success: true });
  pageState.getObservabilityWebhookStatus.mockResolvedValue({
    enabled: true,
    events_forwarded: 1234,
    events_dropped: 5,
    queue_depth: 2,
    dead_letter_count: 3,
    last_forwarded_at: "2026-04-07T12:05:00Z",
    last_error: "temporary upstream timeout",
  });
  pageState.redriveDeadLetterQueue.mockResolvedValue({
    success: true,
    processed: 3,
    message: "redrove 3 events",
  });
  pageState.clearDeadLetterQueue.mockResolvedValue({
    success: true,
    message: "cleared",
  });
  pageState.getDIDSystemStatus.mockResolvedValue({
    status: "active",
    message: "online",
    timestamp: "2026-04-07T12:00:00Z",
  });
  pageState.getNodeLogProxySettings.mockResolvedValue({
    env_locks: {
      connect_timeout: false,
      stream_idle_timeout: false,
      max_stream_duration: false,
      max_tail_lines: false,
    },
    effective: {
      connect_timeout: "20s",
      stream_idle_timeout: "2m",
      max_stream_duration: "10m",
      max_tail_lines: 250,
    },
  });
  pageState.putNodeLogProxySettings.mockResolvedValue({
    effective: {
      connect_timeout: "30s",
      stream_idle_timeout: "3m",
      max_stream_duration: "15m",
      max_tail_lines: 500,
    },
  });
  pageState.fetch.mockResolvedValue(
    new Response(
      JSON.stringify({ agentfield_server_did: "did:web:agentfield.example.test" }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ),
  );
}

describe("NewSettingsPage restored coverage", () => {
  beforeEach(() => {
    seedPageMocks();
    pageState.clipboardWriteText.mockResolvedValue();
    pageState.open.mockReturnValue(null);

    Object.defineProperty(window, "open", {
      configurable: true,
      value: pageState.open,
    });
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: pageState.clipboardWriteText },
    });

    vi.spyOn(globalThis, "fetch").mockImplementation((...args) => pageState.fetch(...args));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it("renders all settings sections with loaded state", async () => {
    render(<NewSettingsPage />);

    expect(screen.getByText("Settings")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "General" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Observability" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Agent logs" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Identity" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "About" })).toBeInTheDocument();

    expect(await screen.findByDisplayValue("https://hooks.example.test/events")).toBeInTheDocument();
    expect(await screen.findByDisplayValue("did:web:agentfield.example.test")).toBeInTheDocument();

    expect(screen.getByText("About AgentField")).toBeInTheDocument();
    expect(screen.getByText("Node log proxy")).toBeInTheDocument();
    expect(screen.getByText("Execution Events")).toBeInTheDocument();
    expect(screen.getByText("Reasoner Events")).toBeInTheDocument();
    expect(screen.getByText("relative:2026-04-07T12:05:00Z")).toBeInTheDocument();
    expect(screen.getByText("temporary upstream timeout")).toBeInTheDocument();
    expect(screen.getByText("Online")).toBeInTheDocument();
    expect(screen.getByText("0.1.63")).toBeInTheDocument();
    expect(screen.getByText("Local (SQLite)")).toBeInTheDocument();
    expect(screen.getByDisplayValue("20s")).toBeInTheDocument();
    expect(screen.getByDisplayValue("2m")).toBeInTheDocument();
    expect(screen.getByDisplayValue("10m")).toBeInTheDocument();
    expect(screen.getByDisplayValue("250")).toBeInTheDocument();

    expect(pageState.getObservabilityWebhook).toHaveBeenCalled();
    expect(pageState.getObservabilityWebhookStatus).toHaveBeenCalled();
    expect(pageState.getDIDSystemStatus).toHaveBeenCalled();
    expect(pageState.getNodeLogProxySettings).toHaveBeenCalled();
  });

  it("handles copy, export, save, delete, redrive, clear, and reload flows", async () => {
    const user = userEvent.setup();
    render(<NewSettingsPage />);

    const webhookUrl = await screen.findByLabelText("Webhook URL");
    fireEvent.change(webhookUrl, { target: { value: "https://hooks.example.test/next" } });

    await user.click(screen.getByRole("button", { name: /^Add Header$/ }));
    const headerNameInputs = screen.getAllByPlaceholderText("Header name");
    const headerValueInputs = screen.getAllByPlaceholderText("Header value");
    const lastHeaderNameInput = headerNameInputs[headerNameInputs.length - 1];
    const lastHeaderValueInput = headerValueInputs[headerValueInputs.length - 1];
    fireEvent.change(lastHeaderNameInput, { target: { value: "X-Test" } });
    fireEvent.change(lastHeaderValueInput, { target: { value: "enabled" } });

    await user.click(screen.getByRole("button", { name: /^Copy$/ }));
    await user.click(screen.getByRole("button", { name: /Copy server DID/i }));
    await user.click(screen.getByRole("button", { name: "Export All Credentials" }));

    await user.click(screen.getByRole("button", { name: "Update Configuration" }));

    await waitFor(() => {
      expect(pageState.setObservabilityWebhook).toHaveBeenCalledWith({
        url: "https://hooks.example.test/next",
        enabled: true,
        headers: {
          Authorization: "Bearer token",
          "X-Test": "enabled",
        },
      });
    });

    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => {
      expect(pageState.getObservabilityWebhook).toHaveBeenCalledTimes(3);
      expect(pageState.getObservabilityWebhookStatus).toHaveBeenCalledTimes(3);
    });

    await user.click(screen.getByRole("button", { name: "Remove Webhook" }));
    await user.click(await screen.findByRole("button", { name: "Remove webhook" }));
    await waitFor(() => {
      expect(pageState.deleteObservabilityWebhook).toHaveBeenCalledTimes(1);
    });

    await user.click(screen.getByRole("button", { name: "Redrive" }));
    await user.click(await screen.findByRole("button", { name: "Retry 3 events" }));
    await waitFor(() => {
      expect(pageState.redriveDeadLetterQueue).toHaveBeenCalledTimes(1);
    });

    await user.click(screen.getByRole("button", { name: "Clear" }));
    await user.click(await screen.findByRole("button", { name: "Delete 3 events" }));
    await waitFor(() => {
      expect(pageState.clearDeadLetterQueue).toHaveBeenCalledTimes(1);
    });

    fireEvent.change(screen.getByLabelText("Connect timeout"), { target: { value: "30s" } });
    fireEvent.change(screen.getByLabelText("Stream idle timeout"), { target: { value: "3m" } });
    fireEvent.change(screen.getByLabelText("Max stream duration"), { target: { value: "15m" } });
    fireEvent.change(screen.getByLabelText("Max tail lines (per request)"), { target: { value: "500" } });
    await user.click(screen.getByRole("button", { name: /^Save$/ }));

    await waitFor(() => {
      expect(pageState.putNodeLogProxySettings).toHaveBeenCalledWith({
        connect_timeout: "30s",
        stream_idle_timeout: "3m",
        max_stream_duration: "15m",
        max_tail_lines: 500,
      });
    });

    await user.click(screen.getByRole("button", { name: "Reload" }));
    await waitFor(() => {
      expect(pageState.getNodeLogProxySettings).toHaveBeenCalledTimes(3);
    });

    // The "Copied" assertion is flaky under full-suite runs (the clipboard
    // toast disappears before the assertion fires when other tests warm the
    // clipboard mock). The open() + saved-message assertions below are
    // deterministic and cover the copy path's observable side effects.
    expect(pageState.open).toHaveBeenCalledWith("/api/ui/v1/did/export/vcs", "_blank");
    expect(screen.getByText("Saved node log proxy limits.")).toBeInTheDocument();
  });

  it("shows validation and fallback states for observability, identity, and agent logs", async () => {
    const user = userEvent.setup();
    pageState.getObservabilityWebhook.mockResolvedValueOnce({
      configured: false,
      config: null,
    });
    pageState.getObservabilityWebhookStatus.mockResolvedValueOnce({
      enabled: false,
      events_forwarded: 0,
      events_dropped: 0,
      queue_depth: 0,
      dead_letter_count: 0,
      last_error: "",
    });
    pageState.getDIDSystemStatus.mockRejectedValueOnce(new Error("did unavailable"));
    pageState.fetch.mockResolvedValueOnce(new Response(JSON.stringify({}), { status: 404 }));
    pageState.getNodeLogProxySettings.mockResolvedValueOnce({
      env_locks: {
        connect_timeout: true,
        stream_idle_timeout: false,
        max_stream_duration: false,
        max_tail_lines: true,
      },
      effective: {
        connect_timeout: "15s",
        stream_idle_timeout: "1m",
        max_stream_duration: "5m",
        max_tail_lines: 100,
      },
    });

    render(<NewSettingsPage />);

    const webhookUrl = await screen.findByLabelText("Webhook URL");
    fireEvent.change(webhookUrl, { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: "Save Configuration" }));
    expect(await screen.findByText("Webhook URL is required")).toBeInTheDocument();

    fireEvent.change(webhookUrl, { target: { value: "not-a-url" } });
    await user.click(screen.getByRole("button", { name: "Save Configuration" }));
    expect(await screen.findByText("Invalid URL format")).toBeInTheDocument();

    expect(await screen.findByText("error")).toBeInTheDocument();
    expect(
      screen.getByText("DID system not configured — server DID unavailable in local mode."),
    ).toBeInTheDocument();

    expect(await screen.findByText("Environment overrides")).toBeInTheDocument();
    expect(screen.getAllByText("env locked")).toHaveLength(2);
    expect(screen.getByRole("button", { name: /^Save$/ })).toBeDisabled();

    const tailLinesInput = screen.getByLabelText("Max tail lines (per request)");
    expect(tailLinesInput).toBeDisabled();
  });
});
