import * as React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { IntegrationsPage } from "@/pages/IntegrationsPage";
import { TriggersPage } from "@/pages/TriggersPage";
import { TooltipProvider } from "@/components/ui/tooltip";

const sources = [
  {
    name: "stripe",
    kind: "http",
    secret_required: true,
    config_schema: { type: "object" },
  },
  {
    name: "cron",
    kind: "loop",
    secret_required: false,
    config_schema: { expression: "string", timezone: "string" },
  },
  {
    name: "snowflake",
    kind: "loop",
    secret_required: true,
    config_schema: { mode: "string", account_url: "string" },
  },
  {
    name: "linear",
    kind: "http",
    secret_required: true,
    config_schema: { tolerance_seconds: "integer" },
  },
  {
    name: "sentry",
    kind: "http",
    secret_required: true,
    config_schema: { type: "object" },
  },
  {
    name: "generic_bearer",
    kind: "http",
    secret_required: true,
    config_schema: {},
  },
];

const triggers = [
  {
    id: "trig_stripe_123456",
    source_name: "stripe",
    config: { account: "acct_123" },
    secret_env_var: "STRIPE_WEBHOOK_SECRET",
    target_node_id: "payments-agent",
    target_reasoner: "handle_payment",
    event_types: ["checkout.session.completed", "invoice.paid"],
    managed_by: "ui",
    enabled: true,
    created_at: "2026-04-01T10:00:00Z",
    updated_at: "2026-04-02T10:00:00Z",
    event_count_24h: 7,
    dispatch_success_24h: 6,
    dispatch_failed_24h: 1,
    last_event_at: "2026-04-02T10:00:00Z",
    dispatch_buckets_24h: [0, 1, 0, 2, 4],
  },
  {
    id: "trig_cron_abcdef",
    source_name: "cron",
    config: { expression: "*/5 * * * *", timezone: "UTC" },
    secret_env_var: "",
    target_node_id: "ops-agent",
    target_reasoner: "handle_tick",
    event_types: [],
    managed_by: "code",
    enabled: false,
    created_at: "2026-04-01T11:00:00Z",
    updated_at: "2026-04-02T11:00:00Z",
    event_count_24h: 0,
    dispatch_success_24h: 0,
    dispatch_failed_24h: 0,
    last_event_at: null,
    dispatch_buckets_24h: [],
  },
];

const events = [
  {
    id: "evt_1",
    trigger_id: "trig_stripe_123456",
    source_name: "stripe",
    event_type: "checkout.session.completed",
    raw_payload: { id: "cs_test", amount_total: 4200 },
    normalized_payload: { checkout_id: "cs_test", amount_total: 4200 },
    idempotency_key: "idem_123456",
    vc_id: "vc_trigger_123456",
    status: "failed",
    error_message: "reasoner timeout",
    received_at: "2026-04-02T10:01:00Z",
    processed_at: "2026-04-02T10:01:03Z",
  },
];

const fetchMock = vi.fn();
const writeTextMock = vi.fn();

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

function jsonResponse(body: unknown, ok = true, status = 200) {
  return Promise.resolve({
    ok,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(typeof body === "string" ? body : JSON.stringify(body)),
  });
}

function installFetch() {
  fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/api/v1/sources")) {
      return jsonResponse({ sources });
    }
    if (url.endsWith("/api/v1/triggers") && !init?.method) {
      return jsonResponse({ triggers });
    }
    if (url.endsWith("/api/v1/triggers") && init?.method === "POST") {
      return jsonResponse({ id: "trig_new" }, true, 201);
    }
    if (url.includes("/api/v1/triggers/trig_stripe_123456/events") && !url.endsWith("/replay")) {
      return jsonResponse({ events });
    }
    if (url.includes("/api/v1/triggers/trig_cron_abcdef/events")) {
      return jsonResponse({ events: [] });
    }
    if (url.includes("/api/v1/triggers/trig_stripe_123456/events/evt_1/replay")) {
      return jsonResponse({ ok: true });
    }
    if (url.includes("/api/v1/triggers/trig_stripe_123456") && init?.method === "PUT") {
      return jsonResponse({ ok: true });
    }
    if (url.includes("/api/v1/triggers/trig_stripe_123456") && init?.method === "DELETE") {
      return jsonResponse({ ok: true });
    }
    return jsonResponse({ message: "not found" }, false, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
}

function renderWithRouter(ui: React.ReactNode, initialPath = "/") {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <TooltipProvider>
        <Routes>
          <Route path="/integrations" element={<IntegrationsPage />} />
          <Route path="/triggers" element={<TriggersPage />} />
          <Route path="/verify" element={<div>Verify destination</div>} />
          <Route path="/runs" element={<div>Runs destination</div>} />
        </Routes>
        {ui}
      </TooltipProvider>
    </MemoryRouter>,
  );
}

describe("trigger management pages", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    installFetch();
    vi.stubGlobal("ResizeObserver", ResizeObserverStub);
    writeTextMock.mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    });
  });

  it("lists integrations, opens the cron create flow, and posts a real trigger payload", async () => {
    const user = userEvent.setup();
    renderWithRouter(null, "/integrations");

    expect(await screen.findByText("Stripe")).toBeInTheDocument();
    expect(screen.getByText("Cron schedule")).toBeInTheDocument();
    expect(screen.getAllByText("1 active").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: /Generic/i }));
    expect(screen.queryByText("Stripe")).not.toBeInTheDocument();
    expect(screen.getByText("Generic Bearer")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Connect/i }));
    expect(await screen.findByRole("heading", { name: "New trigger" })).toBeInTheDocument();
    expect(screen.getByText(/inbound event source/i)).toBeInTheDocument();
    expect(screen.getByText(/Secret env var/i)).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("my-agent"), {
      target: { value: "ops-agent" },
    });
    fireEvent.change(screen.getByPlaceholderText("handle_event"), {
      target: { value: "handle_event" },
    });
    fireEvent.change(screen.getByPlaceholderText("MY_BEARER_TOKEN"), {
      target: { value: "BEARER_TOKEN" },
    });

    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => {
      const createCall = fetchMock.mock.calls.find(
        ([url, init]) => String(url).endsWith("/api/v1/triggers") && init?.method === "POST",
      );
      if (!createCall) throw new Error("expected create trigger call");
      expect(JSON.parse(createCall[1].body as string)).toMatchObject({
        source_name: "generic_bearer",
        target_node_id: "ops-agent",
        target_reasoner: "handle_event",
        event_types: [],
        secret_env_var: "BEARER_TOKEN",
        config: {},
        enabled: true,
      });
    });
  });

  it("opens the Snowflake create flow with poller defaults", async () => {
    const user = userEvent.setup();
    renderWithRouter(null, "/integrations");

    expect(await screen.findByText("Snowflake")).toBeInTheDocument();
    expect(screen.getByText("SQL API")).toBeInTheDocument();

    const connectButtons = screen.getAllByRole("button", { name: /Connect/i });
    const snowflakeConnect = connectButtons.find((button) =>
      button.closest(".group")?.textContent?.includes("Snowflake"),
    );
    expect(snowflakeConnect).toBeTruthy();
    await user.click(snowflakeConnect!);

    expect(await screen.findByRole("heading", { name: "New trigger" })).toBeInTheDocument();
    expect(screen.getByText(/Snowflake event table poller/i)).toBeInTheDocument();
    expect(screen.getAllByText(/programmatic access token/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Event filters/i)).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("handle_snowflake_event")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("SNOWFLAKE_PAT")).toBeInTheDocument();
    expect(screen.getByDisplayValue(/"mode": "event_table_poll"/i)).toBeInTheDocument();
  });

  it("opens Linear and Sentry create flows with provider defaults", async () => {
    const user = userEvent.setup();
    renderWithRouter(null, "/integrations");

    expect(await screen.findByText("Linear")).toBeInTheDocument();
    expect(screen.getByText("Sentry")).toBeInTheDocument();
    expect(screen.getByText("issue.create")).toBeInTheDocument();
    expect(screen.getByText("issue.created")).toBeInTheDocument();

    const linearConnect = screen
      .getAllByRole("button", { name: /Connect/i })
      .find((button) => button.closest(".group")?.textContent?.includes("Linear"));
    expect(linearConnect).toBeTruthy();
    await user.click(linearConnect!);
    expect((await screen.findAllByText(/Linear-Signature/i)).length).toBeGreaterThan(0);
    expect(screen.getByText("Event filters (optional)")).toBeInTheDocument();
    expect(screen.getByText(/Leave blank to accept every event/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText("handle_linear_event")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("LINEAR_WEBHOOK_SECRET")).toBeInTheDocument();
    expect(screen.getByDisplayValue(/"tolerance_seconds": 60/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    const sentryConnect = screen
      .getAllByRole("button", { name: /Connect/i })
      .find((button) => button.closest(".group")?.textContent?.includes("Sentry"));
    expect(sentryConnect).toBeTruthy();
    await user.click(sentryConnect!);
    expect((await screen.findAllByText(/Sentry-Hook-Signature/i)).length).toBeGreaterThan(0);
    expect(screen.getByPlaceholderText("handle_sentry_event")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("SENTRY_CLIENT_SECRET")).toBeInTheDocument();
  });

  it("filters active triggers, opens event evidence, and exercises update/copy/replay paths", async () => {
    const user = userEvent.setup();
    renderWithRouter(null, "/triggers");

    expect(await screen.findByText("Active triggers")).toBeInTheDocument();
    expect(screen.getByText("payments-agent")).toBeInTheDocument();
    expect(screen.getByText("1 failed")).toBeInTheDocument();
    expect(screen.getByText("+1")).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Search triggers…"), {
      target: { value: "cron" },
    });
    expect(screen.getByText("ops-agent")).toBeInTheDocument();
    expect(screen.queryByText("payments-agent")).not.toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Search triggers…"), {
      target: { value: "stripe" },
    });
    const paymentCell = screen.getByText("payments-agent");
    expect(paymentCell).toBeInTheDocument();
    fireEvent.click(paymentCell.closest("tr")!);

    expect(await screen.findByText("Public ingest URL")).toBeInTheDocument();
    expect(screen.getAllByText("checkout.session.completed").length).toBeGreaterThan(0);

    // The events list is fetched asynchronously after the sheet opens
    // (TriggerSheet useEffect → refreshEvents). Under CI load the fetch
    // hasn't resolved when the next assertion runs, so the EventRow
    // button hasn't mounted yet — use findByRole to wait for it instead
    // of getByRole, which races and flakes.
    const eventToggle = await screen.findByRole("button", {
      name: /checkout\.session\.completed/i,
    });
    fireEvent.click(eventToggle);
    expect(await screen.findByText("Verification")).toBeInTheDocument();
    fireEvent.keyDown(eventToggle, { key: "Enter" });
    fireEvent.keyDown(eventToggle, { key: " " });
    expect(await screen.findByText("Verification")).toBeInTheDocument();
    expect(screen.getByText("reasoner timeout")).toBeInTheDocument();
    expect(screen.getByText("Verifiable Credential Chain")).toBeInTheDocument();
    expect(screen.getAllByTitle("vc_trigger_123456").length).toBeGreaterThan(0);

    expect(screen.getByRole("button", { name: /Copy as fixture/i })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /^Replay$/i }));
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/v1/triggers/trig_stripe_123456/events/evt_1/replay",
        expect.objectContaining({ method: "POST" }),
      );
    });

    await user.click(screen.getByLabelText("Disable trigger"));
    await waitFor(() => {
      const updateCall = fetchMock.mock.calls.find(
        ([url, init]) => String(url).includes("/api/v1/triggers/trig_stripe_123456") && init?.method === "PUT",
      );
      if (!updateCall) throw new Error("expected update trigger call");
      expect(JSON.parse(updateCall[1].body as string)).toEqual({ enabled: false });
    });
  });

  it("renders empty/error states and preserves code-managed trigger guardrails", async () => {
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/v1/sources")) return jsonResponse({ sources: [] });
      if (url.endsWith("/api/v1/triggers")) return jsonResponse({ triggers: [] });
      return jsonResponse({ message: "not found" }, false, 404);
    });

    renderWithRouter(null, "/triggers");

    expect(await screen.findByText("No triggers yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Browse integrations/i })).toBeInTheDocument();

    fetchMock.mockImplementation(() => jsonResponse("boom", false, 500));
    renderWithRouter(null, "/integrations");

    expect(await screen.findByText(/HTTP 500/i)).toBeInTheDocument();
    expect(screen.getByText("No integrations match your search")).toBeInTheDocument();
  });

  it("keeps code-managed trigger deletion disabled in the details sheet", async () => {
    const user = userEvent.setup();
    renderWithRouter(null, "/triggers?trigger=trig_cron_abcdef");

    expect(await screen.findByText(/This trigger is paused/i)).toBeInTheDocument();
    expect(screen.getByText("Code-managed")).toBeInTheDocument();
    expect(screen.getAllByText("All events").length).toBeGreaterThan(0);
    // The events list is fetched asynchronously after the sheet opens
    // (TriggerSheet useEffect → refreshEvents). The empty-state copy only
    // renders once `loadingEvents` flips back to false, so this must wait
    // for the fetch to resolve — use findByText, not getByText.
    expect(
      await screen.findByText(
        "No events received yet. Events appear here after the first inbound delivery.",
      ),
    ).toBeInTheDocument();

    const destructiveButtons = screen.getAllByRole("button").filter((button) =>
      button.getAttribute("title")?.includes("Code-managed triggers"),
    );
    expect(destructiveButtons[0]).toBeDisabled();

    await user.click(screen.getByRole("button", { name: /Refresh/i }));
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/v1/triggers/trig_cron_abcdef/events"),
        expect.anything(),
      );
    });
  });
});
