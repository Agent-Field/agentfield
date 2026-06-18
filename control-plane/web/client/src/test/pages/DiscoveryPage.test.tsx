import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DiscoveryPage } from "@/pages/DiscoveryPage";
import type { ARDDashboard, ARDSearchResponse } from "@/types/ard";

const api = vi.hoisted(() => ({
  getARDDashboard: vi.fn<() => Promise<ARDDashboard>>(),
  updateARDSettings: vi.fn(),
  saveARDPublication: vi.fn(),
  searchExternalARD: vi.fn<(request: unknown) => Promise<ARDSearchResponse>>(),
  importARDEntry: vi.fn(),
  saveARDBinding: vi.fn(),
  saveARDRegistries: vi.fn(),
}));

vi.mock("@/services/ardApi", () => api);

function dashboardFixture(): ARDDashboard {
  return {
    config: {
      enabled: true,
      publish_enabled: true,
      registry_enabled: true,
      registry_public: false,
      external_search_enabled: true,
      external_invocation_enabled: true,
      public_base_url: "https://cp.example.com",
      publisher_domain: "example.com",
      display_name: "Example Control Plane",
      documentation_url: "https://docs.example.com",
      logo_url: "",
      identifier: "did:web:example.com",
      allowed_registries: ["https://registry.example.com/api/v1/ard"],
      locked: {},
    },
    state: {
      settings: {},
      publications: {},
      imports: [],
      bindings: {},
      registries: [{ name: "Example Registry", url: "https://registry.example.com/api/v1/ard" }],
    },
    summary: {
      ard_enabled: true,
      catalog_published: true,
      public_url_reachable: true,
      did_available: true,
      published_reasoners: 1,
      published_skills: 0,
      imported_resources: 0,
      callable_external_resources: 0,
      catalog_url: "https://cp.example.com/.well-known/ai-catalog.json",
    },
    publications: [
      {
        key: "node-1.review_contract",
        target_kind: "reasoner",
        node_id: "node-1",
        target_id: "review_contract",
        published: true,
        display_name: "Review Contract",
        description: "Review contract language for risk and missing clauses.",
        tags: ["legal"],
        capabilities: ["ContractReview"],
        representative_queries: ["review this MSA", "find risky indemnity clauses"],
        artifact_type: "application/openapi+json",
        validation_status: "valid",
        status: "published",
        entry: {
          identifier: "urn:ai:example.com:agentfield:node-1:reasoner:review_contract",
          displayName: "Review Contract",
          description: "Review contract language for risk and missing clauses.",
          type: "application/openapi+json",
          url: "https://cp.example.com/api/v1/ard/artifacts/node-1.review_contract",
          tags: ["legal"],
          capabilities: ["ContractReview"],
          representativeQueries: ["review this MSA", "find risky indemnity clauses"],
          trustManifest: { identity: "did:web:example.com", identityType: "did" },
        },
        agentfield: {
          targetKind: "reasoner",
          nodeId: "node-1",
          targetId: "review_contract",
          invocationTarget: "node-1.review_contract",
          healthStatus: "active",
          version: "1.0.0",
        },
        artifact_url: "https://cp.example.com/api/v1/ard/artifacts/node-1.review_contract",
      },
      {
        key: "node-1:skill:summarize",
        target_kind: "skill",
        node_id: "node-1",
        target_id: "summarize",
        published: false,
        display_name: "Summarize",
        description: "Summarize text deterministically.",
        tags: ["text"],
        capabilities: [],
        representative_queries: ["summarize this note", "make this shorter"],
        artifact_type: "application/openapi+json",
        validation_status: "private",
        status: "private",
        entry: {
          identifier: "urn:ai:example.com:agentfield:node-1:skill:summarize",
          displayName: "Summarize",
          description: "Summarize text deterministically.",
          type: "application/openapi+json",
          url: "https://cp.example.com/api/v1/ard/artifacts/node-1%3Askill%3Asummarize",
        },
        agentfield: {
          targetKind: "skill",
          nodeId: "node-1",
          targetId: "summarize",
          invocationTarget: "node-1:skill:summarize",
          healthStatus: "active",
        },
      },
    ],
    catalog: {
      specVersion: "1.0",
      host: {
        displayName: "Example Control Plane",
        identifier: "did:web:example.com",
        documentationUrl: "https://docs.example.com",
      },
      entries: [
        {
          identifier: "urn:ai:example.com:agentfield:node-1:reasoner:review_contract",
          displayName: "Review Contract",
          description: "Review contract language for risk and missing clauses.",
          type: "application/openapi+json",
          url: "https://cp.example.com/api/v1/ard/artifacts/node-1.review_contract",
          representativeQueries: ["review this MSA", "find risky indemnity clauses"],
        },
      ],
    },
    imports: [],
    registries: [{ name: "Example Registry", url: "https://registry.example.com/api/v1/ard" }],
  };
}

function renderDiscovery(path = "/discovery") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <DiscoveryPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  api.getARDDashboard.mockResolvedValue(dashboardFixture());
  api.updateARDSettings.mockImplementation(async () => dashboardFixture());
  api.saveARDPublication.mockImplementation(async () => dashboardFixture());
  api.importARDEntry.mockImplementation(async () => dashboardFixture());
  api.saveARDBinding.mockImplementation(async () => dashboardFixture());
  api.saveARDRegistries.mockImplementation(async () => dashboardFixture());
});

describe("DiscoveryPage", () => {
  it("shows an error state and retries loading", async () => {
    const user = userEvent.setup();
    api.getARDDashboard
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValueOnce(dashboardFixture());

    renderDiscovery();

    expect(await screen.findByText("Could not load ARD state: network down.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("heading", { name: "Discovery" })).toBeInTheDocument();
    expect(api.getARDDashboard).toHaveBeenCalledTimes(2);
  });

  it("shows ARD overview status and deployment controls", async () => {
    renderDiscovery();

    expect(await screen.findByRole("heading", { name: "Discovery" })).toBeInTheDocument();
    expect(screen.getByText("Published reasoners")).toBeInTheDocument();
    expect(screen.getByText("Callable external")).toBeInTheDocument();
    expect(screen.getByText("https://cp.example.com/.well-known/ai-catalog.json")).toBeInTheDocument();
    expect(screen.getByText("Deployment config")).toBeInTheDocument();
  });

  it("saves overview discovery settings and opens catalog actions", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn();
    const open = vi.spyOn(window, "open").mockImplementation(() => null);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    renderDiscovery();

    await screen.findByRole("heading", { name: "Discovery" });
    await user.click(screen.getByRole("button", { name: "Copy" }));
    await user.click(screen.getByRole("button", { name: "Open" }));
    const displayNameInput = screen.getByDisplayValue("Example Control Plane");
    await user.clear(displayNameInput);
    await user.type(displayNameInput, "Runtime Discovery");
    await user.click(screen.getByRole("button", { name: "Save discovery settings" }));

    expect(writeText).toHaveBeenCalledWith("https://cp.example.com/.well-known/ai-catalog.json");
    expect(open).toHaveBeenCalledWith("https://cp.example.com/.well-known/ai-catalog.json", "_blank");
    await waitFor(() => {
      expect(api.updateARDSettings).toHaveBeenCalled();
    });
    expect(api.updateARDSettings.mock.calls[0][0]).toEqual(
      expect.objectContaining({ display_name: "Runtime Discovery" }),
    );
  });

  it("selects a targeted publication from the URL for catalog editing", async () => {
    const user = userEvent.setup();
    renderDiscovery("/discovery?target=node-1%3Askill%3Asummarize");

    await user.click(await screen.findByRole("tab", { name: "Public Catalog" }));

    expect(screen.getByDisplayValue("Summarize")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save exposure" })).toBeInTheDocument();
  });

  it("supports keyboard selection and empty state in the public catalog", async () => {
    const user = userEvent.setup();
    renderDiscovery();

    await user.click(await screen.findByRole("tab", { name: "Public Catalog" }));
    const summarizeRow = screen.getByRole("button", { name: /Summarize/ });
    summarizeRow.focus();
    await user.keyboard("{Enter}");

    expect(screen.getByDisplayValue("Summarize")).toBeInTheDocument();
    expect(screen.getByText("AgentField target")).toBeInTheDocument();
    expect(screen.getByText("node-1:skill:summarize")).toBeInTheDocument();
    cleanup();

    const empty = dashboardFixture();
    empty.publications = [];
    empty.catalog.entries = [];
    api.getARDDashboard.mockResolvedValue(empty);
    renderDiscovery();

    await user.click(await screen.findByRole("tab", { name: "Public Catalog" }));
    expect(screen.getByText("No publishable resources yet")).toBeInTheDocument();
  });

  it("edits publication exposure fields before saving", async () => {
    const user = userEvent.setup();
    const fixture = dashboardFixture();
    fixture.publications[1].validation_errors = ["description is required"];
    fixture.publications[1].validation_status = "invalid";
    api.getARDDashboard.mockResolvedValue(fixture);

    renderDiscovery("/discovery?target=node-1%3Askill%3Asummarize");

    await user.click(await screen.findByRole("tab", { name: "Public Catalog" }));
    const nameInput = screen.getByDisplayValue("Summarize");
    await user.clear(nameInput);
    await user.type(nameInput, "Summarize Notes");
    const tagsInput = screen.getByDisplayValue("text");
    await user.clear(tagsInput);
    await user.type(tagsInput, "notes, internal");
    await user.click(screen.getByRole("button", { name: "Save exposure" }));

    await waitFor(() => {
      expect(api.saveARDPublication).toHaveBeenCalled();
    });
    expect(api.saveARDPublication.mock.calls[0][0]).toEqual(
      expect.objectContaining({
        display_name: "Summarize Notes",
        tags: expect.arrayContaining(["notesinternal"]),
      }),
    );
  });

  it("uses the ARD query model for external search and imports top-level result entries", async () => {
    const user = userEvent.setup();
    let resolveSearch!: (response: ARDSearchResponse) => void;
    api.searchExternalARD.mockImplementation(() => new Promise((resolve) => {
      resolveSearch = resolve;
    }));
    const searchResponse: ARDSearchResponse = {
      sources: [{ url: "https://registry.example.com/api/v1/ard", status: "ok" }],
      results: [
        {
          identifier: "urn:ai:vendor.example:agent:review_contract",
          displayName: "Vendor Contract Reviewer",
          description: "External contract review agent.",
          type: "application/a2a-agent-card+json",
          url: "https://vendor.example/agent-card.json",
          tags: ["legal"],
          score: 94,
          source: "https://registry.example.com/api/v1/ard",
        },
      ],
    };

    renderDiscovery();

    await user.click(await screen.findByRole("tab", { name: "External Search" }));
    await user.type(screen.getByPlaceholderText("review contracts, summarize filings, generate tests"), "review contracts");
    await user.click(screen.getByRole("button", { name: "Search" }));
    expect(screen.getByRole("button", { name: "Searching" })).toBeDisabled();

    await waitFor(() => {
      expect(api.searchExternalARD.mock.calls[0][0]).toEqual({
        query: { text: "review contracts" },
        pageSize: 20,
        federation: "none",
      });
    });
    resolveSearch(searchResponse);

    expect(await screen.findByText("Vendor Contract Reviewer")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Import" }));

    await waitFor(() => {
      expect(api.importARDEntry).toHaveBeenCalledWith(
        expect.objectContaining({
          identifier: "urn:ai:vendor.example:agent:review_contract",
          displayName: "Vendor Contract Reviewer",
        }),
        "https://registry.example.com/api/v1/ard",
      );
    });
  });

  it("shows external search no-results and error states", async () => {
    const user = userEvent.setup();
    api.searchExternalARD.mockResolvedValueOnce({
      sources: [{ url: "https://registry.example.com/api/v1/ard", status: "ok" }],
      results: [],
    });

    renderDiscovery();
    await user.click(await screen.findByRole("tab", { name: "External Search" }));
    await user.type(screen.getByPlaceholderText("review contracts, summarize filings, generate tests"), "missing");
    await user.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("No external ARD resources matched this search.")).toBeInTheDocument();

    api.searchExternalARD.mockRejectedValueOnce(new Error("registry unavailable"));
    await user.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText("registry unavailable")).toBeInTheDocument();
  });

  it("shows empty imports and saves callable bindings for imported resources", async () => {
    const user = userEvent.setup();

    renderDiscovery();
    await user.click(await screen.findByRole("tab", { name: "Imports" }));
    expect(screen.getByText(/Imported external ARD resources will appear here/)).toBeInTheDocument();
    cleanup();

    const imported = dashboardFixture();
    imported.imports = [
      {
        status: "imported",
        entry: {
          id: "ext_123",
          source_registry: "https://registry.example.com/api/v1/ard",
          identifier: "urn:ai:vendor.example:agent:review",
          type: "application/a2a-agent-card+json",
          display_name: "Vendor Review",
          description: "External reviewer.",
          publisher: "vendor.example",
          imported_at: "2026-06-18T12:00:00Z",
        },
      },
    ];
    api.getARDDashboard.mockResolvedValue(imported);
    renderDiscovery();

    await user.click(await screen.findByRole("tab", { name: "Imports" }));
    const targetInput = await screen.findByDisplayValue("external.vendor_example.vendor_review");
    await user.clear(targetInput);
    await user.type(targetInput, "external.vendor.review_contract");
    const timeoutInput = screen.getByDisplayValue("30000");
    await user.clear(timeoutInput);
    await user.type(timeoutInput, "45000");
    await user.type(screen.getByLabelText("Allowed operations"), "review, summarize");
    await user.type(screen.getByLabelText("Policy"), "legal-only");
    await user.click(screen.getByRole("button", { name: "Save callable binding" }));

    await waitFor(() => {
      expect(api.saveARDBinding).toHaveBeenCalled();
    });
    expect(api.saveARDBinding.mock.calls[0][0]).toBe("ext_123");
    expect(api.saveARDBinding.mock.calls[0][1]).toEqual(
      expect.objectContaining({
        external_entry_id: "ext_123",
        local_target: "external.vendor.review_contract",
        allowed_operations: ["review", "summarize"],
        policy: "legal-only",
      }),
    );
  });

  it("adds, validates, removes, and saves registry rows", async () => {
    const user = userEvent.setup();
    renderDiscovery();

    await user.click(await screen.findByRole("tab", { name: "Registry" }));
    await user.click(screen.getByRole("button", { name: "Add registry" }));
    expect(screen.getByText("Registry URL is required.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save registries" })).toBeDisabled();
    const urlInputs = screen.getAllByPlaceholderText("https://registry.example/api/v1/ard");
    await user.type(urlInputs[urlInputs.length - 1], "https://new-registry.example/api/v1/ard");
    await user.click(screen.getByRole("button", { name: "Remove registry Example Registry" }));
    await user.click(screen.getByRole("button", { name: "Save registries" }));

    await waitFor(() => {
      expect(api.saveARDRegistries).toHaveBeenCalled();
    });
    expect(api.saveARDRegistries.mock.calls[0][0]).toEqual(
      [expect.objectContaining({ url: "https://new-registry.example/api/v1/ard" })],
    );
  });
});
