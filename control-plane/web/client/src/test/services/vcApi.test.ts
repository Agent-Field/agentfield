import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setGlobalApiKey } from "@/services/api";
import {
  copyVCToClipboard,
  downloadDIDResolutionBundle,
  downloadExecutionVCBundle,
  downloadVCDocument,
  downloadWorkflowVCAuditFile,
  exportVCs,
  exportWorkflowComplianceReport,
  formatVCStatus,
  getDIDResolutionBundle,
  getExecutionVCDocument,
  getExecutionVCDocumentEnhanced,
  getExecutionVCStatus,
  getVCStatusSummary,
  getWorkflowAuditTrail,
  getWorkflowVCChain,
  getWorkflowVCStatuses,
  isValidVCDocument,
  verifyExecutionVCComprehensive,
  verifyProvenanceAudit,
  verifyVC,
  verifyWorkflowVCComprehensive,
} from "@/services/vcApi";

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
    blob: vi.fn().mockResolvedValue(new Blob([JSON.stringify(body)], { type: "application/json" })),
  } as unknown as Response;
}

describe("vcApi", () => {
  const originalFetch = globalThis.fetch;
  const originalCreateElement = document.createElement.bind(document);
  const originalCreateObjectURL = URL.createObjectURL;
  const originalRevokeObjectURL = URL.revokeObjectURL;
  const originalClipboard = navigator.clipboard;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-07T12:00:00Z"));
    setGlobalApiKey(null);
    URL.createObjectURL = vi.fn(() => "blob:mock-url");
    URL.revokeObjectURL = vi.fn();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    document.createElement = originalCreateElement;
    URL.createObjectURL = originalCreateObjectURL;
    URL.revokeObjectURL = originalRevokeObjectURL;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: originalClipboard,
    });
    setGlobalApiKey(null);
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("builds verification requests and injects API keys", async () => {
    setGlobalApiKey("secret");
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { verified: true }))
      .mockResolvedValueOnce(jsonResponse(200, { verified: true, details: [] }));

    await expect(verifyVC({ id: "vc-1" })).resolves.toMatchObject({ verified: true });
    await expect(verifyProvenanceAudit({ workflow_id: "wf-1" }, { verbose: true })).resolves.toMatchObject({ verified: true });

    const [verifyUrl, verifyInit] = vi.mocked(globalThis.fetch).mock.calls[0] as [string, RequestInit];
    expect(verifyUrl).toBe("/api/ui/v1/did/verify");
    expect(verifyInit.method).toBe("POST");
    expect(new Headers(verifyInit.headers).get("X-API-Key")).toBe("secret");
    expect(JSON.parse(String(verifyInit.body))).toEqual({ vc_document: { id: "vc-1" } });

    const [auditUrl, auditInit] = vi.mocked(globalThis.fetch).mock.calls[1] as [string, RequestInit];
    expect(auditUrl).toBe("/api/ui/v1/did/verify-audit?verbose=true");
    expect(auditInit.body).toBe(JSON.stringify({ workflow_id: "wf-1" }));
  });

  it("exports VC filters and fetches workflow VC chains", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { execution_vcs: [], workflow_vcs: [], total: 0 }))
      .mockResolvedValueOnce(jsonResponse(200, { workflow_id: "wf-1", component_vcs: [], workflow_vc: null }));

    await expect(exportVCs({ limit: 10, status: "verified", workflow_id: "wf-1" } as any)).resolves.toMatchObject({ total: 0 });
    await expect(getWorkflowVCChain("wf-1")).resolves.toMatchObject({ workflow_id: "wf-1" });

    expect(vi.mocked(globalThis.fetch).mock.calls[0]?.[0]).toBe(
      "/api/ui/v1/did/export/vcs?limit=10&status=verified&workflow_id=wf-1"
    );
    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[0]).toBe("/api/ui/v1/workflows/wf-1/vc-chain");
  });

  it("fetches workflow VC statuses from the batch endpoint and fills defaults", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse(200, {
        summaries: [
          {
            workflow_id: "wf-1",
            has_vcs: true,
            vc_count: 2,
            verified_count: 2,
            failed_count: 0,
            last_vc_created: "2026-04-07T11:00:00Z",
            verification_status: "verified",
          },
        ],
      })
    );

    await expect(getWorkflowVCStatuses(["wf-1", "wf-2"]))
      .resolves
      .toEqual({
        "wf-1": {
          has_vcs: true,
          vc_count: 2,
          verified_count: 2,
          failed_count: 0,
          last_vc_created: "2026-04-07T11:00:00Z",
          verification_status: "verified",
        },
        "wf-2": {
          has_vcs: false,
          vc_count: 0,
          verified_count: 0,
          failed_count: 0,
          last_vc_created: "",
          verification_status: "none",
        },
      });
  });

  it("falls back to legacy workflow status derivation when the batch endpoint fails", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(500, { message: "no batch" }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          workflow_id: "wf-1",
          status: "running",
          component_vcs: [
            { vc_id: "vc-1", created_at: "2026-04-07T10:00:00Z", status: "verified" },
            { vc_id: "vc-2", created_at: "2026-04-07T11:00:00Z", status: "failed" },
          ],
          workflow_vc: null,
        })
      )
      .mockResolvedValueOnce(jsonResponse(404, { message: "missing" }));

    await expect(getWorkflowVCStatuses(["wf-1", "wf-2"]))
      .resolves
      .toEqual({
        "wf-1": {
          has_vcs: true,
          vc_count: 2,
          verified_count: 1,
          failed_count: 1,
          last_vc_created: "2026-04-07T11:00:00Z",
          verification_status: "failed",
        },
        "wf-2": {
          has_vcs: false,
          vc_count: 0,
          verified_count: 0,
          failed_count: 0,
          last_vc_created: "",
          verification_status: "none",
        },
      });

    expect(warnSpy).toHaveBeenCalled();
  });

  it("returns a default summary when workflow id is empty", async () => {
    await expect(getVCStatusSummary("")).resolves.toEqual({
      has_vcs: false,
      vc_count: 0,
      verified_count: 0,
      failed_count: 0,
      last_vc_created: "",
      verification_status: "none",
    });
  });

  it("fetches execution VC status directly and falls back to exports", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(200, {
          has_vc: true,
          vc_id: "vc-1",
          status: "verified",
          created_at: "2026-04-07T11:00:00Z",
        })
      )
      .mockResolvedValueOnce(jsonResponse(404, { message: "missing" }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          execution_vcs: [
            {
              execution_id: "exec-2",
              vc_id: "vc-2",
              status: "completed",
              created_at: "2026-04-07T10:00:00Z",
              storage_uri: "s3://bundle",
              document_size_bytes: 42,
            },
          ],
        })
      )
      .mockRejectedValueOnce(new Error("offline"));

    await expect(getExecutionVCStatus("exec-1")).resolves.toMatchObject({ has_vc: true, vc_id: "vc-1" });
    await expect(getExecutionVCStatus("exec-2")).resolves.toMatchObject({ has_vc: true, vc_id: "vc-2", status: "completed" });
    await expect(getExecutionVCStatus("exec-3")).resolves.toEqual({ has_vc: false, status: "none" });

    expect(errorSpy).toHaveBeenCalled();
  });

  it("maps execution VC download errors into user-facing messages", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { vc_id: "vc-1", vc_document: { ok: true } }))
      .mockResolvedValueOnce(jsonResponse(200, { vc_id: "vc-2", vc_document: null }))
      .mockResolvedValueOnce(jsonResponse(404, { message: "not found" }))
      .mockResolvedValueOnce(jsonResponse(503, { message: "service not available" }))
      .mockResolvedValueOnce(jsonResponse(500, { message: "server exploded" }));

    await expect(getExecutionVCDocument("exec-ok")).resolves.toMatchObject({ vc_id: "vc-1" });
    await expect(getExecutionVCDocument("exec-missing-doc")).rejects.toThrow("VC not found for this execution");
    await expect(getExecutionVCDocument("exec-404")).rejects.toThrow("VC not found for this execution");
    await expect(getExecutionVCDocument("exec-503")).rejects.toThrow("VC service is currently unavailable");
    await expect(getExecutionVCDocument("exec-500")).rejects.toThrow("Failed to fetch execution VC document for download");

    expect(errorSpy).toHaveBeenCalled();
  });

  it("builds enhanced execution VC bundles with DID resolution fallbacks", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(200, {
          vc_id: "vc-1",
          workflow_id: "wf-1",
          session_id: "session-1",
          status: "success",
          created_at: "2026-04-07T10:00:00Z",
          issuer_did: "did:key:issuer",
          vc_document: {
            issuer: "did:key:issuer",
            credentialSubject: {
              caller: { did: "did:key:caller" },
              target: { did: "did:web:target.example" },
            },
          },
        })
      )
      .mockResolvedValueOnce(jsonResponse(200, { verification_keys: [{ publicKeyJwk: { kty: "OKP" } }] }))
      .mockResolvedValueOnce(jsonResponse(200, { verification_keys: [{ publicKeyJwk: { kid: "caller" } }] }))
      .mockResolvedValueOnce(jsonResponse(404, { message: "missing" }));

    const bundle = await getExecutionVCDocumentEnhanced("exec-1");
    expect(bundle.workflow_status).toBe("succeeded");
    expect(bundle.execution_vcs).toHaveLength(1);
    expect(bundle.did_resolution_bundle["did:key:issuer"].resolved_from).toBe("bundled");
    expect(bundle.did_resolution_bundle["did:web:target.example"].resolved_from).toBe("failed");
    expect(bundle.verification_metadata.total_signatures).toBe(1);
    expect(warnSpy).toHaveBeenCalled();
  });

  it("maps workflow audit trails and verification endpoints", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(200, {
          component_vcs: [
            {
              vc_id: "vc-1",
              execution_id: "exec-1",
              created_at: "2026-04-07T10:00:00Z",
              caller_did: "did:key:caller",
              target_did: "did:key:target",
              status: "verified",
              input_hash: "in",
              output_hash: "out",
              signature: "sig",
            },
          ],
        })
      )
      .mockResolvedValueOnce(jsonResponse(500, { message: "boom" }))
      .mockResolvedValueOnce(jsonResponse(200, { status: "verified" }))
      .mockResolvedValueOnce(jsonResponse(200, { status: "verified" }));

    await expect(getWorkflowAuditTrail("wf-1")).resolves.toEqual([
      {
        vc_id: "vc-1",
        execution_id: "exec-1",
        timestamp: "2026-04-07T10:00:00Z",
        caller_did: "did:key:caller",
        target_did: "did:key:target",
        status: "verified",
        input_hash: "in",
        output_hash: "out",
        signature: "sig",
      },
    ]);
    await expect(getWorkflowAuditTrail("wf-2")).resolves.toEqual([]);
    await expect(verifyExecutionVCComprehensive("exec-1")).resolves.toMatchObject({ status: "verified" });
    await expect(verifyWorkflowVCComprehensive("wf-1")).resolves.toMatchObject({ status: "verified" });
    expect(errorSpy).toHaveBeenCalled();
  });

  it("downloads audit and VC artifacts", async () => {
    const click = vi.fn();
    const anchor = originalCreateElement("a");
    anchor.click = click;
    document.createElement = vi.fn((tagName: string) => (tagName === "a" ? anchor : originalCreateElement(tagName)));

    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { workflow_id: "wf:1", component_vcs: [], workflow_vc: null }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          vc_id: "vc-2",
          workflow_id: "wf-2",
          session_id: "session-1",
          status: "success",
          created_at: "2026-04-07T10:00:00Z",
          issuer_did: "did:key:issuer",
          vc_document: { issuer: "did:key:issuer", credentialSubject: {} },
        })
      )
      .mockResolvedValueOnce(jsonResponse(200, { verification_keys: [] }))
      .mockResolvedValueOnce({ ok: true, status: 200, blob: vi.fn().mockResolvedValue(new Blob(["did"], { type: "application/json" })) } as unknown as Response);

    await downloadWorkflowVCAuditFile("wf:1");
    expect(anchor.download).toBe("workflow-wf_1-vc-audit.json");

    await downloadVCDocument({ vc_id: "vc-1", vc_document: { proof: true } } as any);
    expect(anchor.download).toBe("vc-vc-1.json");

    await downloadExecutionVCBundle("exec-2");
    expect(anchor.download).toBe("execution-vc-exec-2.json");

    await downloadDIDResolutionBundle("did:key:issuer");
    expect(anchor.download).toBe("did-resolution-bundle-did_key_issuer.json");
    expect(click).toHaveBeenCalledTimes(4);
    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalled();
  });

  it("copies VC documents to the clipboard and handles clipboard failures", async () => {
    await expect(copyVCToClipboard({ vc_document: { hello: "world" } } as any)).resolves.toBe(true);

    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockRejectedValue(new Error("denied")) },
    });
    await expect(copyVCToClipboard({ vc_document: { hello: "world" } } as any)).resolves.toBe(false);
  });

  it("exports workflow compliance reports in JSON and CSV formats", async () => {
    const click = vi.fn();
    const anchor = originalCreateElement("a");
    anchor.click = click;
    document.createElement = vi.fn((tagName: string) => (tagName === "a" ? anchor : originalCreateElement(tagName)));

    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(200, {
          status: "success",
          component_vcs: [
            {
              vc_id: "vc-1",
              execution_id: "exec-1",
              created_at: "2026-04-07T10:00:00Z",
              status: "verified",
              caller_did: "did:key:caller",
              target_did: "did:key:target",
              input_hash: "input",
              output_hash: "output",
              signature: "sig",
              vc_document: JSON.stringify({ credentialSubject: { caller: { did: "did:key:caller" } } }),
            },
          ],
          workflow_vc: {
            total_steps: 1,
            signature: "workflow-sig",
            vc_document: JSON.stringify({ proof: true }),
          },
          did_resolution_bundle: { "did:key:caller": {} },
        })
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          status: "failed",
          component_vcs: [],
          workflow_vc: null,
          did_resolution_bundle: {},
        })
      )
      .mockResolvedValueOnce(jsonResponse(500, { message: "boom" }));

    await expect(exportWorkflowComplianceReport("wf-1", "json")).resolves.toBeUndefined();
    expect(anchor.download).toBe("workflow-compliance-wf-1.json");

    await expect(exportWorkflowComplianceReport("wf-2", "csv")).resolves.toBeUndefined();
    expect(anchor.download).toBe("workflow-compliance-wf-2.csv");

    await expect(exportWorkflowComplianceReport("wf-3", "json")).rejects.toThrow("Failed to export compliance report");
    expect(click).toHaveBeenCalledTimes(2);
  });

  it("validates VC documents, formats statuses, and fetches DID resolution bundles", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse(200, {
        did: "did:key:test",
        resolution_status: "resolved",
        did_document: { id: "did:key:test" },
        verification_keys: [],
        service_endpoints: [],
        related_vcs: [],
        component_dids: [],
        resolution_metadata: {},
      })
    );

    expect(
      isValidVCDocument({
        "@context": ["https://www.w3.org/2018/credentials/v1"],
        type: ["VerifiableCredential"],
        id: "vc-1",
        issuer: "did:key:test",
        issuanceDate: "2026-04-07T10:00:00Z",
        credentialSubject: { id: "subject" },
        proof: { type: "Ed25519Signature2020" },
      })
    ).toBeTruthy();
    expect(isValidVCDocument("not-json")).toBe(false);
    expect(formatVCStatus("verified")).toEqual({ label: "Verified", variant: "default" });
    expect(formatVCStatus("processing")).toEqual({ label: "Pending", variant: "secondary" });
    expect(formatVCStatus("error")).toEqual({ label: "Failed", variant: "destructive" });
    expect(formatVCStatus("custom")).toEqual({ label: "custom", variant: "outline" });

    await expect(getDIDResolutionBundle("did:key:test")).resolves.toMatchObject({ resolution_status: "resolved" });
    expect(vi.mocked(globalThis.fetch).mock.calls[0]?.[0]).toBe(
      "/api/ui/v1/did/did%3Akey%3Atest/resolution-bundle"
    );
  });
});
