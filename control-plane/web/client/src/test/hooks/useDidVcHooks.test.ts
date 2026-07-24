import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  useDIDInfo,
  useDIDOperations,
  useDIDStatus,
  useDIDUpdates,
  useMultipleDIDStatuses,
} from "@/hooks/useDIDInfo";
import {
  useAuditTrail,
  useExecutionVCStatus,
  useVCExport,
  useVCStatus,
  useVCUpdates,
  useVCVerification,
  useWorkflowVCChain,
  useWorkflowVCStatuses,
} from "@/hooks/useVCVerification";

const didApiState = vi.hoisted(() => ({
  getAgentDIDInfo: vi.fn(),
  getDIDStatusSummary: vi.fn(),
}));

const vcApiState = vi.hoisted(() => ({
  getVCStatusSummary: vi.fn(),
  getWorkflowVCChain: vi.fn(),
  verifyVC: vi.fn(),
  getWorkflowAuditTrail: vi.fn(),
  getExecutionVCStatus: vi.fn(),
  getWorkflowVCStatuses: vi.fn(),
}));

vi.mock("@/services/didApi", () => didApiState);
vi.mock("@/services/vcApi", () => vcApiState);

describe("DID and VC hooks", () => {
  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it("useDIDInfo and useDIDStatus fetch and refetch data", async () => {
    const didInfo = { did: "did:example:1" } as any;
    const didStatus = { has_did: true, did_status: "active" } as any;

    didApiState.getAgentDIDInfo.mockResolvedValue(didInfo);
    didApiState.getDIDStatusSummary.mockResolvedValue(didStatus);

    const infoHook = renderHook(() => useDIDInfo("node-1"));
    const statusHook = renderHook(() => useDIDStatus("node-1"));

    await waitFor(() => expect(infoHook.result.current.didInfo).toEqual(didInfo));
    await waitFor(() => expect(statusHook.result.current.status).toEqual(didStatus));

    await act(async () => {
      await infoHook.result.current.refetch();
      await statusHook.result.current.refetch();
    });

    expect(didApiState.getAgentDIDInfo).toHaveBeenCalledTimes(2);
    expect(didApiState.getDIDStatusSummary).toHaveBeenCalledTimes(2);
  });

  it("useMultipleDIDStatuses handles empty input and per-node failures", async () => {
    didApiState.getDIDStatusSummary
      .mockResolvedValueOnce({ has_did: true, did_status: "active" })
      .mockRejectedValueOnce(new Error("boom"));

    const emptyHook = renderHook(() => useMultipleDIDStatuses([]));
    await waitFor(() => expect(emptyHook.result.current.loading).toBe(false));
    expect(emptyHook.result.current.statuses).toEqual({});

    const nodeIds = ["node-1", "node-2"];
    const { result } = renderHook(() => useMultipleDIDStatuses(nodeIds));
    await waitFor(() => expect(Object.keys(result.current.statuses)).toHaveLength(2));
    expect(result.current.getStatus("node-1")).toEqual(
      expect.objectContaining({ has_did: true })
    );
    expect(result.current.getStatus("node-2")).toEqual(
      expect.objectContaining({ did_status: "inactive" })
    );
  });

  it("useDIDOperations executes operations and surfaces errors", async () => {
    const { result } = renderHook(() => useDIDOperations());

    let successResult: string | null = null;
    await act(async () => {
      successResult = await result.current.executeOperation(async () => "ok");
    });
    expect(successResult).toBe("ok");

    let capturedError: unknown;
    await act(async () => {
      try {
        await result.current.executeOperation(async () => {
          throw new Error("operation failed");
        });
      } catch (error) {
        capturedError = error;
      }
    });

    expect(capturedError).toBeInstanceOf(Error);
    expect((capturedError as Error).message).toBe("operation failed");
    expect(result.current.error).toBe("operation failed");
    act(() => {
      result.current.clearError();
    });
    expect(result.current.error).toBeNull();
  });

  it("useDIDUpdates polls and cleans up its interval", async () => {
    vi.useFakeTimers();
    const onUpdate = vi.fn();
    const clearIntervalSpy = vi.spyOn(globalThis, "clearInterval");
    didApiState.getAgentDIDInfo.mockResolvedValue({ did: "did:example:poll" });

    const { result, unmount } = renderHook(() => useDIDUpdates("node-1", onUpdate));

    await act(async () => {
      vi.advanceTimersByTime(30000);
      await Promise.resolve();
    });

    expect(onUpdate).toHaveBeenCalledWith({ did: "did:example:poll" });
    expect(result.current.lastUpdate).toBeInstanceOf(Date);

    unmount();
    expect(clearIntervalSpy).toHaveBeenCalled();
  });

  it("useVCVerification stores verification results and clears them", async () => {
    vcApiState.verifyVC.mockResolvedValue({ verified: true });

    const { result } = renderHook(() => useVCVerification());

    await act(async () => {
      await result.current.verifyVCDocument({ id: "vc-1" });
    });

    expect(result.current.verificationResult).toEqual({ verified: true });

    vcApiState.verifyVC.mockRejectedValueOnce(new Error("verification failed"));
    await expect(result.current.verifyVCDocument({ id: "vc-2" })).rejects.toThrow(
      "verification failed"
    );
    await waitFor(() => expect(result.current.error).toBe("verification failed"));

    act(() => {
      result.current.clearResults();
    });
    expect(result.current.verificationResult).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("VC fetch hooks load data and report fetch failures", async () => {
    vcApiState.getVCStatusSummary
      .mockResolvedValueOnce({ workflow_id: "wf-1", status: "valid" })
      .mockRejectedValueOnce(new Error("status failed"));
    vcApiState.getWorkflowVCChain.mockResolvedValue({ credentials: [] });
    vcApiState.getWorkflowAuditTrail.mockResolvedValue([{ id: "trail-1" }]);
    vcApiState.getExecutionVCStatus.mockResolvedValue({ execution_id: "exec-1" });

    const statusHook = renderHook(() => useVCStatus("wf-1"));
    const chainHook = renderHook(() => useWorkflowVCChain("wf-1"));
    const trailHook = renderHook(() => useAuditTrail("wf-1"));
    const execHook = renderHook(() => useExecutionVCStatus("exec-1"));

    await waitFor(() => expect(statusHook.result.current.status).toBeTruthy());
    await waitFor(() => expect(chainHook.result.current.vcChain).toEqual({ credentials: [] }));
    await waitFor(() => expect(trailHook.result.current.auditTrail).toEqual([{ id: "trail-1" }]));
    await waitFor(() => expect(execHook.result.current.vcStatus).toEqual({ execution_id: "exec-1" }));

    await act(async () => {
      await statusHook.result.current.refetch();
    });
    await waitFor(() => expect(statusHook.result.current.error).toBe("status failed"));
  });

  it("useWorkflowVCStatuses deduplicates ids and merges refetched results", async () => {
    vcApiState.getWorkflowVCStatuses
      .mockResolvedValueOnce({ "wf-1": { status: "valid" } })
      .mockResolvedValueOnce({
        "wf-1": { status: "valid" },
        "wf-2": { status: "invalid" },
      });

    const { result } = renderHook(() =>
      useWorkflowVCStatuses(["wf-1", "wf-1", "", "wf-2"])
    );

    await waitFor(() => expect(result.current.getStatus("wf-1")).toEqual({ status: "valid" }));
    expect(vcApiState.getWorkflowVCStatuses).toHaveBeenCalledWith(["wf-1", "wf-2"]);

    await act(async () => {
      await result.current.refetch();
    });
    await waitFor(() =>
      expect(result.current.getStatus("wf-2")).toEqual({ status: "invalid" })
    );
  });

  it("useVCExport reports progress, handles errors, and clears later", async () => {
    vi.useFakeTimers();
    const progressCallback = vi.fn();
    const { result } = renderHook(() => useVCExport());

    let exportResult: string | null = null;
    await act(async () => {
      exportResult = await result.current.executeExport(async () => "archive", progressCallback);
    });

    expect(exportResult).toBe("archive");

    expect(progressCallback).toHaveBeenCalledWith(0, 100, "Starting export...");
    expect(progressCallback).toHaveBeenCalledWith(100, 100, "Export completed");
    expect(result.current.exportProgress).toEqual({
      current: 100,
      total: 100,
      status: "Export completed",
    });

    await act(async () => {
      vi.advanceTimersByTime(2000);
      await Promise.resolve();
    });
    expect(result.current.exportProgress).toBeNull();

    let exportError: unknown;
    await act(async () => {
      try {
        await result.current.executeExport(async () => {
          throw new Error("export failed");
        });
      } catch (error) {
        exportError = error;
      }
    });
    expect((exportError as Error).message).toBe("export failed");
    expect(result.current.error).toBe("export failed");

    act(() => {
      result.current.clearError();
    });
    expect(result.current.error).toBeNull();
  });

  it("useVCUpdates polls and cleans up its timer", async () => {
    vi.useFakeTimers();
    const onUpdate = vi.fn();
    const clearIntervalSpy = vi.spyOn(globalThis, "clearInterval");
    vcApiState.getWorkflowVCChain.mockResolvedValue({ credentials: ["vc-1"] });

    const { result, unmount } = renderHook(() => useVCUpdates("wf-1", onUpdate));

    await act(async () => {
      vi.advanceTimersByTime(15000);
      await Promise.resolve();
    });

    expect(onUpdate).toHaveBeenCalledWith({ credentials: ["vc-1"] });
    expect(result.current.lastUpdate).toBeInstanceOf(Date);

    unmount();
    expect(clearIntervalSpy).toHaveBeenCalled();
  });
});
