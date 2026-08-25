import { useEffect, useState } from "react";
import type {
    CloudAutoUpdateMode,
    CloudDeployResult,
    CloudTestResult,
    CloudUpdateStatus,
    DesktopSettings,
    RailwayStatus,
} from "../../../shared/types";

export function deployedWorkspacePickerVisible(railway: RailwayStatus): boolean {
    return railway.hasDeployment && railway.workspaces.length > 1;
}

export function deployedWorkspacePickerDisabled(
    railway: RailwayStatus,
    busy: boolean,
): boolean {
    return railway.hasDeployment || busy;
}

export function deploymentActionWorkspaceId(
    railway: RailwayStatus | null,
    selectedWorkspaceId: string,
): string {
    if (railway?.hasDeployment) return railway.deploymentWorkspaceId ?? "";
    return selectedWorkspaceId;
}

export function railwayImageUpdatesVisible(status: CloudUpdateStatus): boolean {
    if (status.canManageRailway !== undefined) return status.canManageRailway;
    return Boolean(
        status.canApply &&
            status.hosting?.platform === "railway" &&
            status.hosting.service_id &&
            status.hosting.environment_id,
    );
}

export function cloudUpdateActionVisible(status: CloudUpdateStatus): boolean {
    return (
        status.canApply &&
        (status.status === "available" ||
            (status.status === "legacy" && status.latest !== null))
    );
}

export function cloudUpdateActionLabel(status: CloudUpdateStatus): string {
    return status.status === "legacy" ? "Update control plane" : "Update now";
}

type CloudUpdateFeedback = { ok: boolean; text: string };

export function cloudUpdateFeedbackClass(feedback: CloudUpdateFeedback): string {
    return feedback.ok ? "row-sub" : "row-sub error-text";
}

type Confirmation = {
    mode: "cloud" | "local";
    host?: string;
};

type CloudTab = "railway" | "manual";

const CLOUD_TABS: Array<{ id: CloudTab; label: string }> = [
    { id: "railway", label: "Railway" },
    { id: "manual", label: "Manual" },
];

export function CloudPanel() {
    const [settings, setSettings] = useState<DesktopSettings | null>(null);
    const [serverUrl, setServerUrl] = useState("");
    const [apiKey, setApiKey] = useState("");
    const [showKey, setShowKey] = useState(false);
    const [testing, setTesting] = useState(false);
    const [saving, setSaving] = useState(false);
    const [result, setResult] = useState<CloudTestResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
    const [railway, setRailway] = useState<RailwayStatus | null>(null);
    const [cloudUpdate, setCloudUpdate] = useState<CloudUpdateStatus | null>(null);
    const [cloudUpdateBusy, setCloudUpdateBusy] = useState<"apply" | "schedule" | null>(null);
    const [cloudUpdateMessage, setCloudUpdateMessage] =
        useState<CloudUpdateFeedback | null>(null);
    const [railwayBusy, setRailwayBusy] = useState<
        "login" | "deploy" | "destroy" | null
    >(null);
    const [workspaceId, setWorkspaceId] = useState("");
    const [deployLines, setDeployLines] = useState<string[]>([]);
    const [deployResult, setDeployResult] = useState<CloudDeployResult | null>(
        null,
    );
    const [deleteText, setDeleteText] = useState("");
    const [showDestroy, setShowDestroy] = useState(false);
    const [destroyed, setDestroyed] = useState(false);
    const [activeTab, setActiveTab] = useState<CloudTab>("railway");

    useEffect(() => {
        void window.agentfield.getSettings().then((next) => {
            setSettings(next);
            setServerUrl(next.cloud?.serverUrl ?? "");
            setApiKey(next.cloud?.apiKey ?? "");
        });
    }, []);

    useEffect(() => {
        void window.agentfield
            .railwayStatus()
            .then(setRailway)
            .catch((err) => {
                setError(err instanceof Error ? err.message : String(err));
            });
        return window.agentfield.onCloudDeployProgress((line) => {
            setDeployLines((lines) => [...lines.slice(-199), line]);
        });
    }, []);

    useEffect(() => {
        if (!settings?.cloud.enabled) return;
        void window.agentfield.checkCloudUpdate().then(setCloudUpdate);
        return window.agentfield.onCloudUpdateStatus(setCloudUpdate);
    }, [settings?.cloud.enabled]);

    useEffect(() => {
        if (!railway?.loggedIn) {
            setWorkspaceId("");
        } else if (railway.hasDeployment && railway.deploymentWorkspaceId) {
            setWorkspaceId(railway.deploymentWorkspaceId);
        } else if (
            workspaceId === "" &&
            railway.deploymentWorkspaceId &&
            railway.workspaces.some((workspace) => workspace.id === railway.deploymentWorkspaceId)
        ) {
            setWorkspaceId(railway.deploymentWorkspaceId);
        } else if (railway.workspaces.length === 1) {
            setWorkspaceId(railway.workspaces[0].id);
        } else if (
            !railway.workspaces.some(
                (workspace) => workspace.id === workspaceId,
            )
        ) {
            setWorkspaceId("");
        }
    }, [railway, workspaceId]);

    useEffect(() => {
        if (railway && !railway.engineAvailable) setActiveTab("manual");
    }, [railway?.engineAvailable]);

    useEffect(() => {
        if (!confirmation) return;
        const timeout = window.setTimeout(() => setConfirmation(null), 4000);
        return () => window.clearTimeout(timeout);
    }, [confirmation]);

    const cloud = settings?.cloud;
    const enabled = cloud?.enabled ?? false;
    const canSubmit = serverUrl.trim() !== "" && apiKey.trim() !== "";
    const busy = testing || saving;

    const test = async () => {
        setTesting(true);
        setError(null);
        setConfirmation(null);
        setResult(null);
        try {
            setResult(
                await window.agentfield.cloudTest(
                    serverUrl.trim(),
                    apiKey.trim(),
                ),
            );
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setTesting(false);
        }
    };

    const saveCloud = async () => {
        if (!result?.ok) {
            const proceed = window.confirm(
                "The connection has not passed its test. Switch to this remote control plane anyway?",
            );
            if (!proceed) return;
        }
        setSaving(true);
        setError(null);
        setConfirmation(null);
        try {
            const next = await window.agentfield.setCloudProfile({
                enabled: true,
                serverUrl: serverUrl.trim(),
                apiKey: apiKey.trim(),
            });
            setSettings(next);
            setServerUrl(next.cloud?.serverUrl ?? serverUrl.trim());
            setApiKey(next.cloud?.apiKey ?? apiKey.trim());
            setConfirmation({
                mode: "cloud",
                host: displayHost(next.cloud?.serverUrl ?? serverUrl.trim()),
            });
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setSaving(false);
        }
    };

    const disconnect = async () => {
        setSaving(true);
        setError(null);
        setConfirmation(null);
        try {
            const next = await window.agentfield.setCloudProfile({
                enabled: false,
                serverUrl: serverUrl.trim(),
                apiKey: apiKey.trim(),
            });
            setSettings(next);
            setConfirmation({ mode: "local" });
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setSaving(false);
        }
    };

    const refreshRailway = async () =>
        setRailway(await window.agentfield.railwayStatus());

    const railwayLogin = async () => {
        setRailwayBusy("login");
        setError(null);
        try {
            const login = await window.agentfield.railwayLogin();
            if (!login.ok) setError(login.message);
            await refreshRailway();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setRailwayBusy(null);
        }
    };

    const railwayLogout = async () => {
        await window.agentfield.railwayLogout();
        setDeployResult(null);
        await refreshRailway();
    };

    const deploy = async () => {
        const actionWorkspaceId = deploymentActionWorkspaceId(
            railway,
            workspaceId,
        );
        if (!actionWorkspaceId) return;
        setRailwayBusy("deploy");
        setDeployLines([]);
        setDeployResult(null);
        setDestroyed(false);
        setError(null);
        try {
            const nextResult =
                await window.agentfield.cloudDeploy(actionWorkspaceId);
            setDeployResult(nextResult);
            if (nextResult.ok) {
                const next = await window.agentfield.getSettings();
                setSettings(next);
                setServerUrl(next.cloud.serverUrl);
                setApiKey(next.cloud.apiKey);
            }
            await refreshRailway();
            if (nextResult.ok) {
                setCloudUpdate(await window.agentfield.checkCloudUpdate());
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setRailwayBusy(null);
        }
    };

    const destroy = async () => {
        if (deleteText !== "delete") return;
        setRailwayBusy("destroy");
        setError(null);
        try {
            const result = await window.agentfield.cloudDestroy();
            if (!result.ok) {
                setError(result.message);
                return;
            }
            const next = await window.agentfield.getSettings();
            setSettings(next);
            setShowDestroy(false);
            setDeleteText("");
            setDeployResult(null);
            setCloudUpdate(null);
            setDestroyed(true);
            await refreshRailway();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setRailwayBusy(null);
        }
    };

    const applyCloudControlPlaneUpdate = async () => {
        setCloudUpdateBusy("apply");
        setCloudUpdateMessage(null);
        try {
            const applied = await window.agentfield.applyCloudUpdate();
            setCloudUpdateMessage({ ok: applied.ok, text: applied.message });
            setCloudUpdate(await window.agentfield.checkCloudUpdate());
        } catch (err) {
            setCloudUpdateMessage({
                ok: false,
                text: `${err instanceof Error ? err.message : String(err)} Check Railway deployment logs and try again.`,
            });
        } finally {
            setCloudUpdateBusy(null);
        }
    };

    const setAutoUpdate = async (mode: CloudAutoUpdateMode) => {
        setCloudUpdateBusy("schedule");
        setCloudUpdateMessage(null);
        try {
            const changed = await window.agentfield.setCloudAutoUpdate(mode);
            if (!changed.ok) {
                setCloudUpdateMessage({ ok: false, text: changed.message });
                return;
            }
            const next = await window.agentfield.getSettings();
            setSettings(next);
            setCloudUpdateMessage({ ok: true, text: changed.message });
        } catch (err) {
            setCloudUpdateMessage({
                ok: false,
                text: `${err instanceof Error ? err.message : String(err)} Try again.`,
            });
        } finally {
            setCloudUpdateBusy(null);
        }
    };

    if (!settings) {
        return (
            <div className="panel">
                <div className="empty secondary">Loading…</div>
            </div>
        );
    }

    return (
        <>
            <p className="view-lede">
                Run your agents from a control plane in the cloud. Deploy one to
                Railway, or connect to a server you already run.
            </p>

            {error && <div className="callout error">{error}</div>}
            {confirmation && (
                <div
                    className="callout success cloud-confirmation"
                    role="status"
                >
                    {confirmation.mode === "cloud"
                        ? `✓ Now managing ${confirmation.host}`
                        : "✓ Switched back to the local control plane"}
                </div>
            )}

            <div className="panel cloud-status-strip">
                <span
                    className={`cloud-status-dot ${enabled ? "connected" : ""}`}
                    aria-hidden="true"
                />
                <div className="row-main">
                    <span className="row-title cloud-status-title">
                        {enabled
                            ? `Remote: ${displayHost(cloud?.serverUrl || serverUrl)}`
                            : "Local control plane"}
                    </span>
                    {enabled && (
                        <span className="row-sub">
                            Local server management is disabled while this remote
                            connection is active.
                        </span>
                    )}
                </div>
                {enabled && (
                    <button
                        className="action-button"
                        disabled={saving}
                        onClick={() => void disconnect()}
                    >
                        Switch back to local
                    </button>
                )}
            </div>

            {enabled && cloudUpdate && (
                <div className="panel cloud-status-strip">
                    <div className="row-main" aria-live="polite">
                        <span className="row-title">
                            Control plane {cloudUpdate.current ? `v${cloudUpdate.current}` : "version unknown"}
                            {" · "}{cloudPlatformLabel(cloudUpdate.hosting?.platform)}
                            {" · "}{cloudUpdateLabel(cloudUpdate)}
                        </span>
                        <span className="row-sub">{cloudUpdate.message}</span>
                        {cloudUpdateMessage && (
                            <span className={cloudUpdateFeedbackClass(cloudUpdateMessage)}>
                                {cloudUpdateMessage.text}
                            </span>
                        )}
                    </div>
                    <div className="row-side">
                        {cloudUpdateActionVisible(cloudUpdate) && (
                            <button
                                type="button"
                                className="action-button primary"
                                disabled={cloudUpdateBusy !== null || cloudUpdate.applying}
                                onClick={() => void applyCloudControlPlaneUpdate()}
                            >
                                {cloudUpdateBusy === "apply" || cloudUpdate.applying
                                    ? "Updating…"
                                    : cloudUpdateActionLabel(cloudUpdate)}
                            </button>
                        )}
                        {railway?.loggedIn &&
                            railwayImageUpdatesVisible(cloudUpdate) && (
                                <label className="cloud-workspace-field">
                                    <span className="row-sub">Railway image updates</span>
                                    <select
                                        className="env-input"
                                        aria-label="Railway image auto-update schedule"
                                        value={settings.cloud.autoUpdate ?? ""}
                                        disabled={cloudUpdateBusy !== null}
                                        onChange={(event) =>
                                            void setAutoUpdate(event.target.value as CloudAutoUpdateMode)
                                        }
                                    >
                                        <option value="" disabled>
                                            Not set — choose a window
                                        </option>
                                        <option value="off">Off — never update automatically</option>
                                        <option value="nightly">Nightly — every day, 02:00–06:00 UTC</option>
                                        <option value="weekends">Weekends — Saturday and Sunday, all day UTC</option>
                                        <option value="anytime">Anytime — apply after Railway detects a release</option>
                                    </select>
                                </label>
                            )}
                    </div>
                </div>
            )}

            <div
                className="cloud-tabs"
                role="tablist"
                aria-label="Remote connection method"
            >
                {CLOUD_TABS.map((tab) => (
                    <button
                        key={tab.id}
                        className={`cloud-tab ${activeTab === tab.id ? "active" : ""}`}
                        type="button"
                        role="tab"
                        aria-selected={activeTab === tab.id}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {activeTab === "manual" && (
                <section className="settings-section" role="tabpanel">
                    <div className="panel cloud-form">
                        <div className="cloud-field">
                            <label
                                className="row-title"
                                htmlFor="cloud-server-url"
                            >
                                Server URL
                            </label>
                            <span className="row-sub">
                                The public address of your AgentField control
                                plane.
                            </span>
                            <div className="cloud-input-row">
                                <input
                                    id="cloud-server-url"
                                    className="env-input cloud-input"
                                    placeholder="https://your-cp.up.railway.app"
                                    value={serverUrl}
                                    disabled={busy}
                                    onChange={(event) => {
                                        setServerUrl(event.target.value);
                                        setResult(null);
                                    }}
                                />
                            </div>
                        </div>
                        <div className="cloud-field">
                            <label
                                className="row-title"
                                htmlFor="cloud-api-key"
                            >
                                API key
                            </label>
                            <span className="row-sub">
                                Stored on this computer and sent only to your
                                server.
                            </span>
                            <div className="cloud-input-row cloud-key-row">
                                <input
                                    id="cloud-api-key"
                                    className="env-input cloud-input"
                                    type={showKey ? "text" : "password"}
                                    value={apiKey}
                                    disabled={busy}
                                    onChange={(event) => {
                                        setApiKey(event.target.value);
                                        setResult(null);
                                    }}
                                />
                                <button
                                    className="action-button cloud-key-toggle"
                                    type="button"
                                    disabled={busy}
                                    onClick={() => setShowKey(!showKey)}
                                >
                                    {showKey ? "Hide" : "Show"}
                                </button>
                            </div>
                        </div>
                    </div>
                    <div className="cloud-actions">
                        <button
                            className={`action-button ${!result || !result.ok || !result.installApi ? "primary" : ""}`}
                            disabled={!canSubmit || busy}
                            onClick={() => void test()}
                        >
                            {testing && (
                                <span
                                    className="cloud-spinner"
                                    aria-hidden="true"
                                />
                            )}
                            {testing ? "Testing…" : "Test connection"}
                        </button>
                        <button
                            className={`action-button ${result?.ok && result.installApi ? "primary" : ""}`}
                            disabled={!canSubmit || busy}
                            onClick={() => void saveCloud()}
                        >
                            {saving ? "Saving…" : "Save & switch to Remote"}
                        </button>
                    </div>
                    {result && <CloudTestFeedback result={result} />}
                </section>
            )}

            {activeTab === "railway" && (
                <section className="settings-section" role="tabpanel">
                    <div className="panel cloud-railway">
                        <div className="cloud-railway-content">
                            {!railway ? (
                                <span className="row-sub">
                                    Checking one-click deploy…
                                </span>
                            ) : !railway.engineAvailable ? (
                                <div className="cloud-engine-info cloud-guided-state">
                                    <span className="row-title">
                                        One-click deploy isn't bundled in this
                                        build
                                    </span>
                                    <button
                                        className="cloud-link-button"
                                        type="button"
                                        onClick={() =>
                                            void window.agentfield.cloudDeployRailway()
                                        }
                                    >
                                        Use the Railway template instead
                                    </button>
                                </div>
                            ) : !railway.loggedIn ? (
                                <div className="cloud-guided-state">
                                    <span className="row-sub cloud-railway-copy">
                                        Deploys the control plane to YOUR
                                        Railway account — usage is billed by
                                        Railway.
                                    </span>
                                    <button
                                        className="action-button primary"
                                        type="button"
                                        disabled={railwayBusy !== null}
                                        onClick={() => void railwayLogin()}
                                    >
                                        {railwayBusy === "login" && (
                                            <span
                                                className="cloud-spinner"
                                                aria-hidden="true"
                                            />
                                        )}
                                        {railwayBusy === "login"
                                            ? "Waiting for browser…"
                                            : "Log in with Railway"}
                                    </button>
                                </div>
                            ) : railwayBusy === "deploy" ? (
                                <div className="cloud-guided-state">
                                    <span className="row-title cloud-progress-title">
                                        <span
                                            className="cloud-spinner"
                                            aria-hidden="true"
                                        />{" "}
                                        Deploying control plane…
                                    </span>
                                    <div
                                        className="cloud-deploy-log"
                                        role="log"
                                        aria-live="polite"
                                    >
                                        {deployLines.map((line, index) => (
                                            <span
                                                className="install-progress-line"
                                                key={`${index}-${line}`}
                                            >
                                                {line}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ) : railway.hasDeployment || deployResult?.ok ? (
                                <div className="cloud-guided-state">
                                    <div
                                        className="callout success"
                                        role="status"
                                    >
                                        ✓ Deployed — connected to{" "}
                                        {displayHost(
                                            deployResult?.url ??
                                                settings.cloud.serverUrl,
                                        )}
                                    </div>
                                    {deployedWorkspacePickerVisible(railway) && (
                                        <WorkspacePicker
                                            railway={railway}
                                            value={workspaceId}
                                            disabled={deployedWorkspacePickerDisabled(
                                                railway,
                                                railwayBusy !== null,
                                            )}
                                            onChange={setWorkspaceId}
                                        />
                                    )}
                                    <div className="cloud-actions">
                                        <button
                                            className="action-button primary"
                                            type="button"
                                            disabled={
                                                !deploymentActionWorkspaceId(
                                                    railway,
                                                    workspaceId,
                                                ) ||
                                                railwayBusy !== null
                                            }
                                            onClick={() => void deploy()}
                                        >
                                            Re-run deploy
                                        </button>
                                        <button
                                            className="cloud-link-button danger"
                                            type="button"
                                            onClick={() => setShowDestroy(true)}
                                        >
                                            Tear down
                                        </button>
                                    </div>
                                    {showDestroy && (
                                        <div className="cloud-destroy-confirm">
                                            <label
                                                className="row-sub"
                                                htmlFor="cloud-delete-confirm"
                                            >
                                                Type <strong>delete</strong> to
                                                tear down this deployment.
                                            </label>
                                            <div className="cloud-actions">
                                                <input
                                                    id="cloud-delete-confirm"
                                                    className="env-input"
                                                    value={deleteText}
                                                    onChange={(event) =>
                                                        setDeleteText(
                                                            event.target.value,
                                                        )
                                                    }
                                                />
                                                <button
                                                    className="action-button danger"
                                                    disabled={
                                                        deleteText !==
                                                            "delete" ||
                                                        railwayBusy !== null
                                                    }
                                                    onClick={() =>
                                                        void destroy()
                                                    }
                                                >
                                                    {railwayBusy === "destroy"
                                                        ? "Tearing down…"
                                                        : "Tear down"}
                                                </button>
                                                <button
                                                    className="action-button"
                                                    disabled={
                                                        railwayBusy !== null
                                                    }
                                                    onClick={() =>
                                                        setShowDestroy(false)
                                                    }
                                                >
                                                    Cancel
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="cloud-guided-state">
                                    <div className="cloud-railway-heading">
                                        <span className="row-title">
                                            Ready to deploy
                                        </span>
                                        <button
                                            className="cloud-link-button"
                                            type="button"
                                            onClick={() => void railwayLogout()}
                                        >
                                            Sign out
                                        </button>
                                    </div>
                                    {railway.workspaces.length > 1 && (
                                        <WorkspacePicker
                                            railway={railway}
                                            value={workspaceId}
                                            disabled={railwayBusy !== null}
                                            onChange={setWorkspaceId}
                                        />
                                    )}
                                    {railway.workspaces.length === 0 && (
                                        <div className="callout warning">
                                            No Railway workspace is available
                                            for this account.
                                        </div>
                                    )}
                                    <div className="cloud-actions">
                                        <button
                                            className="action-button primary"
                                            type="button"
                                            disabled={
                                                !workspaceId ||
                                                railwayBusy !== null
                                            }
                                            onClick={() => void deploy()}
                                        >
                                            Deploy control plane
                                        </button>
                                    </div>
                                    {deployResult && !deployResult.ok && (
                                        <div
                                            className={`callout ${deployResult.ok ? "success" : "error"}`}
                                            role="status"
                                        >
                                            {deployResult.message}
                                        </div>
                                    )}
                                </div>
                            )}
                            {destroyed && (
                                <div className="callout">
                                    Deployment removed. AgentField is using the
                                    local control plane.
                                </div>
                            )}
                            {railway?.message && (
                                <div className="callout warning">{railway.message}</div>
                            )}
                            <span className="cloud-railway-footnote">
                                Powered by bundled OpenTofu.{" "}
                                <button
                                    className="cloud-link-button"
                                    type="button"
                                    onClick={() =>
                                        void window.agentfield.cloudDeployRailway()
                                    }
                                >
                                    Use the Railway template instead
                                </button>
                            </span>
                        </div>
                    </div>
                </section>
            )}
        </>
    );
}

function CloudTestFeedback({ result }: { result: CloudTestResult }) {
    const success = result.ok && result.installApi;
    const degraded = result.ok && !result.installApi;
    const state = success ? "success" : degraded ? "warning" : "error";
    const heading = success
        ? `✓ Connected${result.version ? ` — control plane v${result.version.replace(/^v/, "")}` : ""}`
        : degraded
          ? "⚠ Connected, but this control plane is too old for desktop agent management — update the AgentField server, then test again."
          : result.message;

    const checks: Array<{
        label: string;
        state: "passed" | "warning" | "failed" | "pending";
    }> = [
        { label: "Reachable", state: result.healthy ? "passed" : "failed" },
        {
            label: "API key accepted",
            state: result.authOk
                ? "passed"
                : result.healthy
                  ? "failed"
                  : "pending",
        },
        {
            label: "Agent management available",
            state: result.installApi
                ? "passed"
                : degraded
                  ? "warning"
                  : result.authOk
                    ? "failed"
                    : "pending",
        },
        {
            label: result.furrowReported
                ? "Workspace sync"
                : "Workspace sync — not reported by this server version",
            state: result.furrowReported
                ? result.furrowAvailable
                    ? "passed"
                    : "failed"
                : "pending",
        },
    ];

    return (
        <div
            className={`callout ${state} cloud-result`}
            role={success ? "status" : "alert"}
        >
            <div className="cloud-result-heading">{heading}</div>
            <ul className="cloud-checks">
                {checks.map((check) => (
                    <li key={check.label} className={check.state}>
                        <span className="cloud-check-icon" aria-hidden="true">
                            {check.state === "passed"
                                ? "✓"
                                : check.state === "warning"
                                  ? "⚠"
                                  : check.state === "failed"
                                    ? "✗"
                                    : "—"}
                        </span>
                        <span>{check.label}</span>
                    </li>
                ))}
            </ul>
        </div>
    );
}

function displayHost(serverUrl: string) {
    try {
        return new URL(serverUrl).host;
    } catch {
        return serverUrl;
    }
}

function cloudPlatformLabel(platform: "railway" | "docker" | "local" | undefined) {
    if (platform === "railway") return "Railway";
    if (platform === "docker") return "Docker";
    if (platform === "local") return "Local";
    return "hosting unknown";
}

function cloudUpdateLabel(status: CloudUpdateStatus) {
    if (status.status === "current") return "up to date";
    if (status.status === "available") return `v${status.latest} available`;
    if (status.status === "legacy") return "update status unavailable";
    return "update check unavailable";
}

function WorkspacePicker({
    railway,
    value,
    disabled,
    onChange,
}: {
    railway: RailwayStatus;
    value: string;
    disabled: boolean;
    onChange: (workspaceId: string) => void;
}) {
    const recordedWorkspaceMissing =
        railway.hasDeployment &&
        value !== "" &&
        !railway.workspaces.some((workspace) => workspace.id === value);
    return (
        <label className="cloud-workspace-field">
            <span className="row-sub">Railway workspace</span>
            <select
                className="env-input cloud-input"
                value={value}
                disabled={disabled}
                onChange={(event) => onChange(event.target.value)}
            >
                <option value="">Choose a workspace…</option>
                {recordedWorkspaceMissing && (
                    <option value={value}>Deployment workspace ({value})</option>
                )}
                {railway.workspaces.map((workspace) => (
                    <option key={workspace.id} value={workspace.id}>
                        {workspace.name}
                    </option>
                ))}
            </select>
        </label>
    );
}
