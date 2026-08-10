import { useEffect, useState } from "react";
import type {
    CloudDeployResult,
    CloudImageUpdate,
    CloudTestResult,
    DesktopSettings,
    RailwayStatus,
} from "../../../shared/types";

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
    const [cloudImageUpdate, setCloudImageUpdate] =
        useState<CloudImageUpdate | null>(null);
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
        if (!railway?.loggedIn) {
            setWorkspaceId("");
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
        if (activeTab !== "railway" || !railway?.hasDeployment) {
            setCloudImageUpdate(null);
            return;
        }
        void window.agentfield
            .checkCloudImageUpdate()
            .then(setCloudImageUpdate)
            .catch(() => setCloudImageUpdate(null));
    }, [activeTab, railway?.hasDeployment]);

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
            const next = await window.agentfield.setSettings({
                cloud: {
                    enabled: true,
                    serverUrl: serverUrl.trim(),
                    apiKey: apiKey.trim(),
                },
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
            const next = await window.agentfield.setSettings({
                cloud: {
                    enabled: false,
                    serverUrl: serverUrl.trim(),
                    apiKey: apiKey.trim(),
                },
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
        if (!workspaceId) return;
        setRailwayBusy("deploy");
        setDeployLines([]);
        setDeployResult(null);
        setDestroyed(false);
        setError(null);
        try {
            const nextResult = await window.agentfield.cloudDeploy(workspaceId);
            setDeployResult(nextResult);
            if (nextResult.ok) {
                const next = await window.agentfield.getSettings();
                setSettings(next);
                setServerUrl(next.cloud.serverUrl);
                setApiKey(next.cloud.apiKey);
            }
            await refreshRailway();
            setCloudImageUpdate(
                await window.agentfield
                    .checkCloudImageUpdate()
                    .catch(() => null),
            );
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
            setCloudImageUpdate(null);
            setDestroyed(true);
            await refreshRailway();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setRailwayBusy(null);
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
                                    {cloudImageUpdate?.updateAvailable && (
                                        <div
                                            className="callout warning"
                                            role="status"
                                        >
                                            Control plane{" "}
                                            {cloudImageUpdate.current} →{" "}
                                            {cloudImageUpdate.latest} available
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
                                            {cloudImageUpdate?.updateAvailable
                                                ? "Upgrade & redeploy"
                                                : "Re-run deploy"}
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
                                        <label className="cloud-workspace-field">
                                            <span className="row-sub">
                                                Railway workspace
                                            </span>
                                            <select
                                                className="env-input cloud-input"
                                                value={workspaceId}
                                                disabled={railwayBusy !== null}
                                                onChange={(event) =>
                                                    setWorkspaceId(
                                                        event.target.value,
                                                    )
                                                }
                                            >
                                                <option value="">
                                                    Choose a workspace…
                                                </option>
                                                {railway.workspaces.map(
                                                    (workspace) => (
                                                        <option
                                                            key={workspace.id}
                                                            value={workspace.id}
                                                        >
                                                            {workspace.name}
                                                        </option>
                                                    ),
                                                )}
                                            </select>
                                        </label>
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
