package handlers

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// orphanReapStorageStub captures MarkAgentExecutionsOrphaned calls so a
// handler-level test can pin the exact (agent_id, reason) pair the registration
// path emits when an instance change is detected. Embeds nodeRESTStorageStub
// so it inherits all the standard agent CRUD plumbing (RegisterAgent,
// GetAgent, GetAgentVersion, etc.) used by RegisterNodeHandler.
type orphanReapStorageStub struct {
	nodeRESTStorageStub
	orphanCalls []orphanReapCall
}

type orphanReapCall struct {
	agentNodeID string
	reason      string
}

func (s *orphanReapStorageStub) MarkAgentExecutionsOrphaned(_ context.Context, agentNodeID, reasonMessage string) (int, error) {
	s.orphanCalls = append(s.orphanCalls, orphanReapCall{agentNodeID: agentNodeID, reason: reasonMessage})
	return len(s.orphanCalls), nil // pretend we reaped one row per call
}

// registerNodeWithBody is a tiny test helper to keep the body construction out
// of the test bodies — the registration endpoint is verbose.
func registerNodeWithBody(t *testing.T, router *gin.Engine, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/nodes/register", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

// TestRegisterNodeHandler_ReapsOrphansOnInstanceChange is the load-bearing
// functional test for the redeploy-orphan fix. It simulates the exact
// production scenario from run_1778004368903_9345a88f:
//
//  1. Agent boots with instance "alpha", registers, and (notionally) starts a
//     long-running cross-agent call. The call is `running` in the DAG.
//  2. The Python process is killed by a redeploy. The new process starts with
//     a fresh instance "beta" and re-registers.
//  3. The registration handler must detect alpha != beta and call
//     MarkAgentExecutionsOrphaned so the DAG doesn't leave the parent
//     reasoner stuck in `running` forever.
//
// Without this test the regression risk is severe: a refactor that drops the
// instance comparison would silently restore the "stuck reasoner" bug, and the
// bug only surfaces in production after a redeploy lands during real traffic.
func TestRegisterNodeHandler_ReapsOrphansOnInstanceChange(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &orphanReapStorageStub{
		nodeRESTStorageStub: nodeRESTStorageStub{
			agent: &types.AgentNode{
				ID:              "github-buddy",
				BaseURL:         "http://10.0.0.5:8080",
				LifecycleStatus: types.AgentStatusReady,
				HealthStatus:    types.HealthStatusActive,
				InstanceID:      "alpha", // previous OS process
			},
		},
	}
	router := gin.New()
	router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

	// New process registering with a different instance_id — same agent id,
	// fresh PID, fresh UUID. This is the exact shape the SDK emits after a
	// graceful SIGTERM redeploy.
	body := `{
		"id":"github-buddy",
		"base_url":"http://10.0.0.5:8080",
		"instance_id":"beta",
		"callback_discovery":{"mode":"manual","preferred":"http://10.0.0.5:8080"}
	}`

	rec := registerNodeWithBody(t, router, body)
	require.Equal(t, http.StatusCreated, rec.Code,
		"re-registration with new instance must succeed; instance mismatch is signal, not error")

	require.Len(t, store.orphanCalls, 1,
		"exactly one orphan-reap must fire when stored instance differs from incoming")
	assert.Equal(t, "github-buddy", store.orphanCalls[0].agentNodeID,
		"reap must target the re-registering agent only")
	assert.Contains(t, store.orphanCalls[0].reason, "agent_restart_orphaned",
		"reason must lead with the audit token operators grep for")
	assert.Contains(t, store.orphanCalls[0].reason, "alpha",
		"reason must record the old instance for forensics")
	assert.Contains(t, store.orphanCalls[0].reason, "beta",
		"reason must record the new instance for forensics")
}

// TestRegisterNodeHandler_NoReapOnSameInstance pins that an idempotent
// re-registration (same instance_id arriving twice — e.g., the connection
// manager's reconnect loop firing after a brief network blip) does NOT trigger
// a reap. This is the most likely false-positive: the agent is fine, just
// reconnecting; reaping its in-flight work would be exactly the bug we're
// fixing, in reverse.
func TestRegisterNodeHandler_NoReapOnSameInstance(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &orphanReapStorageStub{
		nodeRESTStorageStub: nodeRESTStorageStub{
			agent: &types.AgentNode{
				ID:              "github-buddy",
				BaseURL:         "http://10.0.0.5:8080",
				LifecycleStatus: types.AgentStatusReady,
				HealthStatus:    types.HealthStatusActive,
				InstanceID:      "alpha",
			},
		},
	}
	router := gin.New()
	router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

	body := `{
		"id":"github-buddy",
		"base_url":"http://10.0.0.5:8080",
		"instance_id":"alpha",
		"callback_discovery":{"mode":"manual","preferred":"http://10.0.0.5:8080"}
	}`

	rec := registerNodeWithBody(t, router, body)
	require.Equal(t, http.StatusCreated, rec.Code)
	assert.Empty(t, store.orphanCalls,
		"same instance_id on re-registration must NOT trigger orphan reap "+
			"(otherwise every reconnect would nuke in-flight work)")
}

// TestRegisterNodeHandler_NoReapOnFirstRegistration covers the boot path: a
// brand-new agent ID has no stored instance. We must NOT reap anything (the
// agent has no prior in-flight executions to orphan, and reaping would be a
// no-op at best — but the principle matters: only an *instance change* on an
// existing agent is the orphan signal).
func TestRegisterNodeHandler_NoReapOnFirstRegistration(t *testing.T) {
	gin.SetMode(gin.TestMode)
	// agent: nil → GetAgent returns nil, nil → isReRegistration is false.
	store := &orphanReapStorageStub{}
	router := gin.New()
	router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

	body := `{
		"id":"brand-new-agent",
		"base_url":"http://10.0.0.5:8080",
		"instance_id":"alpha",
		"callback_discovery":{"mode":"manual","preferred":"http://10.0.0.5:8080"}
	}`

	rec := registerNodeWithBody(t, router, body)
	require.Equal(t, http.StatusCreated, rec.Code)
	assert.Empty(t, store.orphanCalls,
		"first-time registration has nothing to orphan; reap must not fire")
}

// TestRegisterNodeHandler_NoReapOnLegacySDK pins backward compatibility: an
// agent on an older SDK that doesn't send instance_id (empty string) must
// continue to work without triggering a reap. The previous behavior is
// preserved exactly; the new protection is opt-in by sending a non-empty
// instance_id. Without this guard, every old-SDK reconnect would falsely
// reap its own in-flight work.
func TestRegisterNodeHandler_NoReapOnLegacySDK(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &orphanReapStorageStub{
		nodeRESTStorageStub: nodeRESTStorageStub{
			agent: &types.AgentNode{
				ID:              "old-agent",
				BaseURL:         "http://10.0.0.5:8080",
				LifecycleStatus: types.AgentStatusReady,
				HealthStatus:    types.HealthStatusActive,
				InstanceID:      "", // never sent — pre-fix SDK
			},
		},
	}
	router := gin.New()
	router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

	// Older SDK: no instance_id field in payload.
	body := `{
		"id":"old-agent",
		"base_url":"http://10.0.0.5:8080",
		"callback_discovery":{"mode":"manual","preferred":"http://10.0.0.5:8080"}
	}`

	rec := registerNodeWithBody(t, router, body)
	require.Equal(t, http.StatusCreated, rec.Code)
	assert.Empty(t, store.orphanCalls,
		"legacy SDK (no instance_id) must keep working without spurious reaps")
}

// TestRegisterNodeHandler_PersistsInstanceID confirms the registration handler
// actually writes the new instance_id to the stored agent record. Without
// this, the comparison on the *next* re-registration would always read empty
// and never detect a restart — the whole mechanism would be broken.
func TestRegisterNodeHandler_PersistsInstanceID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &orphanReapStorageStub{}
	router := gin.New()
	router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

	body := `{
		"id":"github-buddy",
		"base_url":"http://10.0.0.5:8080",
		"instance_id":"alpha",
		"callback_discovery":{"mode":"manual","preferred":"http://10.0.0.5:8080"}
	}`
	rec := registerNodeWithBody(t, router, body)
	require.Equal(t, http.StatusCreated, rec.Code)
	require.NotNil(t, store.registeredAgent)
	assert.Equal(t, "alpha", store.registeredAgent.InstanceID,
		"instance_id must be propagated through to the stored agent record")
}

// TestRegisterNodeHandler_InstanceChangeWithNothingToReap covers the
// realistic "agent restarted but had no in-flight work" branch: an instance
// flip is detected, the reap query runs, and zero rows match. The handler
// must take the "nothing to reap" log branch (rather than the success or
// error branches) and the registration must succeed normally. Without this
// case, the most common production trace — restart of an idle agent — would
// be uncovered.
func TestRegisterNodeHandler_InstanceChangeWithNothingToReap(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &zeroReapStorageStub{
		orphanReapStorageStub: orphanReapStorageStub{
			nodeRESTStorageStub: nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "github-buddy",
					BaseURL:         "http://10.0.0.5:8080",
					LifecycleStatus: types.AgentStatusReady,
					HealthStatus:    types.HealthStatusActive,
					InstanceID:      "alpha",
				},
			},
		},
	}
	router := gin.New()
	router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

	body := `{
		"id":"github-buddy",
		"base_url":"http://10.0.0.5:8080",
		"instance_id":"beta",
		"callback_discovery":{"mode":"manual","preferred":"http://10.0.0.5:8080"}
	}`
	rec := registerNodeWithBody(t, router, body)
	require.Equal(t, http.StatusCreated, rec.Code,
		"instance change with nothing to reap is the common idle-agent "+
			"restart case; registration must complete normally")
	require.Len(t, store.orphanCalls, 1,
		"the reap is still invoked — the handler can't know in advance "+
			"whether there's anything to reap")
	require.NotNil(t, store.registeredAgent)
	assert.Equal(t, "beta", store.registeredAgent.InstanceID)
}

// zeroReapStorageStub returns 0 from MarkAgentExecutionsOrphaned, simulating
// a successful reap query that matched no rows.
type zeroReapStorageStub struct {
	orphanReapStorageStub
}

func (s *zeroReapStorageStub) MarkAgentExecutionsOrphaned(_ context.Context, agentNodeID, reasonMessage string) (int, error) {
	s.orphanCalls = append(s.orphanCalls, orphanReapCall{agentNodeID: agentNodeID, reason: reasonMessage})
	return 0, nil
}

// TestRegisterNodeHandlerReapFailureDoesNotBlockRegistration guards the
// best-effort error path. The reap is logged-and-continue precisely so a
// transient DB hiccup during cleanup never wedges agent registration —
// without this, a fragile reap could brick the entire restart flow.
func TestRegisterNodeHandlerReapFailureDoesNotBlockRegistration(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &reapFailingStorageStub{
		orphanReapStorageStub: orphanReapStorageStub{
			nodeRESTStorageStub: nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "github-buddy",
					BaseURL:         "http://10.0.0.5:8080",
					LifecycleStatus: types.AgentStatusReady,
					HealthStatus:    types.HealthStatusActive,
					InstanceID:      "alpha",
				},
			},
		},
	}
	router := gin.New()
	router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

	body := `{
		"id":"github-buddy",
		"base_url":"http://10.0.0.5:8080",
		"instance_id":"beta",
		"callback_discovery":{"mode":"manual","preferred":"http://10.0.0.5:8080"}
	}`
	rec := registerNodeWithBody(t, router, body)

	// Critical: registration must still succeed even though reap returned an
	// error. The agent is functionally re-registered; the existing 30-min
	// stale-execution sweep is the safety net that catches the missed reap.
	require.Equal(t, http.StatusCreated, rec.Code,
		"reap failure must not block re-registration — graceful degradation")
	require.NotNil(t, store.registeredAgent)
	assert.Equal(t, "beta", store.registeredAgent.InstanceID,
		"new instance_id must persist even when reap errored")
}

// reapFailingStorageStub is orphanReapStorageStub but every reap call returns
// an error, so we can prove the registration path tolerates failures cleanly.
type reapFailingStorageStub struct {
	orphanReapStorageStub
}

func (s *reapFailingStorageStub) MarkAgentExecutionsOrphaned(ctx context.Context, agentNodeID, reasonMessage string) (int, error) {
	return 0, assertErrFakeReapFailure
}

// Sentinel error so the test can be specific about what failed.
var assertErrFakeReapFailure = &fakeReapError{}

type fakeReapError struct{}

func (*fakeReapError) Error() string { return "fake reap failure (expected in test)" }
