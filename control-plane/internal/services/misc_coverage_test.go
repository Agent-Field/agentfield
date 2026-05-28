package services

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestDIDServiceGetControlPlaneIssuerDID(t *testing.T) {
	service, registry, _, _, agentfieldID := setupDIDTestEnvironment(t)

	issuerDID, err := service.GetControlPlaneIssuerDID()
	require.NoError(t, err)
	require.NotEmpty(t, issuerDID)

	loaded, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)
	loaded.RootDID = ""
	registry.registries[agentfieldID] = loaded

	_, err = service.GetControlPlaneIssuerDID()
	require.ErrorContains(t, err, "root DID not initialized")

	service.config.Enabled = false
	_, err = service.GetControlPlaneIssuerDID()
	require.ErrorContains(t, err, "DID system is disabled")
}

func TestExecutionMetricsReservedHelpers(t *testing.T) {
	recordWorkerAcquire(" agent-1 ")
	recordWorkerRelease(" agent-1 ")
	observeStepDuration("completed", 2*time.Second)
	incrementStepRetry(" agent-1 ")
	incrementBackpressure("")
}

func TestCloneAgentStatus(t *testing.T) {
	lastVerified := time.Now().UTC()
	original := &types.AgentStatus{
		State: types.AgentStateActive,
		StateTransition: &types.StateTransition{
			From:      types.AgentStateStarting,
			To:        types.AgentStateActive,
			StartedAt: lastVerified,
		},
		LastVerified: &lastVerified,
	}

	cloned := cloneAgentStatus(original)
	require.NotSame(t, original, cloned)
	require.NotSame(t, original.StateTransition, cloned.StateTransition)
	require.NotSame(t, original.LastVerified, cloned.LastVerified)
	require.Nil(t, cloneAgentStatus(nil))
}

func TestLLMHealthMonitorLifecycleHelpers(t *testing.T) {
	require.False(t, (*LLMHealthMonitor)(nil).Enabled())
	require.Zero(t, (*LLMHealthMonitor)(nil).EndpointCount())

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	disabled := NewLLMHealthMonitor(config.LLMHealthConfig{Enabled: false}, nil)
	require.False(t, disabled.Enabled())
	disabled.Start()
	disabled.Stop()

	monitor := NewLLMHealthMonitor(config.LLMHealthConfig{
		Enabled:           true,
		CheckInterval:     10 * time.Millisecond,
		CheckTimeout:      100 * time.Millisecond,
		FailureThreshold:  1,
		RecoveryTimeout:   10 * time.Millisecond,
		HalfOpenMaxProbes: 1,
		Endpoints: []config.LLMEndpoint{
			{Name: "Primary", URL: server.URL},
		},
	}, nil)

	require.True(t, monitor.Enabled())
	require.Equal(t, 1, monitor.EndpointCount())

	done := make(chan struct{})
	go func() {
		monitor.Start()
		close(done)
	}()

	time.Sleep(30 * time.Millisecond)
	monitor.Stop()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("monitor did not stop")
	}
}
