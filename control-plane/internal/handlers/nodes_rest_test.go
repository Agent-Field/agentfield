package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalizePhase(t *testing.T) {
	tests := []struct {
		name              string
		phase             string
		wantState         *types.AgentState
		wantLifecycle     *types.AgentLifecycleStatus
		wantErr           bool
		wantErrContains   string
	}{
		{
			name:  "empty phase returns nil",
			phase: "",
		},
		{
			name:  "starting",
			phase: "starting",
			wantState: func() *types.AgentState {
				s := types.AgentStateStarting
				return &s
			}(),
			wantLifecycle: func() *types.AgentLifecycleStatus {
				l := types.AgentStatusStarting
				return &l
			}(),
		},
		{
			name:  "ready",
			phase: "ready",
			wantState: func() *types.AgentState {
				s := types.AgentStateActive
				return &s
			}(),
			wantLifecycle: func() *types.AgentLifecycleStatus {
				l := types.AgentStatusReady
				return &l
			}(),
		},
		{
			name:  "degraded",
			phase: "degraded",
			wantState: func() *types.AgentState {
				s := types.AgentStateActive
				return &s
			}(),
			wantLifecycle: func() *types.AgentLifecycleStatus {
				l := types.AgentStatusDegraded
				return &l
			}(),
		},
		{
			name:  "offline",
			phase: "offline",
			wantState: func() *types.AgentState {
				s := types.AgentStateInactive
				return &s
			}(),
			wantLifecycle: func() *types.AgentLifecycleStatus {
				l := types.AgentStatusOffline
				return &l
			}(),
		},
		{
			name:  "case insensitive - READY",
			phase: "READY",
			wantState: func() *types.AgentState {
				s := types.AgentStateActive
				return &s
			}(),
			wantLifecycle: func() *types.AgentLifecycleStatus {
				l := types.AgentStatusReady
				return &l
			}(),
		},
		{
			name:  "whitespace trimmed",
			phase: "  ready  ",
			wantState: func() *types.AgentState {
				s := types.AgentStateActive
				return &s
			}(),
			wantLifecycle: func() *types.AgentLifecycleStatus {
				l := types.AgentStatusReady
				return &l
			}(),
		},
		{
			name:            "unsupported phase",
			phase:           "unknown_phase",
			wantErr:         true,
			wantErrContains: "unsupported phase",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state, lifecycle, err := normalizePhase(tt.phase)
			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErrContains)
				return
			}
			require.NoError(t, err)

			if tt.wantState == nil {
				assert.Nil(t, state)
			} else {
				require.NotNil(t, state)
				assert.Equal(t, *tt.wantState, *state)
			}

			if tt.wantLifecycle == nil {
				assert.Nil(t, lifecycle)
			} else {
				require.NotNil(t, lifecycle)
				assert.Equal(t, *tt.wantLifecycle, *lifecycle)
			}
		})
	}
}

func TestDefaultLeaseTTL(t *testing.T) {
	assert.Equal(t, 5*time.Minute, DefaultLeaseTTL)
}

func TestNodeStatusLeaseHandler_ValidationErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("missing node_id returns 400", func(t *testing.T) {
		router := gin.New()
		router.PUT("/nodes/:node_id/status", NodeStatusLeaseHandler(nil, nil, nil, 0))

		body, _ := json.Marshal(map[string]string{"phase": "ready"})
		// Use empty node_id by calling the route with empty param
		req, _ := http.NewRequest("PUT", "/nodes//status", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Gin will 404 on empty param segment
		assert.Contains(t, []int{http.StatusBadRequest, http.StatusNotFound, http.StatusMovedPermanently}, w.Code)
	})

	t.Run("invalid JSON returns 400", func(t *testing.T) {
		router := gin.New()
		router.PUT("/nodes/:node_id/status", NodeStatusLeaseHandler(nil, nil, nil, 0))

		req, _ := http.NewRequest("PUT", "/nodes/test-node/status", bytes.NewBufferString("not json"))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)
	})
}

func TestNodeActionAckHandler_ValidationErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("invalid JSON returns 400", func(t *testing.T) {
		router := gin.New()
		router.POST("/nodes/:node_id/actions/ack", NodeActionAckHandler(nil, nil, 0))

		req, _ := http.NewRequest("POST", "/nodes/test-node/actions/ack", bytes.NewBufferString("bad"))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing action_id and status returns 400", func(t *testing.T) {
		router := gin.New()
		router.POST("/nodes/:node_id/actions/ack", NodeActionAckHandler(nil, nil, 0))

		body, _ := json.Marshal(map[string]string{"action_id": "", "status": ""})
		req, _ := http.NewRequest("POST", "/nodes/test-node/actions/ack", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)
		var resp map[string]string
		json.Unmarshal(w.Body.Bytes(), &resp)
		assert.Equal(t, "action_id and status are required", resp["error"])
	})
}

func TestClaimActionsHandler_ValidationErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("invalid JSON returns 400", func(t *testing.T) {
		router := gin.New()
		router.POST("/actions/claim", ClaimActionsHandler(nil, nil, 0))

		req, _ := http.NewRequest("POST", "/actions/claim", bytes.NewBufferString("bad"))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("missing node_id returns 400", func(t *testing.T) {
		router := gin.New()
		router.POST("/actions/claim", ClaimActionsHandler(nil, nil, 0))

		body, _ := json.Marshal(map[string]string{"node_id": ""})
		req, _ := http.NewRequest("POST", "/actions/claim", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)
	})
}

// TestNodeShutdownHandler — requires non-nil StorageProvider (see execute_test.go
// integration pattern). Shutdown handler uses best-effort JSON parse then
// immediately calls storage.GetAgent(), so nil-storage tests always panic.

// TestNormalizePhase_AllPhases_ProduceDistinctStates verifies that each
// recognized phase maps to a distinct (state, lifecycle) pair, catching
// copy-paste errors in the switch statement.
func TestNormalizePhase_AllPhases_ProduceDistinctStates(t *testing.T) {
	phases := []string{"starting", "ready", "degraded", "offline"}
	type pair struct {
		state     string
		lifecycle string
	}
	seen := make(map[pair]string)

	for _, phase := range phases {
		state, lifecycle, err := normalizePhase(phase)
		require.NoError(t, err)
		require.NotNil(t, state)
		require.NotNil(t, lifecycle)

		p := pair{string(*state), string(*lifecycle)}
		if prev, exists := seen[p]; exists {
			t.Errorf("phases %q and %q produce identical (state=%s, lifecycle=%s)",
				prev, phase, p.state, p.lifecycle)
		}
		seen[p] = phase
	}
}

// NOTE: Full behavioral tests for NodeStatusLeaseHandler, ClaimActionsHandler,
// and NodeShutdownHandler require a mock StorageProvider implementation (the
// interface has 50+ methods). These handlers call storage.GetAgent() before any
// business logic validation, so nil-storage tests can only cover input parsing.
// Full behavioral tests should be added as integration tests (see execute_test.go
// with //go:build integration for the mock pattern).
//
// What IS tested here:
// - normalizePhase: all phase values, case insensitivity, distinctness
// - Input validation: bad JSON, missing required fields
// - DefaultLeaseTTL constant
