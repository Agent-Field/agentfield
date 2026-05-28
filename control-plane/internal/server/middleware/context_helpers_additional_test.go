package middleware

import (
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestMemoryContextHelpers(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("caller agent id", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		require.Empty(t, GetCallerAgentID(c))

		c.Set(string(CallerAgentIDKey), 42)
		require.Empty(t, GetCallerAgentID(c))

		c.Set(string(CallerAgentIDKey), "agent-123")
		require.Equal(t, "agent-123", GetCallerAgentID(c))
	})

	t.Run("memory permission result", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		require.Nil(t, GetMemoryPermissionResult(c))

		c.Set(string(MemoryPermissionResultKey), "wrong-type")
		require.Nil(t, GetMemoryPermissionResult(c))

		result := &MemoryPermissionResult{Allowed: true, Reason: "ok", CallerID: "agent-123"}
		c.Set(string(MemoryPermissionResultKey), result)
		require.Same(t, result, GetMemoryPermissionResult(c))
	})
}

func TestPermissionContextHelpers(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("permission result", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		require.Nil(t, GetPermissionCheckResult(c))

		c.Set(string(PermissionCheckResultKey), "wrong-type")
		require.Nil(t, GetPermissionCheckResult(c))

		result := &PermissionCheckResult{Allowed: true, RequiresPermission: true}
		c.Set(string(PermissionCheckResultKey), result)
		require.Same(t, result, GetPermissionCheckResult(c))
	})

	t.Run("target agent", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		require.Nil(t, GetTargetAgent(c))

		c.Set(string(TargetAgentKey), "wrong-type")
		require.Nil(t, GetTargetAgent(c))

		agent := &types.AgentNode{ID: "agent-456"}
		c.Set(string(TargetAgentKey), agent)
		require.Same(t, agent, GetTargetAgent(c))
	})

	t.Run("target did", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		require.Empty(t, GetTargetDID(c))

		c.Set(string(TargetDIDKey), 123)
		require.Empty(t, GetTargetDID(c))

		c.Set(string(TargetDIDKey), "did:web:example.com:agents:agent-456")
		require.Equal(t, "did:web:example.com:agents:agent-456", GetTargetDID(c))
	})
}
