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
	"github.com/stretchr/testify/require"
)

func TestStartSessionHandlerCreatesSessionForRegisteredDefinition(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &nodeRESTStorageStub{agent: sessionTestAgent()}
	router := gin.New()
	router.POST("/api/v1/session-targets/:target/start", StartSessionHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/session-targets/support.voice/start", bytes.NewBufferString(`{"provider":"openai","transport":"webrtc"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusCreated, rec.Code)
	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.Equal(t, "support.voice", body["target"])
	require.Equal(t, "openai", body["provider"])
	require.Equal(t, "webrtc", body["transport"])
	require.Equal(t, []interface{}{"voice", "pii"}, body["tags"])
	require.Equal(t, map[string]interface{}{"resolve_voice_turn": "support.resolve_voice_turn"}, body["tool_targets"])
}

func TestStartSessionHandlerRejectsUnsupportedTransport(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &nodeRESTStorageStub{agent: sessionTestAgent()}
	router := gin.New()
	router.POST("/api/v1/session-targets/:target/start", StartSessionHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/session-targets/support.voice/start", bytes.NewBufferString(`{"provider":"openrouter","transport":"webrtc"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	require.Contains(t, rec.Body.String(), "does not infer or switch providers")
}

func TestSessionRoutesRegisterTogether(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &nodeRESTStorageStub{agent: sessionTestAgent()}

	require.NotPanics(t, func() {
		router := gin.New()
		group := router.Group("/api/v1/sessions")
		group.POST("/:target/start", StartSessionHandler(store))
		group.POST("/:target/realtime-offer", SessionRealtimeOfferHandler(store))
		group.POST("/:target/tools/:tool", SessionToolHandler(store, time.Second, ""))
	})
}

func sessionTestAgent() *types.AgentNode {
	return &types.AgentNode{
		ID: "support",
		Metadata: types.AgentMetadata{Custom: map[string]interface{}{
			"sessions": []interface{}{
				map[string]interface{}{
					"name":          "voice",
					"provider":      "openai",
					"transport":     "webrtc",
					"model":         "gpt-realtime-2",
					"modalities":    []interface{}{"audio", "text"},
					"tools":         []interface{}{"support.resolve_voice_turn"},
					"tags":          []interface{}{"voice", "pii"},
					"approved_tags": []interface{}{"voice", "pii"},
				},
			},
		}},
	}
}
