package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
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

func TestStartSessionHandlerRejectsInvalidOrMissingDefinitions(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name    string
		store   *nodeRESTStorageStub
		path    string
		status  int
		message string
	}{
		{
			name:    "bad target",
			store:   &nodeRESTStorageStub{agent: sessionTestAgent()},
			path:    "/api/v1/sessions/support/start",
			status:  http.StatusBadRequest,
			message: "session target must be",
		},
		{
			name:    "missing agent",
			store:   &nodeRESTStorageStub{},
			path:    "/api/v1/sessions/support.voice/start",
			status:  http.StatusNotFound,
			message: "agent not found",
		},
		{
			name:    "missing metadata",
			store:   &nodeRESTStorageStub{agent: &types.AgentNode{ID: "support"}},
			path:    "/api/v1/sessions/support.voice/start",
			status:  http.StatusNotFound,
			message: "agent has no registered sessions",
		},
		{
			name: "missing sessions list",
			store: &nodeRESTStorageStub{agent: &types.AgentNode{
				ID:       "support",
				Metadata: types.AgentMetadata{Custom: map[string]interface{}{"sessions": "bad"}},
			}},
			path:    "/api/v1/sessions/support.voice/start",
			status:  http.StatusNotFound,
			message: "agent has no registered sessions",
		},
		{
			name:    "unknown session",
			store:   &nodeRESTStorageStub{agent: sessionTestAgent()},
			path:    "/api/v1/sessions/support.chat/start",
			status:  http.StatusNotFound,
			message: "session not registered",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/api/v1/sessions/:target/start", StartSessionHandler(tt.store))

			req := httptest.NewRequest(http.MethodPost, tt.path, bytes.NewBufferString(`{}`))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, tt.status, rec.Code)
			require.Contains(t, rec.Body.String(), tt.message)
		})
	}
}

func TestStartSessionHandlerRejectsCapabilityOverrideMismatch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &nodeRESTStorageStub{agent: sessionTestAgent()}
	router := gin.New()
	router.POST("/api/v1/sessions/:target/start", StartSessionHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/support.voice/start", bytes.NewBufferString(`{"provider":"openai","transport":"websocket"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	require.Contains(t, rec.Body.String(), "registered for provider=openai transport=webrtc")
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

func TestSessionRealtimeOfferHandlerValidatesInputs(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &nodeRESTStorageStub{}

	tests := []struct {
		name    string
		path    string
		body    string
		status  int
		message string
		envKey  string
		custom  bool
	}{
		{
			name:    "missing provider",
			path:    "/api/v1/sessions/sess-1/realtime-offer",
			body:    `{}`,
			status:  http.StatusBadRequest,
			message: "session provider is required",
		},
		{
			name:    "non webrtc transport",
			path:    "/api/v1/sessions/sess-1/realtime-offer?provider=openai&transport=websocket",
			body:    "v=0",
			status:  http.StatusBadRequest,
			message: "requires transport=webrtc",
		},
		{
			name:    "non openai provider",
			path:    "/api/v1/sessions/sess-1/realtime-offer?provider=custom&transport=webrtc",
			body:    "v=0",
			status:  http.StatusBadRequest,
			message: "require provider=openai",
			custom:  true,
		},
		{
			name:    "missing key",
			path:    "/api/v1/sessions/sess-1/realtime-offer?provider=openai&transport=webrtc",
			body:    "v=0",
			status:  http.StatusBadGateway,
			message: "OPENAI_API_KEY is required",
		},
		{
			name:    "missing sdp",
			path:    "/api/v1/sessions/sess-1/realtime-offer?provider=openai&transport=webrtc",
			body:    " ",
			status:  http.StatusBadRequest,
			message: "SDP offer body is required",
			envKey:  "test-key",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.custom {
				original := types.SupportedSessionTransports["custom"]
				types.SupportedSessionTransports["custom"] = []string{"webrtc"}
				t.Cleanup(func() {
					if original == nil {
						delete(types.SupportedSessionTransports, "custom")
					} else {
						types.SupportedSessionTransports["custom"] = original
					}
				})
			}
			if tt.envKey != "" {
				t.Setenv("OPENAI_API_KEY", tt.envKey)
			} else {
				t.Setenv("OPENAI_API_KEY", "")
			}
			router := gin.New()
			router.POST("/api/v1/sessions/:target/realtime-offer", SessionRealtimeOfferHandler(store))

			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, tt.status, rec.Code)
			require.Contains(t, rec.Body.String(), tt.message)
		})
	}
}

func TestSessionRealtimeOfferHandlerCallsRealtimeProvider(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OPENAI_API_KEY", "test-key")
	originalTransport := http.DefaultClient.Transport
	t.Cleanup(func() { http.DefaultClient.Transport = originalTransport })

	var gotSafetyID string
	var gotSession string
	http.DefaultClient.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		require.Equal(t, "https://api.openai.com/v1/realtime/calls", req.URL.String())
		require.Equal(t, "Bearer test-key", req.Header.Get("Authorization"))
		gotSafetyID = req.Header.Get("OpenAI-Safety-Identifier")

		reader, err := req.MultipartReader()
		require.NoError(t, err)
		for {
			part, err := reader.NextPart()
			if err == io.EOF {
				break
			}
			require.NoError(t, err)
			data, err := io.ReadAll(part)
			require.NoError(t, err)
			if part.FormName() == "session" {
				gotSession = string(data)
			}
		}

		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader("v=0\r\nanswer\r\n")),
		}, nil
	})

	router := gin.New()
	router.POST("/api/v1/sessions/:target/realtime-offer", SessionRealtimeOfferHandler(&nodeRESTStorageStub{}))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/sess-1/realtime-offer?provider=openai&transport=webrtc&model=gpt-test&voice=cedar", strings.NewReader("v=0\r\noffer\r\n"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "application/sdp", rec.Header().Get("Content-Type"))
	require.Equal(t, "v=0\r\nanswer\r\n", rec.Body.String())
	require.Len(t, gotSafetyID, 32)
	require.Contains(t, gotSession, `"model":"gpt-test"`)
	require.Contains(t, gotSession, `"voice":"cedar"`)
	var config map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(gotSession), &config))
	input := config["audio"].(map[string]interface{})["input"].(map[string]interface{})
	require.Equal(t, true, input["turn_detection"].(map[string]interface{})["interrupt_response"])
}

func TestSessionRealtimeOfferHandlerSurfacesProviderErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OPENAI_API_KEY", "test-key")
	originalTransport := http.DefaultClient.Transport
	t.Cleanup(func() { http.DefaultClient.Transport = originalTransport })
	http.DefaultClient.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadRequest,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader("bad offer")),
		}, nil
	})

	router := gin.New()
	router.POST("/api/v1/sessions/:target/realtime-offer", SessionRealtimeOfferHandler(&nodeRESTStorageStub{}))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/sess-1/realtime-offer?provider=openai&transport=webrtc", strings.NewReader("v=0"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusBadGateway, rec.Code)
	require.Contains(t, rec.Body.String(), "OpenAI realtime call failed")
}

func TestSessionToolHandlerValidatesAndForwards(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	var gotSessionID string
	var gotAuth string
	var gotBody map[string]interface{}
	router.POST("/api/v1/execute/async/:target", func(c *gin.Context) {
		gotSessionID = c.GetHeader("X-Session-ID")
		gotAuth = c.GetHeader("Authorization")
		require.NoError(t, c.ShouldBindJSON(&gotBody))
		require.Equal(t, "support.resolve", c.Param("target"))
		c.JSON(http.StatusAccepted, gin.H{"execution_id": "exec-1"})
	})
	router.POST("/api/v1/sessions/:target/tools/:tool", SessionToolHandler(&nodeRESTStorageStub{}, time.Second, "internal-token"))
	server := httptest.NewServer(router)
	t.Cleanup(server.Close)

	req, err := http.NewRequest(http.MethodPost, server.URL+"/api/v1/sessions/sess-1/tools/resolve", strings.NewReader(`{"target":"support.resolve","input":{"topic":"billing"}}`))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	resp, err := server.Client().Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	require.Equal(t, http.StatusAccepted, resp.StatusCode)
	require.JSONEq(t, `{"execution_id":"exec-1"}`, string(body))
	require.Equal(t, "sess-1", gotSessionID)
	require.Equal(t, "Bearer internal-token", gotAuth)
	require.Equal(t, map[string]interface{}{"input": map[string]interface{}{"topic": "billing"}}, gotBody)
}

func TestSessionToolHandlerRejectsInvalidRequests(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/api/v1/sessions/:target/tools/:tool", SessionToolHandler(&nodeRESTStorageStub{}, time.Second, ""))

	tests := []struct {
		name    string
		body    string
		message string
	}{
		{name: "bad json", body: "{", message: "invalid JSON body"},
		{name: "bad target", body: `{"target":"resolve"}`, message: "session tool target must be"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/sess-1/tools/resolve", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, http.StatusBadRequest, rec.Code)
			require.Contains(t, rec.Body.String(), tt.message)
		})
	}
}

func TestSessionHelpers(t *testing.T) {
	node, session, ok := splitSessionTarget(" support.voice ")
	require.True(t, ok)
	require.Equal(t, "support", node)
	require.Equal(t, "voice", session)

	_, _, ok = splitSessionTarget("support")
	require.False(t, ok)
	require.Equal(t, "fallback", firstNonEmptySession(" ", "fallback"))
	require.Equal(t, map[string]string{
		"local":   "support.local",
		"resolve": "support.resolve",
	}, sessionToolTargets("support", []string{"", "local", "support.resolve"}))
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
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

// Exercise the complete metadata -> start -> offer -> provider boundary, rather
// than just checking that a new field exists in the registration response.
func TestSessionTurnDetectionReachesProvider(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OPENAI_API_KEY", "test-key")
	original := http.DefaultClient.Transport
	t.Cleanup(func() { http.DefaultClient.Transport = original })
	for _, tc := range []struct{ name, config, expected string }{
		{"legacy defaults", "", `{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,"silence_duration_ms":500,"create_response":true,"interrupt_response":true}`},
		{"custom server", `{"type":"server_vad","threshold":0,"prefix_padding_ms":0,"silence_duration_ms":750,"create_response":false,"interrupt_response":false}`, `{"type":"server_vad","threshold":0,"prefix_padding_ms":0,"silence_duration_ms":750,"create_response":false,"interrupt_response":false}`},
		{"semantic", `{"type":"semantic_vad","eagerness":"low"}`, `{"type":"semantic_vad","eagerness":"low","create_response":true,"interrupt_response":true}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			agent := sessionTestAgent()
			raw := agent.Metadata.Custom["sessions"].([]interface{})[0].(map[string]interface{})
			raw["model"], raw["voice"] = "gpt-vad-test", "cedar"
			if tc.config != "" {
				raw["turn_detection"] = json.RawMessage(tc.config)
			}
			store := &nodeRESTStorageStub{agent: agent}
			router := gin.New()
			router.POST("/api/v1/session-targets/:target/start", StartSessionHandler(store))
			router.POST("/api/v1/session-instances/:session_id/realtime-offer", SessionRealtimeOfferHandler(store))
			start := httptest.NewRecorder()
			router.ServeHTTP(start, httptest.NewRequest(http.MethodPost, "/api/v1/session-targets/support.voice/start", strings.NewReader(`{}`)))
			require.Equal(t, http.StatusCreated, start.Code, start.Body.String())
			var result struct {
				OfferURL      string          `json:"offer_url"`
				TurnDetection json.RawMessage `json:"turn_detection"`
			}
			require.NoError(t, json.Unmarshal(start.Body.Bytes(), &result))
			require.JSONEq(t, tc.expected, string(result.TurnDetection))
			calls := 0
			http.DefaultClient.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
				calls++
				require.NoError(t, req.ParseMultipartForm(1<<20))
				require.Equal(t, "v=0\r\noffer\r\n", req.FormValue("sdp"))
				var config struct {
					Model string `json:"model"`
					Audio struct {
						Output struct {
							Voice string `json:"voice"`
						} `json:"output"`
						Input struct {
							TurnDetection json.RawMessage `json:"turn_detection"`
						} `json:"input"`
					} `json:"audio"`
				}
				require.NoError(t, json.Unmarshal([]byte(req.FormValue("session")), &config))
				require.JSONEq(t, tc.expected, string(config.Audio.Input.TurnDetection))
				require.Equal(t, "gpt-vad-test", config.Model)
				require.Equal(t, "cedar", config.Audio.Output.Voice)
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("answer"))}, nil
			})
			if tc.name == "semantic" {
				// CLI offers identify the target without repeating model/voice.
				offerURL, err := url.Parse(result.OfferURL)
				require.NoError(t, err)
				query := offerURL.Query()
				query.Del("model")
				query.Del("voice")
				offerURL.RawQuery = query.Encode()
				result.OfferURL = offerURL.String()
			}
			offer := httptest.NewRecorder()
			router.ServeHTTP(offer, httptest.NewRequest(http.MethodPost, result.OfferURL, strings.NewReader("v=0\r\noffer\r\n")))
			require.Equal(t, http.StatusOK, offer.Code, offer.Body.String())
			require.Equal(t, 1, calls)
		})
	}
}

func TestStartSessionRejectsInvalidTurnDetection(t *testing.T) {
	for _, config := range []string{
		`{}`, `{"Type":"server_vad"}`, `{"type":"semantic_vad","eagerness":""}`, `{"type":"client_vad"}`, `{"type":"server_vad","threshold":2}`,
		`{"type":"server_vad","silence_duration_ms":-1}`, `{"type":"server_vad","prefix_padding_ms":1.5}`,
		`{"type":"server_vad","create_response":"false"}`, `{"type":"server_vad","interrupt_response":null}`,
		`{"type":"semantic_vad","threshold":0.5}`, `{"type":"server_vad","eagerness":"low"}`,
		`{"type":"semantic_vad","eagerness":"urgent"}`, `{"type":"server_vad","unknown":true}`, `[]`,
	} {
		t.Run(config, func(t *testing.T) {
			agent := sessionTestAgent()
			agent.Metadata.Custom["sessions"].([]interface{})[0].(map[string]interface{})["turn_detection"] = json.RawMessage(config)
			router := gin.New()
			router.POST("/:target/start", StartSessionHandler(&nodeRESTStorageStub{agent: agent}))
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/support.voice/start", strings.NewReader(`{}`)))
			require.Equal(t, http.StatusBadRequest, rec.Code, rec.Body.String())
			require.Contains(t, rec.Body.String(), "turn_detection")
		})
	}
}

func TestSessionOfferRevalidatesRegisteredTurnDetection(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "") // Invalid config must fail before credentials/upstream.
	agent := sessionTestAgent()
	agent.Metadata.Custom["sessions"].([]interface{})[0].(map[string]interface{})["turn_detection"] = map[string]interface{}{
		"type": "semantic_vad", "silence_duration_ms": 500,
	}
	router := gin.New()
	router.POST("/:target/realtime-offer", SessionRealtimeOfferHandler(&nodeRESTStorageStub{agent: agent}))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost,
		"/sess-1/realtime-offer?provider=openai&transport=webrtc&target=support.voice", strings.NewReader("v=0")))
	require.Equal(t, http.StatusBadRequest, rec.Code)
	require.Contains(t, rec.Body.String(), "turn_detection")
}

func TestStartSessionRejectsTurnDetectionForOpenRouter(t *testing.T) {
	agent := sessionTestAgent()
	raw := agent.Metadata.Custom["sessions"].([]interface{})[0].(map[string]interface{})
	raw["provider"], raw["transport"] = "openrouter", "audio_turns"
	raw["turn_detection"] = map[string]interface{}{"type": "server_vad"}
	router := gin.New()
	router.POST("/:target/start", StartSessionHandler(&nodeRESTStorageStub{agent: agent}))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/support.voice/start", strings.NewReader(`{}`)))
	require.Equal(t, http.StatusBadRequest, rec.Code)
	require.Contains(t, rec.Body.String(), "turn_detection requires")
}
