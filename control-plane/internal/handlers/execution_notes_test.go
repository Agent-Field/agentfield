package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type executionNoteDIDAuthStorage struct {
	*testExecutionStorage
	didDocuments map[string]*types.DIDDocumentRecord
	agentDIDs    []*types.AgentDIDInfo
	didLookupErr error
	listErr      error
}

type getCountingExecutionNoteStorage struct {
	*testExecutionStorage
	getCalls int
}

func (s *getCountingExecutionNoteStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	s.getCalls++
	return s.testExecutionStorage.GetExecutionRecord(ctx, executionID)
}

func (s *executionNoteDIDAuthStorage) GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error) {
	if s.didLookupErr != nil {
		return nil, s.didLookupErr
	}
	if s.didDocuments == nil {
		return nil, nil
	}
	return s.didDocuments[did], nil
}

func (s *executionNoteDIDAuthStorage) ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error) {
	if s.listErr != nil {
		return nil, s.listErr
	}
	return s.agentDIDs, nil
}

func TestAddExecutionNoteHandler_AppendsNoteAndPublishesEvent(t *testing.T) {
	gin.SetMode(gin.TestMode)

	executionID := "exec-1"
	runID := "wf-1" // run_id is the workflow ID equivalent
	agentID := "agent-1"

	storage := newTestExecutionStorage(nil)
	exec := &types.Execution{
		ExecutionID: executionID,
		RunID:       runID,
		AgentNodeID: agentID,
		Notes:       []types.ExecutionNote{},
		UpdatedAt:   time.Now(),
	}
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), exec))

	// Subscribe to event bus to ensure event emitted
	subscriber := storage.GetExecutionEventBus().Subscribe("test-subscriber")
	defer storage.GetExecutionEventBus().Unsubscribe("test-subscriber")

	router := gin.New()
	router.POST("/api/v1/executions/note", func(c *gin.Context) {
		c.Set("execution_id", executionID)
		// Simulate APIKeyAuth having captured the authenticated caller identity.
		c.Set(string(middleware.CallerAgentIDKey), agentID)
		AddExecutionNoteHandler(storage, true)(c)
	})

	reqBody := `{"message":"This is a note","tags":["debug"]}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/note", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload AddNoteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.True(t, payload.Success)
	require.Equal(t, "Note added successfully", payload.Message)
	require.Equal(t, []string{"debug"}, payload.Note.Tags)

	// Verify execution updated
	updated, err := storage.GetExecutionRecord(context.Background(), executionID)
	require.NoError(t, err)
	require.Len(t, updated.Notes, 1)
	require.Equal(t, "This is a note", updated.Notes[0].Message)

	// Ensure event published
	select {
	case evt := <-subscriber:
		require.Equal(t, runID, evt.WorkflowID)
		require.Equal(t, executionID, evt.ExecutionID)
		require.Equal(t, "note_added", evt.Status)
	case <-time.After(time.Second):
		t.Fatal("expected workflow note event")
	}
}

func TestAddExecutionNoteHandler_RejectsNonOwnerAPIKeyCaller(t *testing.T) {
	gin.SetMode(gin.TestMode)

	executionID := "exec-owned-by-b"
	storage := newTestExecutionStorage(nil)
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: executionID,
		RunID:       "run-1",
		AgentNodeID: "agent-b",
		Notes:       []types.ExecutionNote{},
		UpdatedAt:   time.Now(),
	}))

	router := gin.New()
	router.POST("/api/v1/executions/note", func(c *gin.Context) {
		// APIKeyAuth would have validated the key and recorded the caller identity.
		c.Set(string(middleware.CallerAgentIDKey), "agent-a")
		AddExecutionNoteHandler(storage, true)(c)
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/note", strings.NewReader(`{"message":"poisoned note"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Execution-ID", executionID)

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusForbidden, resp.Code)
	require.Contains(t, resp.Body.String(), "this execution does not belong to the requesting agent")

	updated, err := storage.GetExecutionRecord(context.Background(), executionID)
	require.NoError(t, err)
	require.Empty(t, updated.Notes)
}

func TestAddExecutionNoteHandler_RejectsMissingOwnerOrCaller(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		ownerID    string
		callerID   string
		wantStatus int
		wantBody   string
	}{
		{
			name:       "execution owner missing",
			ownerID:    "",
			callerID:   "agent-a",
			wantStatus: http.StatusForbidden,
			wantBody:   "execution owner is required to add notes",
		},
		{
			name:       "caller identity missing",
			ownerID:    "agent-a",
			callerID:   "",
			wantStatus: http.StatusForbidden,
			wantBody:   "caller agent identity is required to add notes to this execution",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executionID := "exec-" + strings.ReplaceAll(tt.name, " ", "-")
			storage := newTestExecutionStorage(nil)
			require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
				ExecutionID: executionID,
				RunID:       "run-1",
				AgentNodeID: tt.ownerID,
				Notes:       []types.ExecutionNote{},
				UpdatedAt:   time.Now(),
			}))

			router := gin.New()
			router.POST("/api/v1/executions/note", func(c *gin.Context) {
				if tt.callerID != "" {
					c.Set(string(middleware.CallerAgentIDKey), tt.callerID)
				}
				AddExecutionNoteHandler(storage, true)(c)
			})

			req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/note", strings.NewReader(`{"message":"should be rejected"}`))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Execution-ID", executionID)

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, tt.wantStatus, resp.Code)
			require.Contains(t, resp.Body.String(), tt.wantBody)

			updated, err := storage.GetExecutionRecord(context.Background(), executionID)
			require.NoError(t, err)
			require.Empty(t, updated.Notes)
		})
	}
}

func TestAddExecutionNoteHandler_DIDCallerOwnership(t *testing.T) {
	gin.SetMode(gin.TestMode)

	const callerDID = "did:web:example.com:agents:agent-a"

	tests := []struct {
		name           string
		executionOwner string
		wantStatus     int
		wantNotes      int
	}{
		{
			name:           "owner write succeeds",
			executionOwner: "agent-a",
			wantStatus:     http.StatusOK,
			wantNotes:      1,
		},
		{
			name:           "non owner write forbidden",
			executionOwner: "agent-b",
			wantStatus:     http.StatusForbidden,
			wantNotes:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executionID := "exec-did-" + strings.ReplaceAll(tt.name, " ", "-")
			storage := &executionNoteDIDAuthStorage{
				testExecutionStorage: newTestExecutionStorage(nil),
				didDocuments: map[string]*types.DIDDocumentRecord{
					callerDID: {
						DID:     callerDID,
						AgentID: "agent-a",
					},
				},
			}
			require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
				ExecutionID: executionID,
				RunID:       "run-did",
				AgentNodeID: tt.executionOwner,
				Notes:       []types.ExecutionNote{},
				UpdatedAt:   time.Now(),
			}))

			router := gin.New()
			router.POST("/api/v1/executions/note", func(c *gin.Context) {
				c.Set(string(middleware.VerifiedCallerDIDKey), callerDID)
				AddExecutionNoteHandler(storage, true)(c)
			})

			req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/note", strings.NewReader(`{"message":"did note"}`))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Execution-ID", executionID)

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, tt.wantStatus, resp.Code)

			updated, err := storage.GetExecutionRecord(context.Background(), executionID)
			require.NoError(t, err)
			require.Len(t, updated.Notes, tt.wantNotes)
			if tt.wantStatus == http.StatusForbidden {
				require.Contains(t, resp.Body.String(), "this execution does not belong to the requesting agent")
			}
		})
	}
}

func TestAddExecutionNoteHandler_DIDResolutionFailure(t *testing.T) {
	gin.SetMode(gin.TestMode)

	const callerDID = "did:web:example.com:agents:agent-a"

	tests := []struct {
		name       string
		storage    *executionNoteDIDAuthStorage
		wantStatus int
		wantBody   string
	}{
		{
			name: "DID resolver error returns server error",
			storage: &executionNoteDIDAuthStorage{
				testExecutionStorage: newTestExecutionStorage(nil),
				listErr:              errors.New("DID registry unavailable"),
			},
			wantStatus: http.StatusInternalServerError,
			wantBody:   "Failed to resolve caller identity",
		},
		{
			name: "unresolved DID fails closed",
			storage: &executionNoteDIDAuthStorage{
				testExecutionStorage: newTestExecutionStorage(nil),
				agentDIDs: []*types.AgentDIDInfo{
					{DID: "did:web:example.com:agents:other", AgentNodeID: "agent-other"},
				},
			},
			wantStatus: http.StatusForbidden,
			wantBody:   "caller agent identity is required to add notes to this execution",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executionID := "exec-" + strings.ReplaceAll(tt.name, " ", "-")
			require.NoError(t, tt.storage.CreateExecutionRecord(context.Background(), &types.Execution{
				ExecutionID: executionID,
				RunID:       "run-did",
				AgentNodeID: "agent-a",
				Notes:       []types.ExecutionNote{},
				UpdatedAt:   time.Now(),
			}))

			router := gin.New()
			router.POST("/api/v1/executions/note", func(c *gin.Context) {
				c.Set(string(middleware.VerifiedCallerDIDKey), callerDID)
				AddExecutionNoteHandler(tt.storage, true)(c)
			})

			req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/note", strings.NewReader(`{"message":"did note"}`))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Execution-ID", executionID)

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, tt.wantStatus, resp.Code)
			require.Contains(t, resp.Body.String(), tt.wantBody)

			updated, err := tt.storage.GetExecutionRecord(context.Background(), executionID)
			require.NoError(t, err)
			require.Empty(t, updated.Notes)
		})
	}
}

func TestAddExecutionNoteHandler_SpoofedHeaderRejectedWhenEnforced(t *testing.T) {
	gin.SetMode(gin.TestMode)

	executionID := "exec-owned-by-victim"
	storage := newTestExecutionStorage(nil)
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: executionID,
		RunID:       "run-1",
		AgentNodeID: "victim-agent",
		Notes:       []types.ExecutionNote{},
		UpdatedAt:   time.Now(),
	}))

	router := gin.New()
	// No verified DID, no authenticated context — only raw headers, as an
	// unauthenticated attacker would send while ownership enforcement is on.
	router.POST("/api/v1/executions/note", AddExecutionNoteHandler(storage, true))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/note", strings.NewReader(`{"message":"spoofed"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Execution-ID", executionID)
	req.Header.Set("X-Agent-Node-ID", "victim-agent")   // spoofed owner id
	req.Header.Set("X-Caller-Agent-ID", "victim-agent") // spoofed owner id

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusForbidden, resp.Code)
	require.Contains(t, resp.Body.String(), "caller agent identity is required to add notes to this execution")

	updated, err := storage.GetExecutionRecord(context.Background(), executionID)
	require.NoError(t, err)
	require.Empty(t, updated.Notes)
}

func TestAddExecutionNoteHandler_NoAuthModeSkipsOwnership(t *testing.T) {
	gin.SetMode(gin.TestMode)

	executionID := "exec-noauth"
	storage := newTestExecutionStorage(nil)
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: executionID,
		RunID:       "run-1",
		AgentNodeID: "agent-owner",
		Notes:       []types.ExecutionNote{},
		UpdatedAt:   time.Now(),
	}))

	router := gin.New()
	// ownershipEnforced=false models a fully unauthenticated deployment: app.note()
	// must keep working even though no trusted caller identity exists.
	router.POST("/api/v1/executions/note", AddExecutionNoteHandler(storage, false))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/note", strings.NewReader(`{"message":"local note"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Execution-ID", executionID)
	// Header identity differs from the owner; with no auth configured the
	// ownership check is intentionally skipped, so the write still succeeds.
	req.Header.Set("X-Agent-Node-ID", "someone-else")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	updated, err := storage.GetExecutionRecord(context.Background(), executionID)
	require.NoError(t, err)
	require.Len(t, updated.Notes, 1)
	require.Equal(t, "local note", updated.Notes[0].Message)
}

func TestResolveExecutionNoteAgentIDByDID_RevokedEntriesFailClosed(t *testing.T) {
	const callerDID = "did:web:example.com:agents:agent-a"
	revokedAt := time.Now()

	t.Run("revoked DID document is not used for resolution", func(t *testing.T) {
		storage := &executionNoteDIDAuthStorage{
			testExecutionStorage: newTestExecutionStorage(nil),
			didDocuments: map[string]*types.DIDDocumentRecord{
				callerDID: {DID: callerDID, AgentID: "agent-a", RevokedAt: &revokedAt},
			},
			// No active agent_dids entry to fall back to.
		}

		got, err := resolveExecutionNoteAgentIDByDID(context.Background(), storage, callerDID)

		require.NoError(t, err)
		require.Empty(t, got)
	})

	t.Run("revoked agent_dids entry is skipped", func(t *testing.T) {
		storage := &executionNoteDIDAuthStorage{
			testExecutionStorage: newTestExecutionStorage(nil),
			agentDIDs: []*types.AgentDIDInfo{
				{DID: callerDID, AgentNodeID: "agent-a", Status: types.AgentDIDStatusRevoked},
			},
		}

		got, err := resolveExecutionNoteAgentIDByDID(context.Background(), storage, callerDID)

		require.NoError(t, err)
		require.Empty(t, got)
	})

	t.Run("active entry still resolves", func(t *testing.T) {
		storage := &executionNoteDIDAuthStorage{
			testExecutionStorage: newTestExecutionStorage(nil),
			agentDIDs: []*types.AgentDIDInfo{
				{DID: callerDID, AgentNodeID: "agent-a", Status: types.AgentDIDStatusActive},
			},
		}

		got, err := resolveExecutionNoteAgentIDByDID(context.Background(), storage, callerDID)

		require.NoError(t, err)
		require.Equal(t, "agent-a", got)
	})
}

func TestExecutionNoteCallerAgentIDResolution(t *testing.T) {
	gin.SetMode(gin.TestMode)

	newContext := func() *gin.Context {
		resp := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(resp)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/executions/note", nil)
		return c
	}

	t.Run("authorization error string", func(t *testing.T) {
		err := &executionNoteAuthorizationError{message: "denied"}
		require.Equal(t, "denied", err.Error())
	})

	t.Run("caller context takes precedence", func(t *testing.T) {
		c := newContext()
		c.Set(string(middleware.CallerAgentIDKey), " agent-from-context ")

		got, err := executionNoteCallerAgentID(context.Background(), c, newTestExecutionStorage(nil))

		require.NoError(t, err)
		require.Equal(t, "agent-from-context", got)
	})

	t.Run("raw caller/node headers are not trusted", func(t *testing.T) {
		// Without a verified DID or an authenticated middleware context value,
		// attacker-controlled X-Caller-Agent-ID / X-Agent-Node-ID headers must NOT
		// resolve to an identity — they would otherwise allow ownership spoofing.
		c := newContext()
		c.Request.Header.Set("X-Caller-Agent-ID", "spoofed-caller")
		c.Request.Header.Set("X-Agent-Node-ID", "spoofed-node")

		got, err := executionNoteCallerAgentID(context.Background(), c, newTestExecutionStorage(nil))

		require.NoError(t, err)
		require.Empty(t, got)
	})

	t.Run("non-string context value falls through to empty, not raw headers", func(t *testing.T) {
		// A non-string value on the caller-id key must not silently revert to
		// reading attacker-controlled headers.
		c := newContext()
		c.Set(string(middleware.CallerAgentIDKey), 42)
		c.Request.Header.Set("X-Agent-Node-ID", "spoofed-node")

		got, err := executionNoteCallerAgentID(context.Background(), c, newTestExecutionStorage(nil))

		require.NoError(t, err)
		require.Empty(t, got)
	})

	t.Run("DID list fallback skips nil entries", func(t *testing.T) {
		const callerDID = "did:web:example.com:agents:agent-a"
		c := newContext()
		c.Set(string(middleware.VerifiedCallerDIDKey), callerDID)
		storage := &executionNoteDIDAuthStorage{
			testExecutionStorage: newTestExecutionStorage(nil),
			agentDIDs: []*types.AgentDIDInfo{
				nil,
				{DID: "did:web:example.com:agents:other", AgentNodeID: "agent-other"},
				{DID: callerDID, AgentNodeID: " agent-a "},
			},
		}

		got, err := executionNoteCallerAgentID(context.Background(), c, storage)

		require.NoError(t, err)
		require.Equal(t, "agent-a", got)
	})

	t.Run("DID lookup error falls back to list", func(t *testing.T) {
		const callerDID = "did:web:example.com:agents:agent-a"
		c := newContext()
		c.Set(string(middleware.VerifiedCallerDIDKey), callerDID)
		storage := &executionNoteDIDAuthStorage{
			testExecutionStorage: newTestExecutionStorage(nil),
			didLookupErr:         errors.New("lookup failed"),
			agentDIDs: []*types.AgentDIDInfo{
				{DID: callerDID, AgentNodeID: "agent-a"},
			},
		}

		got, err := executionNoteCallerAgentID(context.Background(), c, storage)

		require.NoError(t, err)
		require.Equal(t, "agent-a", got)
	})

	t.Run("DID with no resolver returns empty caller", func(t *testing.T) {
		c := newContext()
		c.Set(string(middleware.VerifiedCallerDIDKey), "did:web:example.com:agents:agent-a")

		got, err := executionNoteCallerAgentID(context.Background(), c, newTestExecutionStorage(nil))

		require.NoError(t, err)
		require.Empty(t, got)
	})
}

func TestGetExecutionNotesHandler_ReturnsFilteredNotes(t *testing.T) {
	gin.SetMode(gin.TestMode)

	executionID := "exec-2"
	storage := newTestExecutionStorage(nil)
	exec := &types.Execution{
		ExecutionID: executionID,
		Notes: []types.ExecutionNote{
			{Message: "note-one", Tags: []string{"debug"}},
			{Message: "note-two", Tags: []string{"info"}},
		},
	}
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), exec))

	router := gin.New()
	router.GET("/api/v1/executions/:execution_id/notes", GetExecutionNotesHandler(storage, false))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/exec-2/notes?tags=debug", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload GetNotesResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, executionID, payload.ExecutionID)
	require.Equal(t, 1, payload.Total)
	require.Equal(t, "note-one", payload.Notes[0].Message)
}

func TestGetExecutionNotesHandler_AuthorizesOwnerRead(t *testing.T) {
	gin.SetMode(gin.TestMode)

	executionID := "exec-owned-by-a"
	storage := newTestExecutionStorage(nil)
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: executionID,
		AgentNodeID: "agent-a",
		Notes: []types.ExecutionNote{
			{Message: "owner-visible", Tags: []string{"debug"}},
		},
		UpdatedAt: time.Now(),
	}))

	router := gin.New()
	router.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "secret-key"}))
	router.GET("/api/v1/executions/:execution_id/notes", GetExecutionNotesHandler(storage, true))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/exec-owned-by-a/notes", nil)
	req.Header.Set("X-API-Key", "secret-key")
	req.Header.Set("X-Caller-Agent-ID", "agent-a")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload GetNotesResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, executionID, payload.ExecutionID)
	require.Equal(t, 1, payload.Total)
	require.Equal(t, "owner-visible", payload.Notes[0].Message)
}

func TestGetExecutionNotesHandler_RejectsCrossAgentRead(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := newTestExecutionStorage(nil)
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-owned-by-b",
		AgentNodeID: "agent-b",
		Notes: []types.ExecutionNote{
			{Message: "sensitive reasoning trace", Tags: []string{"private"}},
		},
		UpdatedAt: time.Now(),
	}))

	router := gin.New()
	router.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "secret-key"}))
	router.GET("/api/v1/executions/:execution_id/notes", GetExecutionNotesHandler(storage, true))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/exec-owned-by-b/notes", nil)
	req.Header.Set("X-API-Key", "secret-key")
	req.Header.Set("X-Caller-Agent-ID", "agent-a")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusForbidden, resp.Code)
	require.Contains(t, resp.Body.String(), "this execution does not belong to the requesting agent")
	require.NotContains(t, resp.Body.String(), "sensitive reasoning trace")
}

func TestGetExecutionNotesHandler_DIDCallerOwnership(t *testing.T) {
	gin.SetMode(gin.TestMode)

	const callerDID = "did:web:example.com:agents:agent-a"

	tests := []struct {
		name           string
		executionOwner string
		wantStatus     int
		wantTotal      int
	}{
		{
			name:           "owner read succeeds",
			executionOwner: "agent-a",
			wantStatus:     http.StatusOK,
			wantTotal:      1,
		},
		{
			name:           "non owner read forbidden",
			executionOwner: "agent-b",
			wantStatus:     http.StatusForbidden,
			wantTotal:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executionID := "exec-did-read-" + strings.ReplaceAll(tt.name, " ", "-")
			storage := &executionNoteDIDAuthStorage{
				testExecutionStorage: newTestExecutionStorage(nil),
				didDocuments: map[string]*types.DIDDocumentRecord{
					callerDID: {
						DID:     callerDID,
						AgentID: "agent-a",
					},
				},
			}
			require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
				ExecutionID: executionID,
				AgentNodeID: tt.executionOwner,
				Notes: []types.ExecutionNote{
					{Message: "did-visible-note", Tags: []string{"debug"}},
				},
				UpdatedAt: time.Now(),
			}))

			router := gin.New()
			router.GET("/api/v1/executions/:execution_id/notes", func(c *gin.Context) {
				c.Set(string(middleware.VerifiedCallerDIDKey), callerDID)
				GetExecutionNotesHandler(storage, true)(c)
			})

			req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/"+executionID+"/notes", nil)
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, tt.wantStatus, resp.Code)
			if tt.wantStatus == http.StatusOK {
				var payload GetNotesResponse
				require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
				require.Equal(t, tt.wantTotal, payload.Total)
				require.Equal(t, "did-visible-note", payload.Notes[0].Message)
			} else {
				require.Contains(t, resp.Body.String(), "this execution does not belong to the requesting agent")
				require.NotContains(t, resp.Body.String(), "did-visible-note")
			}
		})
	}
}

func TestGetExecutionNotesHandler_SpoofedHeaderRejectedWhenEnforced(t *testing.T) {
	gin.SetMode(gin.TestMode)

	executionID := "exec-owned-by-victim"
	storage := &getCountingExecutionNoteStorage{testExecutionStorage: newTestExecutionStorage(nil)}
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: executionID,
		AgentNodeID: "victim-agent",
		Notes: []types.ExecutionNote{
			{Message: "must not leak", Tags: []string{"private"}},
		},
		UpdatedAt: time.Now(),
	}))

	router := gin.New()
	router.GET("/api/v1/executions/:execution_id/notes", GetExecutionNotesHandler(storage, true))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/exec-owned-by-victim/notes", nil)
	req.Header.Set("X-Agent-Node-ID", "victim-agent")
	req.Header.Set("X-Caller-Agent-ID", "victim-agent")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusForbidden, resp.Code)
	require.Contains(t, resp.Body.String(), "caller agent identity is required to read notes for this execution")
	require.NotContains(t, resp.Body.String(), "must not leak")
	require.Equal(t, 0, storage.getCalls)
}

func TestGetExecutionNotesHandler_UnauthenticatedRequestRejectedBeforeFetch(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &getCountingExecutionNoteStorage{testExecutionStorage: newTestExecutionStorage(nil)}
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-owned-by-a",
		AgentNodeID: "agent-a",
		Notes: []types.ExecutionNote{
			{Message: "must not be fetched", Tags: []string{"private"}},
		},
		UpdatedAt: time.Now(),
	}))

	router := gin.New()
	router.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "secret-key"}))
	router.GET("/api/v1/executions/:execution_id/notes", GetExecutionNotesHandler(storage, true))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/exec-owned-by-a/notes", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusUnauthorized, resp.Code)
	require.Equal(t, 0, storage.getCalls)
}
