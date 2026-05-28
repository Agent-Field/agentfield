package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type executionNoteStorageStub struct {
	record    *types.Execution
	getErr    error
	updateErr error
	eventBus  *events.ExecutionEventBus
}

func (s *executionNoteStorageStub) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	return s.record, s.getErr
}

func (s *executionNoteStorageStub) UpdateExecutionRecord(ctx context.Context, executionID string, updateFunc func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	if s.updateErr != nil {
		return nil, s.updateErr
	}
	return updateFunc(s.record)
}

func (s *executionNoteStorageStub) GetExecutionEventBus() *events.ExecutionEventBus {
	if s.eventBus == nil {
		s.eventBus = events.NewExecutionEventBus()
	}
	return s.eventBus
}

type didVCServiceStub struct {
	fakeVCService
	listAgentTagVCsFn func() ([]*types.AgentTagVCRecord, error)
}

func (s *didVCServiceStub) ListAgentTagVCs() ([]*types.AgentTagVCRecord, error) {
	if s.listAgentTagVCsFn != nil {
		return s.listAgentTagVCsFn()
	}
	return []*types.AgentTagVCRecord{}, nil
}

type persistWorkflowExecutionStore struct {
	storage.StorageProvider
	called bool
	err    error
}

func (s *persistWorkflowExecutionStore) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	s.called = true
	return s.err
}

func TestWorkflowExecutionEventHelpersCoverage(t *testing.T) {
	now := time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)
	parentExecutionID := "parent-exec"
	parentWorkflowID := "parent-run"
	duration := int64(42)
	req := &WorkflowExecutionEventRequest{
		ExecutionID:       "exec-1",
		Type:              "worker",
		Status:            "FAILED",
		ParentExecutionID: &parentExecutionID,
		ParentWorkflowID:  &parentWorkflowID,
		InputData:         map[string]interface{}{"alpha": "beta"},
		Result:            map[string]interface{}{"done": true},
		Error:             "boom",
		DurationMS:        &duration,
	}

	record := buildExecutionRecordFromEvent(req, now)
	require.Equal(t, "exec-1", record.RunID)
	require.Equal(t, "worker", record.AgentNodeID)
	require.Equal(t, "worker", record.NodeID)
	require.Equal(t, "worker", record.ReasonerID)
	require.Equal(t, string(types.ExecutionStatusFailed), record.Status)
	require.Equal(t, &parentExecutionID, record.ParentExecutionID)
	require.NotNil(t, record.InputPayload)
	require.NotNil(t, record.ResultPayload)
	require.Equal(t, &duration, record.DurationMS)
	require.NotNil(t, record.ErrorMessage)
	require.Equal(t, "boom", *record.ErrorMessage)
	require.NotNil(t, record.CompletedAt)

	current := &types.Execution{}
	successReq := &WorkflowExecutionEventRequest{
		RunID:             " run-2 ",
		WorkflowID:        "ignored",
		AgentNodeID:       " node-2 ",
		ReasonerID:        " reasoner-2 ",
		Status:            "succeeded",
		ParentExecutionID: &parentExecutionID,
		InputData:         map[string]interface{}{"hello": "world"},
		Result:            []string{"ok"},
		DurationMS:        &duration,
	}
	applyEventToExecution(current, successReq, now)
	require.Equal(t, "run-2", current.RunID)
	require.Equal(t, "node-2", current.AgentNodeID)
	require.Equal(t, "node-2", current.NodeID)
	require.Equal(t, "reasoner-2", current.ReasonerID)
	require.Equal(t, &parentExecutionID, current.ParentExecutionID)
	require.Equal(t, &duration, current.DurationMS)
	require.NotNil(t, current.CompletedAt)
	require.Nil(t, current.ErrorMessage)
	require.NotZero(t, current.StartedAt)
	require.NotNil(t, current.InputPayload)
	require.NotNil(t, current.ResultPayload)

	current.ErrorMessage = &parentWorkflowID
	applyEventToExecution(current, &WorkflowExecutionEventRequest{Status: "running"}, now.Add(time.Second))
	require.NotNil(t, current.ErrorMessage)
	require.Nil(t, marshalJSON(nil))
	require.Nil(t, marshalJSON(func() {}))
	require.JSONEq(t, `{"v":1}`, string(marshalJSON(map[string]int{"v": 1})))
	require.Equal(t, "value", firstNonEmpty(" ", "\tvalue\t", "fallback"))
	require.Empty(t, firstNonEmpty("", "  "))

	workflowExec := buildWorkflowExecutionFromEvent(req, now)
	require.Equal(t, "exec-1", workflowExec.WorkflowID)
	require.Equal(t, "exec-1", *workflowExec.RunID)
	require.Equal(t, "worker.worker", *workflowExec.WorkflowName)
	require.Equal(t, &parentExecutionID, workflowExec.ParentExecutionID)
	require.Equal(t, &parentWorkflowID, workflowExec.ParentWorkflowID)
	require.NotNil(t, workflowExec.ErrorMessage)
	require.NotNil(t, workflowExec.CompletedAt)
}

func TestWorkflowExecutionEventHandler_ErrorBranches(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("create execution record failure", func(t *testing.T) {
		store := &executionRecordCreateErrorStore{testExecutionStorage: newTestExecutionStorage(nil)}
		router := gin.New()
		router.POST("/events", WorkflowExecutionEventHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/events", strings.NewReader(`{"execution_id":"exec-1","status":"running"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusInternalServerError, rec.Code)
		require.Contains(t, rec.Body.String(), "failed to create execution")
	})

	t.Run("update existing nil execution creates replacement", func(t *testing.T) {
		store := &updateNilExecutionStore{}
		router := gin.New()
		router.POST("/events", WorkflowExecutionEventHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/events", strings.NewReader(`{"execution_id":"exec-2","status":"running","type":"task"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		require.NotNil(t, store.updated)
		require.Equal(t, "exec-2", store.updated.ExecutionID)
		require.Equal(t, "task", store.updated.AgentNodeID)
	})
}

type executionRecordCreateErrorStore struct {
	*testExecutionStorage
}

func (s *executionRecordCreateErrorStore) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	return errors.New("create failed")
}

type updateNilExecutionStore struct {
	*testExecutionStorage
	updated *types.Execution
}

func (s *updateNilExecutionStore) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	return &types.Execution{ExecutionID: executionID}, nil
}

func (s *updateNilExecutionStore) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	updated, err := update(nil)
	if err != nil {
		return nil, err
	}
	s.updated = updated
	return updated, nil
}

func TestExecutionNotesCoverageAdditional(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("add note validation and update failure", func(t *testing.T) {
		router := gin.New()
		stub := &executionNoteStorageStub{record: &types.Execution{ExecutionID: "exec-1"}, updateErr: errors.New("boom")}
		router.POST("/notes", AddExecutionNoteHandler(stub))

		req := httptest.NewRequest(http.MethodPost, "/notes", strings.NewReader(`{"message":"ok"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/notes?execution_id=exec-1", strings.NewReader(`{"message":"   "}`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/notes?execution_id=exec-1", strings.NewReader(`{"message":"saved"}`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("add note initializes nil tags and uses header id", func(t *testing.T) {
		stub := &executionNoteStorageStub{
			record:   &types.Execution{ExecutionID: "exec-2", RunID: "run-2", AgentNodeID: "node-2"},
			eventBus: events.NewExecutionEventBus(),
		}
		router := gin.New()
		router.POST("/notes", AddExecutionNoteHandler(stub))

		req := httptest.NewRequest(http.MethodPost, "/notes", strings.NewReader(`{"message":" kept "}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Execution-ID", "exec-2")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		require.Len(t, stub.record.Notes, 1)
		require.Equal(t, "kept", stub.record.Notes[0].Message)
		require.Empty(t, stub.record.Notes[0].Tags)
	})

	t.Run("get notes errors and empty results", func(t *testing.T) {
		router := gin.New()
		router.GET("/notes/:execution_id", GetExecutionNotesHandler(&executionNoteStorageStub{getErr: errors.New("load failed")}))
		router.GET("/missing", GetExecutionNotesHandler(&executionNoteStorageStub{}))

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/missing", nil)
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodGet, "/notes/exec-1", nil)
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		rec = httptest.NewRecorder()
		router = gin.New()
		router.GET("/notes/:execution_id", GetExecutionNotesHandler(&executionNoteStorageStub{record: nil}))
		req = httptest.NewRequest(http.MethodGet, "/notes/exec-404", nil)
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusNotFound, rec.Code)

		rec = httptest.NewRecorder()
		router = gin.New()
		router.GET("/notes/:execution_id", GetExecutionNotesHandler(&executionNoteStorageStub{record: &types.Execution{ExecutionID: "exec-3"}}))
		req = httptest.NewRequest(http.MethodGet, "/notes/exec-3?tags=debug,%20info%20", nil)
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), `"notes":[]`)
	})

	t.Run("execution id extraction and tag matching", func(t *testing.T) {
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/?execution_id=query-id", nil)
		ctx.Set("execution_id", "context-id")
		ctx.Request.Header.Set("X-Execution-ID", "header-id")
		require.Equal(t, "context-id", getExecutionIDFromContext(ctx))

		ctx = nil
		rec = httptest.NewRecorder()
		ctx, _ = gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/?execution_id=query-id", nil)
		ctx.Request.Header.Set("X-Execution-ID", "header-id")
		require.Equal(t, "header-id", getExecutionIDFromContext(ctx))

		rec = httptest.NewRecorder()
		ctx, _ = gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/?execution_id=query-id", nil)
		require.Equal(t, "query-id", getExecutionIDFromContext(ctx))

		note := types.ExecutionNote{Tags: []string{"Debug", " review "}}
		require.True(t, noteHasTags(note, []string{"debug"}))
		require.True(t, noteHasTags(note, nil))
		require.False(t, noteHasTags(note, []string{"missing"}))
	})
}

func TestDIDHandlersCoverageAdditional(t *testing.T) {
	gin.SetMode(gin.TestMode)

	require.Equal(t, "did:web:localhost%3A8080:agents:test", normalizeDIDWeb("did:web:localhost:8080:agents:test"))
	require.Equal(t, "did:key:z6Mk", normalizeDIDWeb("did:key:z6Mk"))
	require.True(t, isAllDigits("8080"))
	require.False(t, isAllDigits("80a0"))
	require.False(t, isAllDigits(""))

	t.Run("export vcs filters and degraded dependencies", func(t *testing.T) {
		var seenFilters *types.VCFilters
		handler := NewDIDHandlers(&fakeDIDService{
			listFn: func() ([]string, error) { return nil, errors.New("list failed") },
		}, &didVCServiceStub{
			fakeVCService: fakeVCService{
				queryExecsFn: func(filters *types.VCFilters) ([]types.ExecutionVC, error) {
					seenFilters = filters
					return []types.ExecutionVC{{VCID: "vc-1", ExecutionID: "exec-1", WorkflowID: "wf-1", CreatedAt: time.Unix(1, 0).UTC()}}, nil
				},
				listWorkflowVCsFn: func() ([]*types.WorkflowVC, error) {
					return []*types.WorkflowVC{{WorkflowVCID: "wvc-1", WorkflowID: "wf-1"}}, nil
				},
			},
			listAgentTagVCsFn: func() ([]*types.AgentTagVCRecord, error) { return nil, errors.New("tags failed") },
		})

		router := gin.New()
		router.GET("/export", handler.ExportVCs)
		req := httptest.NewRequest(http.MethodGet, "/export?workflow_id=wf-1&session_id=s-1&issuer_did=did:issuer&status=succeeded", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		require.NotNil(t, seenFilters)
		require.Equal(t, "wf-1", *seenFilters.WorkflowID)
		require.Equal(t, "s-1", *seenFilters.SessionID)
		require.Equal(t, "did:issuer", *seenFilters.IssuerDID)
		require.Equal(t, "succeeded", *seenFilters.Status)
		require.Equal(t, 100, seenFilters.Limit)
		require.Contains(t, rec.Body.String(), `"agent_dids":[]`)
		require.Contains(t, rec.Body.String(), `"total_count":2`)
	})

	t.Run("export vcs query failures", func(t *testing.T) {
		router := gin.New()
		handler := NewDIDHandlers(&fakeDIDService{}, &didVCServiceStub{
			fakeVCService: fakeVCService{
				queryExecsFn: func(filters *types.VCFilters) ([]types.ExecutionVC, error) {
					return nil, errors.New("query failed")
				},
			},
		})
		router.GET("/export", handler.ExportVCs)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/export", nil))
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		router = gin.New()
		handler = NewDIDHandlers(&fakeDIDService{}, &didVCServiceStub{
			fakeVCService: fakeVCService{
				queryExecsFn: func(filters *types.VCFilters) ([]types.ExecutionVC, error) { return []types.ExecutionVC{}, nil },
				listWorkflowVCsFn: func() ([]*types.WorkflowVC, error) {
					return nil, errors.New("workflow failed")
				},
			},
		})
		router.GET("/export", handler.ExportVCs)
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/export", nil))
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("get did document did web and validation branches", func(t *testing.T) {
		handler := NewDIDHandlers(&fakeDIDService{
			resolveFn: func(did string) (*types.DIDIdentity, error) {
				return &types.DIDIdentity{DID: did, PublicKeyJWK: `not-json`}, nil
			},
		}, &fakeVCService{})
		handler.SetDIDWebService(&fakeDIDWebService{
			resolveFn: func(ctx context.Context, did string) (*types.DIDResolutionResult, error) {
				switch did {
				case "did:web:localhost%3A8080:agents:test":
					return &types.DIDResolutionResult{
						DIDDocument: &types.DIDWebDocument{ID: did},
					}, nil
				case "did:web:revoked%3A443:agents:test":
					return &types.DIDResolutionResult{
						DIDResolutionMetadata: types.DIDResolutionMetadata{Error: "deactivated"},
					}, nil
				default:
					return nil, errors.New("fallback")
				}
			},
		})

		router := gin.New()
		router.GET("/document/:did", handler.GetDIDDocument)
		router.GET("/missing", handler.GetDIDDocument)

		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/missing", nil))
		require.Equal(t, http.StatusBadRequest, rec.Code)

		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/document/did:web:localhost:8080:agents:test", nil))
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), "did:web:localhost%3A8080:agents:test")

		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/document/did:web:revoked:443:agents:test", nil))
		require.Equal(t, http.StatusGone, rec.Code)

		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/document/did:example:bad-jwk", nil))
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})
}

func TestExecuteReasonerAndWebhookHelpersCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("persist workflow execution tolerates success and error", func(t *testing.T) {
		successStore := &persistWorkflowExecutionStore{}
		persistWorkflowExecution(context.Background(), successStore, &types.WorkflowExecution{ExecutionID: "exec-1"})
		require.True(t, successStore.called)

		errorStore := &persistWorkflowExecutionStore{err: errors.New("store failed")}
		persistWorkflowExecution(context.Background(), errorStore, &types.WorkflowExecution{ExecutionID: "exec-2"})
		require.True(t, errorStore.called)
	})

	t.Run("execute reasoner invalid workflow and missing reasoner", func(t *testing.T) {
		store := newReasonerHandlerStorage(&types.AgentNode{
			ID:              "node-1",
			BaseURL:         "http://agent.invalid",
			Reasoners:       []types.ReasonerDefinition{{ID: "other"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		})
		router := gin.New()
		router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.other", strings.NewReader(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Workflow-ID", strings.Repeat("w", 256))
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/reasoners/node-1.missing", strings.NewReader(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusNotFound, rec.Code)
	})

	t.Run("webhook helper branches", func(t *testing.T) {
		require.Equal(t, "approved", normalizeDecision("approve"))
		require.Equal(t, "approved", normalizeDecision("continue"))
		require.Equal(t, "rejected", normalizeDecision("abort"))
		require.Equal(t, "rejected", normalizeDecision("cancel"))
		require.Equal(t, "request_changes", normalizeDecision("request_changes"))
		require.Equal(t, "custom", normalizeDecision("custom"))
		require.Equal(t, []string{"a", "b", "c"}, splitSignatureParts("a,b,c"))
		require.Equal(t, -1, indexOf("abc", 'z'))
		require.Equal(t, 1, indexOf("abc", 'b'))
		require.Equal(t, "abc", trimSignaturePrefix("sha256=abc"))
		require.Equal(t, "abc", trimSignaturePrefix("abc"))

		callbacks := make(chan []byte, 2)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, err := json.Marshal(map[string]string{"ok": "true"})
			require.NoError(t, err)
			raw, err := io.ReadAll(r.Body)
			require.NoError(t, err)
			callbacks <- raw
			if r.URL.Path == "/fail" {
				w.WriteHeader(http.StatusInternalServerError)
			}
			_, _ = w.Write(body)
		}))
		defer server.Close()

		ctrl := &webhookApprovalController{}
		response := `{"decision":"approved"}`
		ctrl.notifyApprovalCallback(server.URL, "exec-1", "approved", "running", "ok", &response, "req-1")
		ctrl.notifyApprovalCallback(server.URL+"/fail", "exec-2", "rejected", "cancelled", "", nil, "req-2")
		ctrl.notifyApprovalCallback("http://127.0.0.1:1", "exec-3", "approved", "running", "", nil, "req-3")

		first := <-callbacks
		second := <-callbacks
		require.True(t, bytes.Contains(first, []byte(`"approval_request_id":"req-1"`)) || bytes.Contains(second, []byte(`"approval_request_id":"req-1"`)))
	})
}
