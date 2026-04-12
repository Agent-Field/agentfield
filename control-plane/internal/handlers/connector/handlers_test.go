package connector

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Mock storage
// ---------------------------------------------------------------------------

type mockStorage struct {
	mock.Mock
}

// --- Mocked methods used by connector handlers ---

func (m *mockStorage) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentNode), args.Error(1)
}

func (m *mockStorage) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.AgentNode), args.Error(1)
}

func (m *mockStorage) UpdateAgentVersion(ctx context.Context, id string, version string) error {
	args := m.Called(ctx, id, version)
	return args.Error(0)
}

func (m *mockStorage) ListAgentGroups(ctx context.Context, teamID string) ([]types.AgentGroupSummary, error) {
	args := m.Called(ctx, teamID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]types.AgentGroupSummary), args.Error(1)
}

func (m *mockStorage) ListAgentsByGroup(ctx context.Context, groupID string) ([]*types.AgentNode, error) {
	args := m.Called(ctx, groupID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentNode), args.Error(1)
}

func (m *mockStorage) ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentNode), args.Error(1)
}

func (m *mockStorage) GetAgentVersion(ctx context.Context, id string, version string) (*types.AgentNode, error) {
	args := m.Called(ctx, id, version)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.AgentNode), args.Error(1)
}

func (m *mockStorage) UpdateAgentTrafficWeight(ctx context.Context, id string, version string, weight int) error {
	args := m.Called(ctx, id, version, weight)
	return args.Error(0)
}

func (m *mockStorage) UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error {
	args := m.Called(ctx, id, status)
	return args.Error(0)
}

func (m *mockStorage) UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error {
	args := m.Called(ctx, id, status)
	return args.Error(0)
}

func (m *mockStorage) UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error {
	args := m.Called(ctx, id, status, expectedLastHeartbeat)
	return args.Error(0)
}

func (m *mockStorage) UpdateAgentHeartbeat(ctx context.Context, id string, version string, heartbeatTime time.Time) error {
	args := m.Called(ctx, id, version, heartbeatTime)
	return args.Error(0)
}

// --- No-op stubs for the rest of StorageProvider interface ---

func (m *mockStorage) Initialize(ctx context.Context, cfg storage.StorageConfig) error { return nil }
func (m *mockStorage) Close(ctx context.Context) error                                 { return nil }
func (m *mockStorage) HealthCheck(ctx context.Context) error                           { return nil }
func (m *mockStorage) StoreExecution(ctx context.Context, execution *types.AgentExecution) error {
	return nil
}
func (m *mockStorage) GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error) {
	return nil, nil
}
func (m *mockStorage) QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error) {
	return nil, nil
}
func (m *mockStorage) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	return nil
}
func (m *mockStorage) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *mockStorage) QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *mockStorage) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	return nil
}
func (m *mockStorage) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	return nil
}
func (m *mockStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	return nil, nil
}
func (m *mockStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	return nil, nil
}
func (m *mockStorage) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	return nil, nil
}
func (m *mockStorage) QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
	return nil, 0, nil
}
func (m *mockStorage) RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error {
	return nil
}
func (m *mockStorage) GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error) {
	return nil, nil
}
func (m *mockStorage) ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error) {
	return nil, nil
}
func (m *mockStorage) TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error) {
	return false, nil
}
func (m *mockStorage) UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error {
	return nil
}
func (m *mockStorage) HasExecutionWebhook(ctx context.Context, executionID string) (bool, error) {
	return false, nil
}
func (m *mockStorage) ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error) {
	return nil, nil
}
func (m *mockStorage) StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error {
	return nil
}
func (m *mockStorage) ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error) {
	return nil, nil
}
func (m *mockStorage) ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error) {
	return nil, nil
}
func (m *mockStorage) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	return nil
}
func (m *mockStorage) StoreExecutionLogEntry(ctx context.Context, entry *types.ExecutionLogEntry) error {
	return nil
}
func (m *mockStorage) ListExecutionLogEntries(ctx context.Context, executionID string, afterSeq *int64, limit int, levels []string, nodeIDs []string, sources []string, query string) ([]*types.ExecutionLogEntry, error) {
	return nil, nil
}
func (m *mockStorage) PruneExecutionLogEntries(ctx context.Context, executionID string, maxEntries int, olderThan time.Time) error {
	return nil
}
func (m *mockStorage) ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error) {
	return nil, nil
}
func (m *mockStorage) CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error) {
	return 0, nil
}
func (m *mockStorage) MarkStaleExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	return 0, nil
}
func (m *mockStorage) MarkStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	return 0, nil
}
func (m *mockStorage) RetryStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, maxRetries int, limit int) ([]string, error) {
	return nil, nil
}
func (m *mockStorage) CleanupWorkflow(ctx context.Context, workflowID string, dryRun bool) (*types.WorkflowCleanupResult, error) {
	return nil, nil
}
func (m *mockStorage) QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *mockStorage) CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error {
	return nil
}
func (m *mockStorage) GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error) {
	return nil, nil
}
func (m *mockStorage) QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
	return nil, nil
}
func (m *mockStorage) CreateOrUpdateSession(ctx context.Context, session *types.Session) error {
	return nil
}
func (m *mockStorage) GetSession(ctx context.Context, sessionID string) (*types.Session, error) {
	return nil, nil
}
func (m *mockStorage) QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error) {
	return nil, nil
}
func (m *mockStorage) SetMemory(ctx context.Context, memory *types.Memory) error { return nil }
func (m *mockStorage) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	return nil, nil
}
func (m *mockStorage) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	return nil
}
func (m *mockStorage) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	return nil, nil
}
func (m *mockStorage) SetVector(ctx context.Context, record *types.VectorRecord) error { return nil }
func (m *mockStorage) GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	return nil, nil
}
func (m *mockStorage) DeleteVector(ctx context.Context, scope, scopeID, key string) error {
	return nil
}
func (m *mockStorage) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	return 0, nil
}
func (m *mockStorage) SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	return nil, nil
}
func (m *mockStorage) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	return nil
}
func (m *mockStorage) GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	return nil, nil
}
func (m *mockStorage) AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *mockStorage) ReleaseLock(ctx context.Context, lockID string) error { return nil }
func (m *mockStorage) RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *mockStorage) GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *mockStorage) RegisterAgent(ctx context.Context, agent *types.AgentNode) error { return nil }
func (m *mockStorage) DeleteAgentVersion(ctx context.Context, id string, version string) error {
	return nil
}
func (m *mockStorage) GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	return nil, nil
}
func (m *mockStorage) GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
	return nil, nil
}
func (m *mockStorage) StoreAgentConfiguration(ctx context.Context, cfg *types.AgentConfiguration) error {
	return nil
}
func (m *mockStorage) GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
	return nil, nil
}
func (m *mockStorage) QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error) {
	return nil, nil
}
func (m *mockStorage) UpdateAgentConfiguration(ctx context.Context, cfg *types.AgentConfiguration) error {
	return nil
}
func (m *mockStorage) DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error {
	return nil
}
func (m *mockStorage) ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, cfg map[string]interface{}) (*types.ConfigurationValidationResult, error) {
	return nil, nil
}
func (m *mockStorage) StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (m *mockStorage) GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error) {
	return nil, nil
}
func (m *mockStorage) QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
	return nil, nil
}
func (m *mockStorage) UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (m *mockStorage) DeleteAgentPackage(ctx context.Context, packageID string) error { return nil }
func (m *mockStorage) SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error) {
	return nil, nil
}
func (m *mockStorage) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	return nil
}
func (m *mockStorage) GetExecutionEventBus() *events.ExecutionEventBus { return nil }
func (m *mockStorage) GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent] {
	return nil
}
func (m *mockStorage) GetExecutionLogEventBus() *events.EventBus[*types.ExecutionLogEntry] {
	return events.NewEventBus[*types.ExecutionLogEntry]()
}
func (m *mockStorage) StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error {
	return nil
}
func (m *mockStorage) GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error) {
	return nil, nil
}
func (m *mockStorage) ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error) {
	return nil, nil
}
func (m *mockStorage) StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error {
	return nil
}
func (m *mockStorage) GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}
func (m *mockStorage) ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}
func (m *mockStorage) StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error {
	return nil
}
func (m *mockStorage) GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error) {
	return nil, nil
}
func (m *mockStorage) ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error) {
	return nil, nil
}
func (m *mockStorage) StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error {
	return nil
}
func (m *mockStorage) GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error) {
	return nil, nil
}
func (m *mockStorage) ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	return nil, nil
}
func (m *mockStorage) StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []storage.ComponentDIDRequest) error {
	return nil
}
func (m *mockStorage) StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (m *mockStorage) GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (m *mockStorage) ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (m *mockStorage) ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error) {
	return nil, nil
}
func (m *mockStorage) CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error) {
	return 0, nil
}
func (m *mockStorage) StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (m *mockStorage) GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error) {
	return nil, nil
}
func (m *mockStorage) ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error) {
	return nil, nil
}
func (m *mockStorage) GetObservabilityWebhook(ctx context.Context) (*types.ObservabilityWebhookConfig, error) {
	return nil, nil
}
func (m *mockStorage) SetObservabilityWebhook(ctx context.Context, cfg *types.ObservabilityWebhookConfig) error {
	return nil
}
func (m *mockStorage) DeleteObservabilityWebhook(ctx context.Context) error { return nil }
func (m *mockStorage) AddToDeadLetterQueue(ctx context.Context, event *types.ObservabilityEvent, errorMessage string, retryCount int) error {
	return nil
}
func (m *mockStorage) GetDeadLetterQueueCount(ctx context.Context) (int64, error) { return 0, nil }
func (m *mockStorage) GetDeadLetterQueue(ctx context.Context, limit, offset int) ([]types.ObservabilityDeadLetterEntry, error) {
	return nil, nil
}
func (m *mockStorage) DeleteFromDeadLetterQueue(ctx context.Context, ids []int64) error { return nil }
func (m *mockStorage) ClearDeadLetterQueue(ctx context.Context) error                   { return nil }
func (m *mockStorage) GetAccessPolicies(ctx context.Context) ([]*types.AccessPolicy, error) {
	return nil, nil
}
func (m *mockStorage) GetAccessPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error) {
	return nil, nil
}
func (m *mockStorage) CreateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	return nil
}
func (m *mockStorage) UpdateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	return nil
}
func (m *mockStorage) DeleteAccessPolicy(ctx context.Context, id int64) error { return nil }
func (m *mockStorage) StoreAgentTagVC(ctx context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error {
	return nil
}
func (m *mockStorage) GetAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCRecord, error) {
	return nil, nil
}
func (m *mockStorage) ListAgentTagVCs(ctx context.Context) ([]*types.AgentTagVCRecord, error) {
	return nil, nil
}
func (m *mockStorage) RevokeAgentTagVC(ctx context.Context, agentID string) error { return nil }
func (m *mockStorage) StoreDIDDocument(ctx context.Context, record *types.DIDDocumentRecord) error {
	return nil
}
func (m *mockStorage) GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error) {
	return nil, nil
}
func (m *mockStorage) GetDIDDocumentByAgentID(ctx context.Context, agentID string) (*types.DIDDocumentRecord, error) {
	return nil, nil
}
func (m *mockStorage) RevokeDIDDocument(ctx context.Context, did string) error { return nil }
func (m *mockStorage) ListDIDDocuments(ctx context.Context) ([]*types.DIDDocumentRecord, error) {
	return nil, nil
}
func (m *mockStorage) ListAgentsByLifecycleStatus(ctx context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error) {
	return nil, nil
}
func (m *mockStorage) QueryWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) ([]*types.WorkflowRun, error) {
	return nil, nil
}
func (m *mockStorage) CountWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) (int, error) {
	return 0, nil
}
func (m *mockStorage) StoreWorkflowRunEvent(ctx context.Context, event *types.WorkflowRunEvent) error {
	return nil
}
func (m *mockStorage) ListWorkflowRunEvents(ctx context.Context, runID string, afterSeq *int64, limit int) ([]*types.WorkflowRunEvent, error) {
	return nil, nil
}
func (m *mockStorage) ListConfigs(ctx context.Context) ([]*storage.ConfigEntry, error) {
	return nil, nil
}
func (m *mockStorage) GetConfig(ctx context.Context, key string) (*storage.ConfigEntry, error) {
	return nil, nil
}
func (m *mockStorage) SetConfig(ctx context.Context, key string, value string, updatedBy string) error {
	return nil
}
func (m *mockStorage) DeleteConfig(ctx context.Context, key string) error { return nil }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func defaultConnectorConfig() config.ConnectorConfig {
	return config.ConnectorConfig{
		Enabled: true,
		Token:   "test-token",
		Capabilities: map[string]config.ConnectorCapability{
			"reasoner_management": {Enabled: true, ReadOnly: false},
			"status_read":        {Enabled: true, ReadOnly: true},
			"policy_management":  {Enabled: true, ReadOnly: false},
			"tag_management":     {Enabled: true, ReadOnly: false},
		},
	}
}

func sampleAgent() *types.AgentNode {
	return &types.AgentNode{
		ID:              "agent-1",
		GroupID:         "group-1",
		TeamID:          "team-1",
		Version:         "v1.0.0",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		BaseURL:         "http://localhost:9000",
		TrafficWeight:   100,
		LastHeartbeat:   time.Now().UTC(),
		Reasoners: []types.ReasonerDefinition{
			{ID: "reason-1"},
		},
		Skills: []types.SkillDefinition{
			{ID: "skill-1"},
		},
	}
}

func setupRouter(store *mockStorage, statusMgr *services.StatusManager) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	h := NewHandlers(defaultConnectorConfig(), store, statusMgr, nil, nil, nil)
	group := router.Group("/connector")
	h.RegisterRoutes(group)
	return router
}

func parseBody(t *testing.T, rec *httptest.ResponseRecorder) map[string]interface{} {
	t.Helper()
	var body map[string]interface{}
	err := json.Unmarshal(rec.Body.Bytes(), &body)
	require.NoError(t, err)
	return body
}

// ---------------------------------------------------------------------------
// NewHandlers
// ---------------------------------------------------------------------------

func TestNewHandlers(t *testing.T) {
	store := &mockStorage{}
	cfg := defaultConnectorConfig()
	h := NewHandlers(cfg, store, nil, nil, nil, nil)

	require.NotNil(t, h)
	assert.Equal(t, cfg, h.connectorConfig)
	assert.Equal(t, store, h.storage)
	assert.Nil(t, h.statusManager)
	assert.Nil(t, h.accessPolicyService)
	assert.Nil(t, h.tagApprovalService)
	assert.Nil(t, h.didService)
}

// ---------------------------------------------------------------------------
// RegisterRoutes
// ---------------------------------------------------------------------------

func TestRegisterRoutes_ManifestAlwaysAccessible(t *testing.T) {
	store := &mockStorage{}
	router := setupRouter(store, nil)

	req := httptest.NewRequest("GET", "/connector/manifest", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.True(t, body["connector_enabled"].(bool))
}

// ---------------------------------------------------------------------------
// GetManifest
// ---------------------------------------------------------------------------

func TestGetManifest_ReturnsCapabilities(t *testing.T) {
	store := &mockStorage{}
	router := setupRouter(store, nil)

	req := httptest.NewRequest("GET", "/connector/manifest", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)

	caps := body["capabilities"].(map[string]interface{})
	assert.Contains(t, caps, "reasoner_management")
	assert.Contains(t, caps, "status_read")

	features := body["features"].(map[string]interface{})
	assert.False(t, features["did_enabled"].(bool))
	assert.False(t, features["authorization_enabled"].(bool))
}

func TestGetManifest_WithDIDService(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	store := &mockStorage{}
	// Pass a non-nil didService (we can use an empty struct pointer through the services package)
	h := NewHandlers(defaultConnectorConfig(), store, nil, nil, nil, &services.DIDService{})
	group := router.Group("/connector")
	h.RegisterRoutes(group)

	req := httptest.NewRequest("GET", "/connector/manifest", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	features := body["features"].(map[string]interface{})
	assert.True(t, features["did_enabled"].(bool))
}

// ---------------------------------------------------------------------------
// ListNodes
// ---------------------------------------------------------------------------

func TestListNodes_Success(t *testing.T) {
	store := &mockStorage{}
	agents := []*types.AgentNode{sampleAgent()}
	store.On("ListAgents", mock.Anything, types.AgentFilters{}).Return(agents, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/nodes", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, float64(1), body["total"])
	store.AssertExpectations(t)
}

func TestListNodes_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("ListAgents", mock.Anything, types.AgentFilters{}).Return(nil, errors.New("db down"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/nodes", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

func TestListNodes_Empty(t *testing.T) {
	store := &mockStorage{}
	store.On("ListAgents", mock.Anything, types.AgentFilters{}).Return([]*types.AgentNode{}, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/nodes", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, float64(0), body["total"])
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// GetNodeStatus
// ---------------------------------------------------------------------------

func TestGetNodeStatus_Success(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	store.On("GetAgent", mock.Anything, "agent-1").Return(agent, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/nodes/agent-1/status", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, "agent-1", body["id"])
	assert.Equal(t, "group-1", body["group_id"])
	store.AssertExpectations(t)
}

func TestGetNodeStatus_NotFound(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgent", mock.Anything, "missing").Return((*types.AgentNode)(nil), nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/nodes/missing/status", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestGetNodeStatus_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgent", mock.Anything, "agent-1").Return((*types.AgentNode)(nil), errors.New("timeout"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/nodes/agent-1/status", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// ListReasoners
// ---------------------------------------------------------------------------

func TestListReasoners_Success(t *testing.T) {
	store := &mockStorage{}
	agents := []*types.AgentNode{sampleAgent()}
	store.On("ListAgents", mock.Anything, types.AgentFilters{}).Return(agents, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, float64(1), body["total"])
	store.AssertExpectations(t)
}

func TestListReasoners_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("ListAgents", mock.Anything, types.AgentFilters{}).Return(nil, errors.New("fail"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// GetReasoner
// ---------------------------------------------------------------------------

func TestGetReasoner_Success(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	store.On("GetAgent", mock.Anything, "agent-1").Return(agent, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/agent-1", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, "agent-1", body["id"])
	assert.Equal(t, "http://localhost:9000", body["base_url"])
	store.AssertExpectations(t)
}

func TestGetReasoner_NotFound(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgent", mock.Anything, "missing").Return((*types.AgentNode)(nil), nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/missing", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestGetReasoner_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgent", mock.Anything, "agent-1").Return((*types.AgentNode)(nil), errors.New("broken"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/agent-1", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// SetReasonerVersion
// ---------------------------------------------------------------------------

func TestSetReasonerVersion_Success(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	store.On("GetAgent", mock.Anything, "agent-1").Return(agent, nil)
	store.On("UpdateAgentVersion", mock.Anything, "agent-1", "v2.0.0").Return(nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("PUT", "/connector/reasoners/agent-1/version",
		strings.NewReader(`{"version":"v2.0.0"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.True(t, body["success"].(bool))
	assert.Equal(t, "v1.0.0", body["previous_version"])
	store.AssertExpectations(t)
}

func TestSetReasonerVersion_MissingBody(t *testing.T) {
	store := &mockStorage{}
	router := setupRouter(store, nil)

	req := httptest.NewRequest("PUT", "/connector/reasoners/agent-1/version",
		strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestSetReasonerVersion_NotFound(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgent", mock.Anything, "missing").Return((*types.AgentNode)(nil), nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("PUT", "/connector/reasoners/missing/version",
		strings.NewReader(`{"version":"v2.0.0"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestSetReasonerVersion_UpdateError(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	store.On("GetAgent", mock.Anything, "agent-1").Return(agent, nil)
	store.On("UpdateAgentVersion", mock.Anything, "agent-1", "v2.0.0").Return(errors.New("write failed"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("PUT", "/connector/reasoners/agent-1/version",
		strings.NewReader(`{"version":"v2.0.0"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// RestartReasoner
// ---------------------------------------------------------------------------

func TestRestartReasoner_NilStatusManager(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	store.On("GetAgent", mock.Anything, "agent-1").Return(agent, nil)

	router := setupRouter(store, nil) // nil statusManager
	req := httptest.NewRequest("POST", "/connector/reasoners/agent-1/restart", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
	body := parseBody(t, rec)
	assert.Contains(t, body["error"], "status manager not available")
	store.AssertExpectations(t)
}

func TestRestartReasoner_NotFound(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgent", mock.Anything, "missing").Return((*types.AgentNode)(nil), nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("POST", "/connector/reasoners/missing/restart", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestRestartReasoner_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgent", mock.Anything, "agent-1").Return((*types.AgentNode)(nil), errors.New("db err"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("POST", "/connector/reasoners/agent-1/restart", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// ListAgentGroups
// ---------------------------------------------------------------------------

func TestListAgentGroups_Success(t *testing.T) {
	store := &mockStorage{}
	groups := []types.AgentGroupSummary{
		{GroupID: "group-1", NodeCount: 2},
	}
	store.On("ListAgentGroups", mock.Anything, "default").Return(groups, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/groups", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, float64(1), body["total"])
	store.AssertExpectations(t)
}

func TestListAgentGroups_WithTeamID(t *testing.T) {
	store := &mockStorage{}
	store.On("ListAgentGroups", mock.Anything, "team-x").Return([]types.AgentGroupSummary{}, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/groups?team_id=team-x", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	store.AssertExpectations(t)
}

func TestListAgentGroups_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("ListAgentGroups", mock.Anything, "default").Return(nil, errors.New("fail"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/groups", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// ListGroupNodes
// ---------------------------------------------------------------------------

func TestListGroupNodes_Success(t *testing.T) {
	store := &mockStorage{}
	agents := []*types.AgentNode{sampleAgent()}
	store.On("ListAgentsByGroup", mock.Anything, "group-1").Return(agents, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/groups/group-1/nodes", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, float64(1), body["total"])
	store.AssertExpectations(t)
}

func TestListGroupNodes_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("ListAgentsByGroup", mock.Anything, "group-1").Return(nil, errors.New("fail"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/groups/group-1/nodes", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// ListReasonerVersions
// ---------------------------------------------------------------------------

func TestListReasonerVersions_WithDefault(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	store.On("ListAgentVersions", mock.Anything, "agent-1").Return([]*types.AgentNode{}, nil)
	store.On("GetAgent", mock.Anything, "agent-1").Return(agent, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/agent-1/versions", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, float64(1), body["total"])
	assert.Equal(t, "agent-1", body["id"])
	store.AssertExpectations(t)
}

func TestListReasonerVersions_MultipleVersions(t *testing.T) {
	store := &mockStorage{}
	defaultAgent := sampleAgent()
	v2 := sampleAgent()
	v2.Version = "v2.0.0"
	v2.TrafficWeight = 50

	store.On("ListAgentVersions", mock.Anything, "agent-1").Return([]*types.AgentNode{v2}, nil)
	store.On("GetAgent", mock.Anything, "agent-1").Return(defaultAgent, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/agent-1/versions", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, float64(2), body["total"])
	store.AssertExpectations(t)
}

func TestListReasonerVersions_NotFound(t *testing.T) {
	store := &mockStorage{}
	store.On("ListAgentVersions", mock.Anything, "missing").Return([]*types.AgentNode{}, nil)
	store.On("GetAgent", mock.Anything, "missing").Return((*types.AgentNode)(nil), nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/missing/versions", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestListReasonerVersions_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("ListAgentVersions", mock.Anything, "agent-1").Return(nil, errors.New("err"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/agent-1/versions", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// GetReasonerVersion
// ---------------------------------------------------------------------------

func TestGetReasonerVersion_Success(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	store.On("GetAgentVersion", mock.Anything, "agent-1", "v1.0.0").Return(agent, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/agent-1/versions/v1.0.0", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.Equal(t, "agent-1", body["id"])
	assert.Equal(t, "v1.0.0", body["version"])
	store.AssertExpectations(t)
}

func TestGetReasonerVersion_NotFound(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgentVersion", mock.Anything, "agent-1", "v99").Return((*types.AgentNode)(nil), nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/agent-1/versions/v99", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestGetReasonerVersion_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgentVersion", mock.Anything, "agent-1", "v1.0.0").Return((*types.AgentNode)(nil), errors.New("err"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("GET", "/connector/reasoners/agent-1/versions/v1.0.0", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// SetReasonerTrafficWeight
// ---------------------------------------------------------------------------

func TestSetReasonerTrafficWeight_Success(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	agent.TrafficWeight = 100
	store.On("GetAgentVersion", mock.Anything, "agent-1", "v1.0.0").Return(agent, nil)
	store.On("UpdateAgentTrafficWeight", mock.Anything, "agent-1", "v1.0.0", 50).Return(nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("PUT", "/connector/reasoners/agent-1/versions/v1.0.0/weight",
		strings.NewReader(`{"weight":50}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	body := parseBody(t, rec)
	assert.True(t, body["success"].(bool))
	assert.Equal(t, float64(100), body["previous_weight"])
	assert.Equal(t, float64(50), body["new_weight"])
	store.AssertExpectations(t)
}

func TestSetReasonerTrafficWeight_InvalidJSON(t *testing.T) {
	store := &mockStorage{}
	router := setupRouter(store, nil)

	req := httptest.NewRequest("PUT", "/connector/reasoners/agent-1/versions/v1.0.0/weight",
		strings.NewReader(`not-json`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestSetReasonerTrafficWeight_OutOfRange(t *testing.T) {
	tests := []struct {
		name   string
		weight int
	}{
		{"negative", -1},
		{"too_large", 10001},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := &mockStorage{}
			router := setupRouter(store, nil)

			payload := strings.NewReader(`{"weight":` + strings.TrimSpace(toJSON(t, tt.weight)) + `}`)
			req := httptest.NewRequest("PUT", "/connector/reasoners/agent-1/versions/v1.0.0/weight", payload)
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			assert.Equal(t, http.StatusBadRequest, rec.Code)
			body := parseBody(t, rec)
			assert.Contains(t, body["error"], "weight must be between")
		})
	}
}

func TestSetReasonerTrafficWeight_NotFound(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgentVersion", mock.Anything, "agent-1", "v1.0.0").Return((*types.AgentNode)(nil), nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("PUT", "/connector/reasoners/agent-1/versions/v1.0.0/weight",
		strings.NewReader(`{"weight":50}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestSetReasonerTrafficWeight_UpdateError(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	store.On("GetAgentVersion", mock.Anything, "agent-1", "v1.0.0").Return(agent, nil)
	store.On("UpdateAgentTrafficWeight", mock.Anything, "agent-1", "v1.0.0", 50).Return(errors.New("write fail"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("PUT", "/connector/reasoners/agent-1/versions/v1.0.0/weight",
		strings.NewReader(`{"weight":50}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// RestartReasonerVersion
// ---------------------------------------------------------------------------

func TestRestartReasonerVersion_NilStatusManager(t *testing.T) {
	store := &mockStorage{}
	agent := sampleAgent()
	store.On("GetAgentVersion", mock.Anything, "agent-1", "v1.0.0").Return(agent, nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("POST", "/connector/reasoners/agent-1/versions/v1.0.0/restart", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
	body := parseBody(t, rec)
	assert.Contains(t, body["error"], "status manager not available")
	store.AssertExpectations(t)
}

func TestRestartReasonerVersion_NotFound(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgentVersion", mock.Anything, "agent-1", "v99").Return((*types.AgentNode)(nil), nil)

	router := setupRouter(store, nil)
	req := httptest.NewRequest("POST", "/connector/reasoners/agent-1/versions/v99/restart", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestRestartReasonerVersion_StorageError(t *testing.T) {
	store := &mockStorage{}
	store.On("GetAgentVersion", mock.Anything, "agent-1", "v1.0.0").Return((*types.AgentNode)(nil), errors.New("err"))

	router := setupRouter(store, nil)
	req := httptest.NewRequest("POST", "/connector/reasoners/agent-1/versions/v1.0.0/restart", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	store.AssertExpectations(t)
}

// ---------------------------------------------------------------------------
// Capability gating (routes blocked when capability is disabled)
// ---------------------------------------------------------------------------

func TestCapabilityGating_BlockedWhenDisabled(t *testing.T) {
	tests := []struct {
		name   string
		method string
		path   string
		cap    string
	}{
		{"reasoner_list", "GET", "/connector/reasoners", "reasoner_management"},
		{"status_nodes", "GET", "/connector/nodes", "status_read"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := &mockStorage{}
			gin.SetMode(gin.TestMode)
			router := gin.New()

			// All capabilities disabled
			cfg := config.ConnectorConfig{
				Enabled:      true,
				Capabilities: map[string]config.ConnectorCapability{},
			}
			h := NewHandlers(cfg, store, nil, nil, nil, nil)
			group := router.Group("/connector")
			h.RegisterRoutes(group)

			req := httptest.NewRequest(tt.method, tt.path, nil)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			assert.Equal(t, http.StatusForbidden, rec.Code)
		})
	}
}

func TestCapabilityGating_ManifestAlwaysAllowed(t *testing.T) {
	store := &mockStorage{}
	gin.SetMode(gin.TestMode)
	router := gin.New()

	cfg := config.ConnectorConfig{
		Enabled:      true,
		Capabilities: map[string]config.ConnectorCapability{},
	}
	h := NewHandlers(cfg, store, nil, nil, nil, nil)
	group := router.Group("/connector")
	h.RegisterRoutes(group)

	req := httptest.NewRequest("GET", "/connector/manifest", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func toJSON(t *testing.T, v interface{}) string {
	t.Helper()
	b, err := json.Marshal(v)
	require.NoError(t, err)
	return string(b)
}
