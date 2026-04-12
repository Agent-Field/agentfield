package agentic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// mockStatusStorage is a minimal mock for storage.StorageProvider for status tests.
type mockStatusStorage struct {
	mock.Mock
}

func (m *mockStatusStorage) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentNode), args.Error(1)
}

func (m *mockStatusStorage) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	args := m.Called(ctx, filter)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.Execution), args.Error(1)
}

func (m *mockStatusStorage) HealthCheck(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

// Implement storage.StorageProvider by embedding a no-op base that panics on
// unexpected calls. We only mock the methods StatusHandler actually calls.
func (m *mockStatusStorage) Initialize(ctx context.Context, config storage.StorageConfig) error {
	return nil
}
func (m *mockStatusStorage) Close(ctx context.Context) error { return nil }
func (m *mockStatusStorage) StoreExecution(ctx context.Context, execution *types.AgentExecution) error {
	return nil
}
func (m *mockStatusStorage) GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error) {
	return nil, nil
}
func (m *mockStatusStorage) QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	return nil
}
func (m *mockStatusStorage) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *mockStatusStorage) QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *mockStatusStorage) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	return nil
}
func (m *mockStatusStorage) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	return nil
}
func (m *mockStatusStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	return nil, nil
}
func (m *mockStatusStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	return nil, nil
}
func (m *mockStatusStorage) QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
	return nil, 0, nil
}
func (m *mockStatusStorage) RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error {
	return nil
}
func (m *mockStatusStorage) GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error) {
	return nil, nil
}
func (m *mockStatusStorage) TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error) {
	return false, nil
}
func (m *mockStatusStorage) UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error {
	return nil
}
func (m *mockStatusStorage) HasExecutionWebhook(ctx context.Context, executionID string) (bool, error) {
	return false, nil
}
func (m *mockStatusStorage) ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error {
	return nil
}
func (m *mockStatusStorage) ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	return nil
}
func (m *mockStatusStorage) StoreExecutionLogEntry(ctx context.Context, entry *types.ExecutionLogEntry) error {
	return nil
}
func (m *mockStatusStorage) ListExecutionLogEntries(ctx context.Context, executionID string, afterSeq *int64, limit int, levels []string, nodeIDs []string, sources []string, query string) ([]*types.ExecutionLogEntry, error) {
	return nil, nil
}
func (m *mockStatusStorage) PruneExecutionLogEntries(ctx context.Context, executionID string, maxEntries int, olderThan time.Time) error {
	return nil
}
func (m *mockStatusStorage) ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error) {
	return nil, nil
}
func (m *mockStatusStorage) CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error) {
	return 0, nil
}
func (m *mockStatusStorage) MarkStaleExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	return 0, nil
}
func (m *mockStatusStorage) MarkStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	return 0, nil
}
func (m *mockStatusStorage) RegisterAgent(ctx context.Context, agent *types.AgentNode) error {
	return nil
}
func (m *mockStatusStorage) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	return nil, nil
}
func (m *mockStatusStorage) GetAgentVersion(ctx context.Context, id string, version string) (*types.AgentNode, error) {
	return nil, nil
}
func (m *mockStatusStorage) DeleteAgentVersion(ctx context.Context, id string, version string) error {
	return nil
}
func (m *mockStatusStorage) ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListAgentsByGroup(ctx context.Context, groupID string) ([]*types.AgentNode, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListAgentGroups(ctx context.Context, teamID string) ([]types.AgentGroupSummary, error) {
	return nil, nil
}
func (m *mockStatusStorage) UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error {
	return nil
}
func (m *mockStatusStorage) UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error {
	return nil
}
func (m *mockStatusStorage) UpdateAgentHeartbeat(ctx context.Context, id string, version string, heartbeatTime time.Time) error {
	return nil
}
func (m *mockStatusStorage) UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error {
	return nil
}
func (m *mockStatusStorage) UpdateAgentVersion(ctx context.Context, id string, version string) error {
	return nil
}
func (m *mockStatusStorage) UpdateAgentTrafficWeight(ctx context.Context, id string, version string, weight int) error {
	return nil
}
func (m *mockStatusStorage) SetConfig(ctx context.Context, key string, value string, updatedBy string) error {
	return nil
}
func (m *mockStatusStorage) GetConfig(ctx context.Context, key string) (*storage.ConfigEntry, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListConfigs(ctx context.Context) ([]*storage.ConfigEntry, error) {
	return nil, nil
}
func (m *mockStatusStorage) DeleteConfig(ctx context.Context, key string) error { return nil }
func (m *mockStatusStorage) GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	return nil, nil
}
func (m *mockStatusStorage) GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	return nil
}
func (m *mockStatusStorage) GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
	return nil, nil
}
func (m *mockStatusStorage) QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error) {
	return nil, nil
}
func (m *mockStatusStorage) UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	return nil
}
func (m *mockStatusStorage) DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error {
	return nil
}
func (m *mockStatusStorage) ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (m *mockStatusStorage) GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error) {
	return nil, nil
}
func (m *mockStatusStorage) QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
	return nil, nil
}
func (m *mockStatusStorage) UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (m *mockStatusStorage) DeleteAgentPackage(ctx context.Context, packageID string) error {
	return nil
}
func (m *mockStatusStorage) SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error) {
	return nil, nil
}
func (m *mockStatusStorage) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	return nil
}
func (m *mockStatusStorage) SetMemory(ctx context.Context, memory *types.Memory) error { return nil }
func (m *mockStatusStorage) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	return nil, nil
}
func (m *mockStatusStorage) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	return nil
}
func (m *mockStatusStorage) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	return nil
}
func (m *mockStatusStorage) GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	return nil, nil
}
func (m *mockStatusStorage) AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *mockStatusStorage) ReleaseLock(ctx context.Context, lockID string) error { return nil }
func (m *mockStatusStorage) RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *mockStatusStorage) GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error {
	return nil
}
func (m *mockStatusStorage) GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error {
	return nil
}
func (m *mockStatusStorage) GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error {
	return nil
}
func (m *mockStatusStorage) GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error {
	return nil
}
func (m *mockStatusStorage) GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	return nil, nil
}
// StoreAgentDIDWithComponents is superseded by the typed version below.
func (m *mockStatusStorage) StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (m *mockStatusStorage) GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (m *mockStatusStorage) CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error) {
	return 0, nil
}
func (m *mockStatusStorage) StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (m *mockStatusStorage) GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error) {
	return nil, nil
}
func (m *mockStatusStorage) QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *mockStatusStorage) QueryWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) ([]*types.WorkflowRun, error) {
	return nil, nil
}
func (m *mockStatusStorage) CountWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) (int, error) {
	return 0, nil
}
func (m *mockStatusStorage) CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error {
	return nil
}
func (m *mockStatusStorage) GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error) {
	return nil, nil
}
func (m *mockStatusStorage) QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
	return nil, nil
}
func (m *mockStatusStorage) CreateOrUpdateSession(ctx context.Context, session *types.Session) error {
	return nil
}
func (m *mockStatusStorage) GetSession(ctx context.Context, sessionID string) (*types.Session, error) {
	return nil, nil
}
func (m *mockStatusStorage) QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreWorkflowRunEvent(ctx context.Context, event *types.WorkflowRunEvent) error {
	return nil
}
func (m *mockStatusStorage) ListWorkflowRunEvents(ctx context.Context, runID string, afterSeq *int64, limit int) ([]*types.WorkflowRunEvent, error) {
	return nil, nil
}
func (m *mockStatusStorage) RetryStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, maxRetries int, limit int) ([]string, error) {
	return nil, nil
}
func (m *mockStatusStorage) CleanupWorkflow(ctx context.Context, workflowID string, dryRun bool) (*types.WorkflowCleanupResult, error) {
	return nil, nil
}
func (m *mockStatusStorage) SetVector(ctx context.Context, record *types.VectorRecord) error {
	return nil
}
func (m *mockStatusStorage) GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	return nil, nil
}
func (m *mockStatusStorage) DeleteVector(ctx context.Context, scope, scopeID, key string) error {
	return nil
}
func (m *mockStatusStorage) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	return 0, nil
}
func (m *mockStatusStorage) SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	return nil, nil
}
func (m *mockStatusStorage) GetObservabilityWebhook(ctx context.Context) (*types.ObservabilityWebhookConfig, error) {
	return nil, nil
}
func (m *mockStatusStorage) SetObservabilityWebhook(ctx context.Context, config *types.ObservabilityWebhookConfig) error {
	return nil
}
func (m *mockStatusStorage) DeleteObservabilityWebhook(ctx context.Context) error { return nil }
func (m *mockStatusStorage) AddToDeadLetterQueue(ctx context.Context, event *types.ObservabilityEvent, errorMessage string, retryCount int) error {
	return nil
}
func (m *mockStatusStorage) GetDeadLetterQueueCount(ctx context.Context) (int64, error) {
	return 0, nil
}
func (m *mockStatusStorage) GetDeadLetterQueue(ctx context.Context, limit, offset int) ([]types.ObservabilityDeadLetterEntry, error) {
	return nil, nil
}
func (m *mockStatusStorage) DeleteFromDeadLetterQueue(ctx context.Context, ids []int64) error {
	return nil
}
func (m *mockStatusStorage) ClearDeadLetterQueue(ctx context.Context) error { return nil }
func (m *mockStatusStorage) GetAccessPolicies(ctx context.Context) ([]*types.AccessPolicy, error) {
	return nil, nil
}
func (m *mockStatusStorage) GetAccessPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error) {
	return nil, nil
}
func (m *mockStatusStorage) CreateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	return nil
}
func (m *mockStatusStorage) UpdateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	return nil
}
func (m *mockStatusStorage) DeleteAccessPolicy(ctx context.Context, id int64) error { return nil }
func (m *mockStatusStorage) StoreAgentTagVC(ctx context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error {
	return nil
}
func (m *mockStatusStorage) GetAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCRecord, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListAgentTagVCs(ctx context.Context) ([]*types.AgentTagVCRecord, error) {
	return nil, nil
}
func (m *mockStatusStorage) RevokeAgentTagVC(ctx context.Context, agentID string) error { return nil }
func (m *mockStatusStorage) StoreDIDDocument(ctx context.Context, record *types.DIDDocumentRecord) error {
	return nil
}
func (m *mockStatusStorage) GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error) {
	return nil, nil
}
func (m *mockStatusStorage) GetDIDDocumentByAgentID(ctx context.Context, agentID string) (*types.DIDDocumentRecord, error) {
	return nil, nil
}
func (m *mockStatusStorage) RevokeDIDDocument(ctx context.Context, did string) error { return nil }
func (m *mockStatusStorage) ListDIDDocuments(ctx context.Context) ([]*types.DIDDocumentRecord, error) {
	return nil, nil
}
func (m *mockStatusStorage) ListAgentsByLifecycleStatus(ctx context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error) {
	return nil, nil
}
func (m *mockStatusStorage) GetExecutionEventBus() *events.ExecutionEventBus {
	return nil
}
func (m *mockStatusStorage) GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent] {
	return nil
}
func (m *mockStatusStorage) GetExecutionLogEventBus() *events.EventBus[*types.ExecutionLogEntry] {
	return events.NewEventBus[*types.ExecutionLogEntry]()
}
func (m *mockStatusStorage) ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error) {
	return nil, nil
}
func (m *mockStatusStorage) StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []storage.ComponentDIDRequest) error {
	return nil
}

// --- StatusHandler tests ---

func TestStatusHandler_ReturnsOK(t *testing.T) {
	store := &mockStatusStorage{}

	store.On("ListAgents", mock.Anything, mock.Anything).Return([]*types.AgentNode{
		{ID: "agent-1", LifecycleStatus: types.AgentStatusReady},
		{ID: "agent-2", LifecycleStatus: types.AgentStatusStarting},
		{ID: "agent-3", LifecycleStatus: types.AgentStatusOffline},
	}, nil)
	store.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return([]*types.Execution{
		{Status: "succeeded"},
		{Status: "failed"},
	}, nil)
	store.On("HealthCheck", mock.Anything).Return(nil)

	router := gin.New()
	router.GET("/status", StatusHandler(store))

	req := httptest.NewRequest("GET", "/status", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})

	// health section
	health := data["health"].(map[string]interface{})
	assert.Equal(t, "healthy", health["status"])
	assert.Equal(t, "healthy", health["storage"])

	// agents section: total=3, active=2 (ready+starting)
	agents := data["agents"].(map[string]interface{})
	assert.Equal(t, float64(3), agents["total"])
	assert.Equal(t, float64(2), agents["active"])

	// executions_24h section
	execStats := data["executions_24h"].(map[string]interface{})
	assert.Equal(t, float64(2), execStats["total"])

	// server section exists
	assert.Contains(t, data, "server")

	store.AssertExpectations(t)
}

func TestStatusHandler_EmptyAgents(t *testing.T) {
	store := &mockStatusStorage{}

	store.On("ListAgents", mock.Anything, mock.Anything).Return([]*types.AgentNode{}, nil)
	store.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return([]*types.Execution{}, nil)
	store.On("HealthCheck", mock.Anything).Return(nil)

	router := gin.New()
	router.GET("/status", StatusHandler(store))

	req := httptest.NewRequest("GET", "/status", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	agents := data["agents"].(map[string]interface{})
	assert.Equal(t, float64(0), agents["total"])
	assert.Equal(t, float64(0), agents["active"])
}

func TestStatusHandler_StorageUnhealthy(t *testing.T) {
	store := &mockStatusStorage{}

	store.On("ListAgents", mock.Anything, mock.Anything).Return([]*types.AgentNode{}, nil)
	store.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return([]*types.Execution{}, nil)
	store.On("HealthCheck", mock.Anything).Return(assert.AnError)

	router := gin.New()
	router.GET("/status", StatusHandler(store))

	req := httptest.NewRequest("GET", "/status", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	health := data["health"].(map[string]interface{})
	assert.Equal(t, "unhealthy", health["status"])
	assert.Equal(t, "unhealthy", health["storage"])
}

func TestBoolToStatus(t *testing.T) {
	assert.Equal(t, "healthy", boolToStatus(true))
	assert.Equal(t, "unhealthy", boolToStatus(false))
}
