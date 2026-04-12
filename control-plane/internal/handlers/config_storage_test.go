package handlers

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
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

// configStorageMock is a testify mock for storage.StorageProvider focused on config methods.
// All methods not relevant to config storage return zero values.
type configStorageMock struct {
	mock.Mock
}

// --- Mocked config methods ---

func (m *configStorageMock) ListConfigs(ctx context.Context) ([]*storage.ConfigEntry, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*storage.ConfigEntry), args.Error(1)
}

func (m *configStorageMock) GetConfig(ctx context.Context, key string) (*storage.ConfigEntry, error) {
	args := m.Called(ctx, key)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*storage.ConfigEntry), args.Error(1)
}

func (m *configStorageMock) SetConfig(ctx context.Context, key string, value string, updatedBy string) error {
	args := m.Called(ctx, key, value, updatedBy)
	return args.Error(0)
}

func (m *configStorageMock) DeleteConfig(ctx context.Context, key string) error {
	args := m.Called(ctx, key)
	return args.Error(0)
}

// --- No-op stubs for the rest of the interface ---

func (m *configStorageMock) Initialize(ctx context.Context, config storage.StorageConfig) error {
	return nil
}
func (m *configStorageMock) Close(ctx context.Context) error                    { return nil }
func (m *configStorageMock) HealthCheck(ctx context.Context) error              { return nil }
func (m *configStorageMock) StoreExecution(ctx context.Context, execution *types.AgentExecution) error {
	return nil
}
func (m *configStorageMock) GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error) {
	return nil, nil
}
func (m *configStorageMock) QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error) {
	return nil, nil
}
func (m *configStorageMock) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	return nil
}
func (m *configStorageMock) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *configStorageMock) QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *configStorageMock) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	return nil
}
func (m *configStorageMock) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	return nil
}
func (m *configStorageMock) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	return nil, nil
}
func (m *configStorageMock) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	return nil, nil
}
func (m *configStorageMock) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	return nil, nil
}
func (m *configStorageMock) QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
	return nil, 0, nil
}
func (m *configStorageMock) RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error {
	return nil
}
func (m *configStorageMock) GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error) {
	return nil, nil
}
func (m *configStorageMock) ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error) {
	return nil, nil
}
func (m *configStorageMock) TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error) {
	return false, nil
}
func (m *configStorageMock) UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error {
	return nil
}
func (m *configStorageMock) HasExecutionWebhook(ctx context.Context, executionID string) (bool, error) {
	return false, nil
}
func (m *configStorageMock) ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error) {
	return nil, nil
}
func (m *configStorageMock) StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error {
	return nil
}
func (m *configStorageMock) ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error) {
	return nil, nil
}
func (m *configStorageMock) ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error) {
	return nil, nil
}
func (m *configStorageMock) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	return nil
}
func (m *configStorageMock) StoreExecutionLogEntry(ctx context.Context, entry *types.ExecutionLogEntry) error {
	return nil
}
func (m *configStorageMock) ListExecutionLogEntries(ctx context.Context, executionID string, afterSeq *int64, limit int, levels []string, nodeIDs []string, sources []string, query string) ([]*types.ExecutionLogEntry, error) {
	return nil, nil
}
func (m *configStorageMock) PruneExecutionLogEntries(ctx context.Context, executionID string, maxEntries int, olderThan time.Time) error {
	return nil
}
func (m *configStorageMock) ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error) {
	return nil, nil
}
func (m *configStorageMock) CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error) {
	return 0, nil
}
func (m *configStorageMock) MarkStaleExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	return 0, nil
}
func (m *configStorageMock) MarkStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	return 0, nil
}
func (m *configStorageMock) RetryStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, maxRetries int, limit int) ([]string, error) {
	return nil, nil
}
func (m *configStorageMock) CleanupWorkflow(ctx context.Context, workflowID string, dryRun bool) (*types.WorkflowCleanupResult, error) {
	return nil, nil
}
func (m *configStorageMock) QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *configStorageMock) CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error {
	return nil
}
func (m *configStorageMock) GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error) {
	return nil, nil
}
func (m *configStorageMock) QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
	return nil, nil
}
func (m *configStorageMock) CreateOrUpdateSession(ctx context.Context, session *types.Session) error {
	return nil
}
func (m *configStorageMock) GetSession(ctx context.Context, sessionID string) (*types.Session, error) {
	return nil, nil
}
func (m *configStorageMock) QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error) {
	return nil, nil
}
func (m *configStorageMock) SetMemory(ctx context.Context, memory *types.Memory) error { return nil }
func (m *configStorageMock) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	return nil, nil
}
func (m *configStorageMock) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	return nil
}
func (m *configStorageMock) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	return nil, nil
}
func (m *configStorageMock) SetVector(ctx context.Context, record *types.VectorRecord) error {
	return nil
}
func (m *configStorageMock) GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	return nil, nil
}
func (m *configStorageMock) DeleteVector(ctx context.Context, scope, scopeID, key string) error {
	return nil
}
func (m *configStorageMock) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	return 0, nil
}
func (m *configStorageMock) SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	return nil, nil
}
func (m *configStorageMock) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	return nil
}
func (m *configStorageMock) GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	return nil, nil
}
func (m *configStorageMock) AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *configStorageMock) ReleaseLock(ctx context.Context, lockID string) error { return nil }
func (m *configStorageMock) RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *configStorageMock) GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *configStorageMock) RegisterAgent(ctx context.Context, agent *types.AgentNode) error {
	return nil
}
func (m *configStorageMock) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	return nil, nil
}
func (m *configStorageMock) GetAgentVersion(ctx context.Context, id string, version string) (*types.AgentNode, error) {
	return nil, nil
}
func (m *configStorageMock) DeleteAgentVersion(ctx context.Context, id string, version string) error {
	return nil
}
func (m *configStorageMock) ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error) {
	return nil, nil
}
func (m *configStorageMock) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	return nil, nil
}
func (m *configStorageMock) ListAgentsByGroup(ctx context.Context, groupID string) ([]*types.AgentNode, error) {
	return nil, nil
}
func (m *configStorageMock) ListAgentGroups(ctx context.Context, teamID string) ([]types.AgentGroupSummary, error) {
	return nil, nil
}
func (m *configStorageMock) UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error {
	return nil
}
func (m *configStorageMock) UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error {
	return nil
}
func (m *configStorageMock) UpdateAgentHeartbeat(ctx context.Context, id string, version string, heartbeatTime time.Time) error {
	return nil
}
func (m *configStorageMock) UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error {
	return nil
}
func (m *configStorageMock) UpdateAgentVersion(ctx context.Context, id string, version string) error {
	return nil
}
func (m *configStorageMock) UpdateAgentTrafficWeight(ctx context.Context, id string, version string, weight int) error {
	return nil
}
func (m *configStorageMock) GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	return nil, nil
}
func (m *configStorageMock) GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
	return nil, nil
}
func (m *configStorageMock) StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	return nil
}
func (m *configStorageMock) GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
	return nil, nil
}
func (m *configStorageMock) QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error) {
	return nil, nil
}
func (m *configStorageMock) UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	return nil
}
func (m *configStorageMock) DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error {
	return nil
}
func (m *configStorageMock) ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
	return nil, nil
}
func (m *configStorageMock) StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (m *configStorageMock) GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error) {
	return nil, nil
}
func (m *configStorageMock) QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
	return nil, nil
}
func (m *configStorageMock) UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (m *configStorageMock) DeleteAgentPackage(ctx context.Context, packageID string) error {
	return nil
}
func (m *configStorageMock) SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error) {
	return nil, nil
}
func (m *configStorageMock) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	return nil
}
func (m *configStorageMock) GetExecutionEventBus() *events.ExecutionEventBus { return nil }
func (m *configStorageMock) GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent] {
	return nil
}
func (m *configStorageMock) GetExecutionLogEventBus() *events.EventBus[*types.ExecutionLogEntry] {
	return events.NewEventBus[*types.ExecutionLogEntry]()
}
func (m *configStorageMock) StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error {
	return nil
}
func (m *configStorageMock) GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error) {
	return nil, nil
}
func (m *configStorageMock) ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error) {
	return nil, nil
}
func (m *configStorageMock) StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error {
	return nil
}
func (m *configStorageMock) GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}
func (m *configStorageMock) ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}
func (m *configStorageMock) StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error {
	return nil
}
func (m *configStorageMock) GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error) {
	return nil, nil
}
func (m *configStorageMock) ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error) {
	return nil, nil
}
func (m *configStorageMock) StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error {
	return nil
}
func (m *configStorageMock) GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error) {
	return nil, nil
}
func (m *configStorageMock) ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	return nil, nil
}
func (m *configStorageMock) StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []storage.ComponentDIDRequest) error {
	return nil
}
func (m *configStorageMock) StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (m *configStorageMock) GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (m *configStorageMock) ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (m *configStorageMock) ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error) {
	return nil, nil
}
func (m *configStorageMock) CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error) {
	return 0, nil
}
func (m *configStorageMock) StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (m *configStorageMock) GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error) {
	return nil, nil
}
func (m *configStorageMock) ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error) {
	return nil, nil
}
func (m *configStorageMock) GetObservabilityWebhook(ctx context.Context) (*types.ObservabilityWebhookConfig, error) {
	return nil, nil
}
func (m *configStorageMock) SetObservabilityWebhook(ctx context.Context, config *types.ObservabilityWebhookConfig) error {
	return nil
}
func (m *configStorageMock) DeleteObservabilityWebhook(ctx context.Context) error { return nil }
func (m *configStorageMock) AddToDeadLetterQueue(ctx context.Context, event *types.ObservabilityEvent, errorMessage string, retryCount int) error {
	return nil
}
func (m *configStorageMock) GetDeadLetterQueueCount(ctx context.Context) (int64, error) {
	return 0, nil
}
func (m *configStorageMock) GetDeadLetterQueue(ctx context.Context, limit, offset int) ([]types.ObservabilityDeadLetterEntry, error) {
	return nil, nil
}
func (m *configStorageMock) DeleteFromDeadLetterQueue(ctx context.Context, ids []int64) error {
	return nil
}
func (m *configStorageMock) ClearDeadLetterQueue(ctx context.Context) error { return nil }
func (m *configStorageMock) GetAccessPolicies(ctx context.Context) ([]*types.AccessPolicy, error) {
	return nil, nil
}
func (m *configStorageMock) GetAccessPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error) {
	return nil, nil
}
func (m *configStorageMock) CreateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	return nil
}
func (m *configStorageMock) UpdateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	return nil
}
func (m *configStorageMock) DeleteAccessPolicy(ctx context.Context, id int64) error { return nil }
func (m *configStorageMock) StoreAgentTagVC(ctx context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error {
	return nil
}
func (m *configStorageMock) GetAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCRecord, error) {
	return nil, nil
}
func (m *configStorageMock) ListAgentTagVCs(ctx context.Context) ([]*types.AgentTagVCRecord, error) {
	return nil, nil
}
func (m *configStorageMock) RevokeAgentTagVC(ctx context.Context, agentID string) error { return nil }
func (m *configStorageMock) StoreDIDDocument(ctx context.Context, record *types.DIDDocumentRecord) error {
	return nil
}
func (m *configStorageMock) GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error) {
	return nil, nil
}
func (m *configStorageMock) GetDIDDocumentByAgentID(ctx context.Context, agentID string) (*types.DIDDocumentRecord, error) {
	return nil, nil
}
func (m *configStorageMock) RevokeDIDDocument(ctx context.Context, did string) error { return nil }
func (m *configStorageMock) ListDIDDocuments(ctx context.Context) ([]*types.DIDDocumentRecord, error) {
	return nil, nil
}
func (m *configStorageMock) ListAgentsByLifecycleStatus(ctx context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error) {
	return nil, nil
}
func (m *configStorageMock) QueryWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) ([]*types.WorkflowRun, error) {
	return nil, nil
}
func (m *configStorageMock) CountWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) (int, error) {
	return 0, nil
}
func (m *configStorageMock) StoreWorkflowRunEvent(ctx context.Context, event *types.WorkflowRunEvent) error {
	return nil
}
func (m *configStorageMock) ListWorkflowRunEvents(ctx context.Context, runID string, afterSeq *int64, limit int) ([]*types.WorkflowRunEvent, error) {
	return nil, nil
}

// --- helper ---

func setupConfigRouter(store *configStorageMock, reloadFn ConfigReloadFunc) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	h := NewConfigStorageHandlers(store, reloadFn)
	group := router.Group("")
	h.RegisterRoutes(group)
	return router
}

// --- tests ---

func TestConfigStorage_ListConfigs_Success(t *testing.T) {
	store := &configStorageMock{}
	entries := []*storage.ConfigEntry{
		{Key: "key1", Value: "value1"},
		{Key: "key2", Value: "value2"},
	}
	store.On("ListConfigs", mock.Anything).Return(entries, nil)

	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("GET", "/configs", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	store.AssertExpectations(t)
}

func TestConfigStorage_ListConfigs_Empty(t *testing.T) {
	store := &configStorageMock{}
	store.On("ListConfigs", mock.Anything).Return([]*storage.ConfigEntry{}, nil)

	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("GET", "/configs", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	store.AssertExpectations(t)
}

func TestConfigStorage_GetConfig_Found(t *testing.T) {
	store := &configStorageMock{}
	entry := &storage.ConfigEntry{Key: "mykey", Value: "myvalue"}
	store.On("GetConfig", mock.Anything, "mykey").Return(entry, nil)

	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("GET", "/configs/mykey", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	store.AssertExpectations(t)
}

func TestConfigStorage_GetConfig_NotFound(t *testing.T) {
	store := &configStorageMock{}
	store.On("GetConfig", mock.Anything, "missing").Return((*storage.ConfigEntry)(nil), nil)

	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("GET", "/configs/missing", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestConfigStorage_SetConfig_Success(t *testing.T) {
	store := &configStorageMock{}
	entry := &storage.ConfigEntry{Key: "newkey", Value: "newvalue"}
	store.On("SetConfig", mock.Anything, "newkey", "newvalue", "api").Return(nil)
	store.On("GetConfig", mock.Anything, "newkey").Return(entry, nil)

	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("PUT", "/configs/newkey", strings.NewReader("newvalue"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	store.AssertExpectations(t)
}

func TestConfigStorage_SetConfig_EmptyBody(t *testing.T) {
	store := &configStorageMock{}

	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("PUT", "/configs/mykey", strings.NewReader(""))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestConfigStorage_SetConfig_CustomUpdatedBy(t *testing.T) {
	store := &configStorageMock{}
	entry := &storage.ConfigEntry{Key: "k", Value: "v", UpdatedBy: "operator"}
	store.On("SetConfig", mock.Anything, "k", "v", "operator").Return(nil)
	store.On("GetConfig", mock.Anything, "k").Return(entry, nil)

	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("PUT", "/configs/k", strings.NewReader("v"))
	req.Header.Set("X-Updated-By", "operator")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	store.AssertExpectations(t)
}

func TestConfigStorage_DeleteConfig_Success(t *testing.T) {
	store := &configStorageMock{}
	store.On("DeleteConfig", mock.Anything, "mykey").Return(nil)

	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("DELETE", "/configs/mykey", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	store.AssertExpectations(t)
}

func TestConfigStorage_DeleteConfig_NotFound(t *testing.T) {
	store := &configStorageMock{}
	store.On("DeleteConfig", mock.Anything, "nokey").Return(errors.New("not found"))

	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("DELETE", "/configs/nokey", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
	store.AssertExpectations(t)
}

func TestConfigStorage_ReloadConfig_NotConfigured(t *testing.T) {
	store := &configStorageMock{}
	// reloadFn is nil → should return 503
	router := setupConfigRouter(store, nil)
	req := httptest.NewRequest("POST", "/configs/reload", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
}

func TestConfigStorage_ReloadConfig_Success(t *testing.T) {
	store := &configStorageMock{}
	called := false
	reloadFn := func() error {
		called = true
		return nil
	}

	router := setupConfigRouter(store, reloadFn)
	req := httptest.NewRequest("POST", "/configs/reload", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.True(t, called)
}

func TestConfigStorage_ReloadConfig_Error(t *testing.T) {
	store := &configStorageMock{}
	reloadFn := func() error {
		return errors.New("db error")
	}

	router := setupConfigRouter(store, reloadFn)
	req := httptest.NewRequest("POST", "/configs/reload", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
}
