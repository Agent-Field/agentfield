package server

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// stubStorage implements storage.StorageProvider with minimal functionality for testing
type stubStorage struct {
	eventBus *events.ExecutionEventBus
}

func newStubStorage() *stubStorage {
	return &stubStorage{
		eventBus: events.NewExecutionEventBus(),
	}
}

// Required methods for ExecuteHandler
func (s *stubStorage) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	return nil, nil
}
func (s *stubStorage) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	return nil
}
func (s *stubStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	return nil, nil
}
func (s *stubStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	return nil, nil
}
func (s *stubStorage) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	return nil, nil
}
func (s *stubStorage) RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error {
	return nil
}
func (s *stubStorage) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	return nil
}
func (s *stubStorage) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(*types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	return nil
}
func (s *stubStorage) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	return nil, nil
}
func (s *stubStorage) GetExecutionEventBus() *events.ExecutionEventBus {
	return s.eventBus
}

// Stub implementations for remaining StorageProvider methods
func (s *stubStorage) Initialize(ctx context.Context, config storage.StorageConfig) error { return nil }
func (s *stubStorage) Close(ctx context.Context) error                                    { return nil }
func (s *stubStorage) HealthCheck(ctx context.Context) error                              { return nil }
func (s *stubStorage) StoreExecution(ctx context.Context, execution *types.AgentExecution) error {
	return nil
}
func (s *stubStorage) GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error) {
	return nil, nil
}
func (s *stubStorage) QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error) {
	return nil, nil
}
func (s *stubStorage) QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (s *stubStorage) QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
	return nil, 0, nil
}
func (s *stubStorage) GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error) {
	return nil, nil
}
func (s *stubStorage) ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error) {
	return nil, nil
}
func (s *stubStorage) TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error) {
	return false, nil
}
func (s *stubStorage) UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error {
	return nil
}
func (s *stubStorage) HasExecutionWebhook(ctx context.Context, executionID string) (bool, error) {
	return false, nil
}
func (s *stubStorage) ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error) {
	return nil, nil
}
func (s *stubStorage) StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error {
	return nil
}
func (s *stubStorage) ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error) {
	return nil, nil
}
func (s *stubStorage) ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error) {
	return nil, nil
}
func (s *stubStorage) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	return nil
}
func (s *stubStorage) ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error) {
	return nil, nil
}
func (s *stubStorage) CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error) {
	return 0, nil
}
func (s *stubStorage) MarkStaleExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	return 0, nil
}
func (s *stubStorage) CleanupWorkflow(ctx context.Context, workflowID string, dryRun bool) (*types.WorkflowCleanupResult, error) {
	return nil, nil
}
func (s *stubStorage) QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (s *stubStorage) CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error {
	return nil
}
func (s *stubStorage) GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error) {
	return nil, nil
}
func (s *stubStorage) QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
	return nil, nil
}
func (s *stubStorage) CreateOrUpdateSession(ctx context.Context, session *types.Session) error {
	return nil
}
func (s *stubStorage) GetSession(ctx context.Context, sessionID string) (*types.Session, error) {
	return nil, nil
}
func (s *stubStorage) QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error) {
	return nil, nil
}

// Memory operations
func (s *stubStorage) SetMemory(ctx context.Context, memory *types.Memory) error { return nil }
func (s *stubStorage) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	return nil, nil
}
func (s *stubStorage) DeleteMemory(ctx context.Context, scope, scopeID, key string) error { return nil }
func (s *stubStorage) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	return nil, nil
}
func (s *stubStorage) SetVector(ctx context.Context, record *types.VectorRecord) error    { return nil }
func (s *stubStorage) DeleteVector(ctx context.Context, scope, scopeID, key string) error { return nil }
func (s *stubStorage) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	return 0, nil
}
func (s *stubStorage) SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	return nil, nil
}

// Event operations
func (s *stubStorage) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	return nil
}
func (s *stubStorage) GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	return nil, nil
}

// Distributed Lock operations
func (s *stubStorage) AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error) {
	return nil, nil
}
func (s *stubStorage) ReleaseLock(ctx context.Context, lockID string) error { return nil }
func (s *stubStorage) RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error) {
	return nil, nil
}
func (s *stubStorage) GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error) {
	return nil, nil
}

// Agent registry
func (s *stubStorage) RegisterAgent(ctx context.Context, agent *types.AgentNode) error { return nil }
func (s *stubStorage) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	return nil, nil
}
func (s *stubStorage) UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error {
	return nil
}
func (s *stubStorage) UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error {
	return nil
}
func (s *stubStorage) UpdateAgentHeartbeat(ctx context.Context, id string, heartbeatTime time.Time) error {
	return nil
}
func (s *stubStorage) UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error {
	return nil
}

// Configuration
func (s *stubStorage) SetConfig(ctx context.Context, key string, value interface{}) error { return nil }
func (s *stubStorage) GetConfig(ctx context.Context, key string) (interface{}, error) {
	return nil, nil
}

// Reasoner Performance and History
func (s *stubStorage) GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	return nil, nil
}
func (s *stubStorage) GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
	return nil, nil
}

// Agent Configuration Management
func (s *stubStorage) StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	return nil
}
func (s *stubStorage) GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
	return nil, nil
}
func (s *stubStorage) QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error) {
	return nil, nil
}
func (s *stubStorage) UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	return nil
}
func (s *stubStorage) DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error {
	return nil
}
func (s *stubStorage) ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
	return nil, nil
}

// Agent Package Management
func (s *stubStorage) StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (s *stubStorage) GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error) {
	return nil, nil
}
func (s *stubStorage) QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
	return nil, nil
}
func (s *stubStorage) UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (s *stubStorage) DeleteAgentPackage(ctx context.Context, packageID string) error { return nil }

// Real-time features
func (s *stubStorage) SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error) {
	return nil, nil
}
func (s *stubStorage) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	return nil
}
func (s *stubStorage) GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent] {
	return nil
}

// DID Registry operations
func (s *stubStorage) StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error {
	return nil
}
func (s *stubStorage) GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error) {
	return nil, nil
}
func (s *stubStorage) ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error) {
	return nil, nil
}

// AgentField Server DID operations
func (s *stubStorage) StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error {
	return nil
}
func (s *stubStorage) GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}
func (s *stubStorage) ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}

// Agent DID operations
func (s *stubStorage) StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error {
	return nil
}
func (s *stubStorage) GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error) {
	return nil, nil
}
func (s *stubStorage) ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error) {
	return nil, nil
}

// Component DID operations
func (s *stubStorage) StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error {
	return nil
}
func (s *stubStorage) GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error) {
	return nil, nil
}
func (s *stubStorage) ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	return nil, nil
}

// Multi-step DID operations
func (s *stubStorage) StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []storage.ComponentDIDRequest) error {
	return nil
}

// Execution VC operations
func (s *stubStorage) StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (s *stubStorage) GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (s *stubStorage) ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (s *stubStorage) ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error) {
	return nil, nil
}

// Workflow VC operations
func (s *stubStorage) StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (s *stubStorage) GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error) {
	return nil, nil
}
func (s *stubStorage) ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error) {
	return nil, nil
}
func (s *stubStorage) CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error) {
	return 0, nil
}

// stubPayloadStore implements services.PayloadStore
type stubPayloadStore struct{}

func (s *stubPayloadStore) SaveFromReader(ctx context.Context, r io.Reader) (*services.PayloadRecord, error) {
	return nil, nil
}
func (s *stubPayloadStore) SaveBytes(ctx context.Context, data []byte) (*services.PayloadRecord, error) {
	return nil, nil
}
func (s *stubPayloadStore) Open(ctx context.Context, uri string) (io.ReadCloser, error) {
	return nil, nil
}
func (s *stubPayloadStore) Remove(ctx context.Context, uri string) error {
	return nil
}

// stubWebhookDispatcher implements services.WebhookDispatcher
type stubWebhookDispatcher struct{}

func (s *stubWebhookDispatcher) Start(ctx context.Context) error {
	return nil
}
func (s *stubWebhookDispatcher) Stop(ctx context.Context) error {
	return nil
}
func (s *stubWebhookDispatcher) Notify(ctx context.Context, executionID string) error {
	return nil
}

func TestSetupRoutesRegistersMetricsAndUI(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{
		Router:            gin.New(),
		storage:           newStubStorage(),
		payloadStore:      &stubPayloadStore{},
		webhookDispatcher: &stubWebhookDispatcher{},
		config: &config.Config{
			UI:  config.UIConfig{Enabled: true, Mode: "embedded"},
			API: config.APIConfig{},
		},
	}

	srv.setupRoutes()

	t.Run("metrics endpoint", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodGet, "/metrics", nil)
		req.Header.Set("Origin", "http://localhost:5173")
		w := httptest.NewRecorder()
		srv.Router.ServeHTTP(w, req)
		require.Equal(t, http.StatusOK, w.Code)
		require.Equal(t, "http://localhost:5173", w.Header().Get("Access-Control-Allow-Origin"))
	})

	t.Run("root redirect", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodGet, "/", nil)
		w := httptest.NewRecorder()
		srv.Router.ServeHTTP(w, req)
		require.Equal(t, http.StatusMovedPermanently, w.Code)
		require.Equal(t, "/ui/", w.Header().Get("Location"))
	})
}

//nolint:unused // Reserved for future test cases
type stubHealthMonitor struct {
	*services.HealthMonitor
}

func TestUnregisterAgentFromMonitoringResponses(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{}

	t.Run("missing node id returns 400", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodDelete, "/internal/nodes//monitor", nil)
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Params = append(c.Params, gin.Param{Key: "node_id", Value: ""})
		c.Request = req

		srv.unregisterAgentFromMonitoring(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("health monitor unavailable", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodDelete, "/internal/nodes/node-1/monitor", nil)
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Params = append(c.Params, gin.Param{Key: "node_id", Value: "node-1"})
		c.Request = req

		srv.unregisterAgentFromMonitoring(c)
		require.Equal(t, http.StatusServiceUnavailable, w.Code)
	})

	t.Run("successful unregister", func(t *testing.T) {
		hm := services.NewHealthMonitor(nil, services.HealthMonitorConfig{}, nil, nil, nil, nil)
		req, _ := http.NewRequest(http.MethodDelete, "/internal/nodes/node-42/monitor", nil)
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Params = append(c.Params, gin.Param{Key: "node_id", Value: "node-42"})
		c.Request = req

		srv.healthMonitor = hm
		srv.unregisterAgentFromMonitoring(c)
		require.Equal(t, http.StatusOK, w.Code)
	})
}
