//go:build integration
// +build integration

package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockStorageProvider is a mock implementation of storage.StorageProvider for testing
type MockStorageProvider struct {
	mock.Mock
}

func (m *MockStorageProvider) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	args := m.Called(ctx, executionID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.WorkflowExecution), args.Error(1)
}

func (m *MockStorageProvider) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.AgentNode), args.Error(1)
}

func (m *MockStorageProvider) CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error) {
	args := m.Called(ctx, retentionPeriod, batchSize)
	return args.Int(0), args.Error(1)
}

func (m *MockStorageProvider) MarkStaleExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	args := m.Called(ctx, staleAfter, limit)
	return args.Int(0), args.Error(1)
}

// Add other required methods as no-ops for the interface
func (m *MockStorageProvider) Initialize(ctx context.Context, config interface{}) error { return nil }
func (m *MockStorageProvider) Close(ctx context.Context) error                          { return nil }
func (m *MockStorageProvider) HealthCheck(ctx context.Context) error                    { return nil }
func (m *MockStorageProvider) StoreExecution(ctx context.Context, execution *types.AgentExecution) error {
	return nil
}
func (m *MockStorageProvider) GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error) {
	return nil, nil
}
func (m *MockStorageProvider) QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	return nil
}
func (m *MockStorageProvider) StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error {
	return nil
}
func (m *MockStorageProvider) ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error) {
	return nil, nil
}
func (m *MockStorageProvider) ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error) {
	return map[string][]*types.ExecutionWebhookEvent{}, nil
}
func (m *MockStorageProvider) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	return nil
}
func (m *MockStorageProvider) ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreWorkflowRunEvent(ctx context.Context, event *types.WorkflowRunEvent) error {
	return nil
}
func (m *MockStorageProvider) ListWorkflowRunEvents(ctx context.Context, runID string, afterSeq *int64, limit int) ([]*types.WorkflowRunEvent, error) {
	return nil, nil
}
func (m *MockStorageProvider) QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *MockStorageProvider) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	return nil
}
func (m *MockStorageProvider) QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error) {
	return nil, nil
}
func (m *MockStorageProvider) QueryWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) ([]*types.WorkflowRun, error) {
	return nil, nil
}
func (m *MockStorageProvider) CountWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) (int, error) {
	return 0, nil
}
func (m *MockStorageProvider) CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error {
	return nil
}
func (m *MockStorageProvider) GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error) {
	return nil, nil
}
func (m *MockStorageProvider) QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
	return nil, nil
}
func (m *MockStorageProvider) CreateOrUpdateSession(ctx context.Context, session *types.Session) error {
	return nil
}
func (m *MockStorageProvider) GetSession(ctx context.Context, sessionID string) (*types.Session, error) {
	return nil, nil
}
func (m *MockStorageProvider) QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error) {
	return nil, nil
}
func (m *MockStorageProvider) SetMemory(ctx context.Context, memory *types.Memory) error { return nil }
func (m *MockStorageProvider) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	return nil, nil
}
func (m *MockStorageProvider) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	return nil
}
func (m *MockStorageProvider) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	return nil
}
func (m *MockStorageProvider) GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	return nil, nil
}
func (m *MockStorageProvider) AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *MockStorageProvider) ReleaseLock(ctx context.Context, lockID string) error { return nil }
func (m *MockStorageProvider) RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *MockStorageProvider) GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error) {
	return nil, nil
}
func (m *MockStorageProvider) RegisterAgent(ctx context.Context, agent *types.AgentNode) error {
	return nil
}
func (m *MockStorageProvider) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	return nil, nil
}
func (m *MockStorageProvider) UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error {
	return nil
}
func (m *MockStorageProvider) UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error {
	return nil
}
func (m *MockStorageProvider) UpdateAgentHeartbeat(ctx context.Context, id string, heartbeatTime time.Time) error {
	return nil
}
func (m *MockStorageProvider) UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error {
	return nil
}
func (m *MockStorageProvider) SetConfig(ctx context.Context, key string, value interface{}) error {
	return nil
}
func (m *MockStorageProvider) GetConfig(ctx context.Context, key string) (interface{}, error) {
	return nil, nil
}
func (m *MockStorageProvider) GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	return nil, nil
}
func (m *MockStorageProvider) GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	return nil
}
func (m *MockStorageProvider) GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
	return nil, nil
}
func (m *MockStorageProvider) QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error) {
	return nil, nil
}
func (m *MockStorageProvider) UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	return nil
}
func (m *MockStorageProvider) DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error {
	return nil
}
func (m *MockStorageProvider) ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (m *MockStorageProvider) GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error) {
	return nil, nil
}
func (m *MockStorageProvider) QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
	return nil, nil
}
func (m *MockStorageProvider) UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return nil
}
func (m *MockStorageProvider) DeleteAgentPackage(ctx context.Context, packageID string) error {
	return nil
}
func (m *MockStorageProvider) SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error) {
	return nil, nil
}
func (m *MockStorageProvider) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	return nil
}
func (m *MockStorageProvider) GetExecutionEventBus() interface{} { return nil }
func (m *MockStorageProvider) StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error {
	return nil
}
func (m *MockStorageProvider) GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error) {
	return nil, nil
}
func (m *MockStorageProvider) ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error {
	return nil
}
func (m *MockStorageProvider) GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}
func (m *MockStorageProvider) ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error {
	return nil
}
func (m *MockStorageProvider) GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error) {
	return nil, nil
}
func (m *MockStorageProvider) ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error {
	return nil
}
func (m *MockStorageProvider) GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error) {
	return nil, nil
}
func (m *MockStorageProvider) ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []interface{}) error {
	return nil
}
func (m *MockStorageProvider) StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (m *MockStorageProvider) GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (m *MockStorageProvider) ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	return nil, nil
}
func (m *MockStorageProvider) StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error {
	return nil
}
func (m *MockStorageProvider) GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error) {
	return nil, nil
}
func (m *MockStorageProvider) ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error) {
	return nil, nil
}

func TestBatchExecutionStatusHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name           string
		requestBody    BatchStatusRequest
		setupMocks     func(*MockStorageProvider)
		expectedStatus int
		expectedCount  int
	}{
		{
			name: "successful batch status check",
			requestBody: BatchStatusRequest{
				ExecutionIDs: []string{"exec-1", "exec-2"},
			},
			setupMocks: func(mockStorage *MockStorageProvider) {
				// Mock successful execution retrieval
				execution1 := &types.WorkflowExecution{
					ExecutionID: "exec-1",
					WorkflowID:  "workflow-1",
					AgentNodeID: "agent-1",
					ReasonerID:  "reasoner-1",
					Status:      string(types.ExecutionStatusSucceeded),
					StartedAt:   time.Now(),
					CompletedAt: &time.Time{},
					DurationMS:  &[]int64{1000}[0],
				}
				execution2 := &types.WorkflowExecution{
					ExecutionID: "exec-2",
					WorkflowID:  "workflow-2",
					AgentNodeID: "agent-1",
					ReasonerID:  "reasoner-2",
					Status:      "running",
					StartedAt:   time.Now(),
				}

				agent := &types.AgentNode{
					ID: "agent-1",
					Reasoners: []types.Reasoner{
						{ID: "reasoner-1"},
						{ID: "reasoner-2"},
					},
				}

				mockStorage.On("GetWorkflowExecution", mock.Anything, "exec-1").Return(execution1, nil)
				mockStorage.On("GetWorkflowExecution", mock.Anything, "exec-2").Return(execution2, nil)
				mockStorage.On("GetAgent", mock.Anything, "agent-1").Return(agent, nil)
			},
			expectedStatus: http.StatusOK,
			expectedCount:  2,
		},
		{
			name: "empty execution IDs",
			requestBody: BatchStatusRequest{
				ExecutionIDs: []string{},
			},
			setupMocks:     func(mockStorage *MockStorageProvider) {},
			expectedStatus: http.StatusBadRequest,
			expectedCount:  0,
		},
		{
			name: "too many execution IDs",
			requestBody: BatchStatusRequest{
				ExecutionIDs: make([]string, 51), // Exceeds max batch size of 50
			},
			setupMocks:     func(mockStorage *MockStorageProvider) {},
			expectedStatus: http.StatusBadRequest,
			expectedCount:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockStorage := &MockStorageProvider{}
			tt.setupMocks(mockStorage)

			// Create request
			requestBody, _ := json.Marshal(tt.requestBody)
			req, _ := http.NewRequest("POST", "/api/v1/executions/batch-status", bytes.NewBuffer(requestBody))
			req.Header.Set("Content-Type", "application/json")

			// Create response recorder
			w := httptest.NewRecorder()

			// Create gin context
			router := gin.New()
			router.POST("/api/v1/executions/batch-status", BatchExecutionStatusHandler(mockStorage))

			// Perform request
			router.ServeHTTP(w, req)

			// Assert status code
			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.expectedStatus == http.StatusOK {
				var response BatchStatusResponse
				err := json.Unmarshal(w.Body.Bytes(), &response)
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedCount, len(response))
			}

			mockStorage.AssertExpectations(t)
		})
	}
}

func TestCleanupOldExecutions(t *testing.T) {
	mockStorage := &MockStorageProvider{}

	// Test successful cleanup
	mockStorage.On("CleanupOldExecutions", mock.Anything, 24*time.Hour, 100).Return(5, nil)

	count, err := mockStorage.CleanupOldExecutions(context.Background(), 24*time.Hour, 100)

	assert.NoError(t, err)
	assert.Equal(t, 5, count)
	mockStorage.AssertExpectations(t)
}
