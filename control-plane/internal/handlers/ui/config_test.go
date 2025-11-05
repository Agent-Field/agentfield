//go:build integration
// +build integration

package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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
)

// MockStorageProvider is a mock implementation of storage.StorageProvider
type MockStorageProvider struct {
	mock.Mock
}

func (m *MockStorageProvider) Initialize(ctx context.Context, config storage.StorageConfig) error {
	args := m.Called(ctx, config)
	return args.Error(0)
}

func (m *MockStorageProvider) Close(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *MockStorageProvider) HealthCheck(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *MockStorageProvider) StoreExecution(ctx context.Context, execution *types.AgentExecution) error {
	args := m.Called(ctx, execution)
	return args.Error(0)
}

func (m *MockStorageProvider) GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.AgentExecution), args.Error(1)
}

func (m *MockStorageProvider) QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentExecution), args.Error(1)
}

func (m *MockStorageProvider) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	args := m.Called(ctx, execution)
	return args.Error(0)
}

func (m *MockStorageProvider) RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error {
	return nil
}

func (m *MockStorageProvider) GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error) {
	return nil, nil
}

func (m *MockStorageProvider) ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error) {
	return nil, nil
}

func (m *MockStorageProvider) TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error) {
	return false, nil
}

func (m *MockStorageProvider) UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error {
	return nil
}

func (m *MockStorageProvider) HasExecutionWebhook(ctx context.Context, executionID string) (bool, error) {
	return false, nil
}

func (m *MockStorageProvider) ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error) {
	return map[string]bool{}, nil
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

func (m *MockStorageProvider) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	args := m.Called(ctx, executionID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.WorkflowExecution), args.Error(1)
}

func (m *MockStorageProvider) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	args := m.Called(ctx, executionID, updateFunc)
	return args.Error(0)
}

func (m *MockStorageProvider) QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.WorkflowExecution), args.Error(1)
}

func (m *MockStorageProvider) GetWorkflowStep(ctx context.Context, stepID string) (*types.WorkflowStep, error) {
	args := m.Called(ctx, stepID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.WorkflowStep), args.Error(1)
}

func (m *MockStorageProvider) QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error) {
	args := m.Called(ctx, rootWorkflowID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.WorkflowExecution), args.Error(1)
}

func (m *MockStorageProvider) CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error {
	args := m.Called(ctx, workflow)
	return args.Error(0)
}

func (m *MockStorageProvider) GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error) {
	args := m.Called(ctx, workflowID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.Workflow), args.Error(1)
}

func (m *MockStorageProvider) QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.Workflow), args.Error(1)
}

func (m *MockStorageProvider) CreateOrUpdateSession(ctx context.Context, session *types.Session) error {
	args := m.Called(ctx, session)
	return args.Error(0)
}

func (m *MockStorageProvider) GetSession(ctx context.Context, sessionID string) (*types.Session, error) {
	args := m.Called(ctx, sessionID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.Session), args.Error(1)
}

func (m *MockStorageProvider) QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.Session), args.Error(1)
}

func (m *MockStorageProvider) SetMemory(ctx context.Context, memory *types.Memory) error {
	args := m.Called(ctx, memory)
	return args.Error(0)
}

func (m *MockStorageProvider) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	args := m.Called(ctx, scope, scopeID, key)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.Memory), args.Error(1)
}

func (m *MockStorageProvider) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	args := m.Called(ctx, scope, scopeID, key)
	return args.Error(0)
}

func (m *MockStorageProvider) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	args := m.Called(ctx, scope, scopeID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.Memory), args.Error(1)
}

func (m *MockStorageProvider) RegisterAgent(ctx context.Context, agent *types.AgentNode) error {
	args := m.Called(ctx, agent)
	return args.Error(0)
}

func (m *MockStorageProvider) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.AgentNode), args.Error(1)
}

func (m *MockStorageProvider) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentNode), args.Error(1)
}

func (m *MockStorageProvider) UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error {
	args := m.Called(ctx, id, status)
	return args.Error(0)
}

func (m *MockStorageProvider) UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error {
	args := m.Called(ctx, id, status, expectedLastHeartbeat)
	return args.Error(0)
}

func (m *MockStorageProvider) UpdateAgentHeartbeat(ctx context.Context, id string, heartbeatTime time.Time) error {
	args := m.Called(ctx, id, heartbeatTime)
	return args.Error(0)
}

func (m *MockStorageProvider) UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error {
	args := m.Called(ctx, id, status)
	return args.Error(0)
}

func (m *MockStorageProvider) SetConfig(ctx context.Context, key string, value interface{}) error {
	args := m.Called(ctx, key, value)
	return args.Error(0)
}

func (m *MockStorageProvider) GetConfig(ctx context.Context, key string) (interface{}, error) {
	args := m.Called(ctx, key)
	return args.Get(0), args.Error(1)
}

func (m *MockStorageProvider) GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	args := m.Called(ctx, reasonerID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.ReasonerPerformanceMetrics), args.Error(1)
}

func (m *MockStorageProvider) GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
	args := m.Called(ctx, reasonerID, page, limit)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.ReasonerExecutionHistory), args.Error(1)
}

func (m *MockStorageProvider) StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	args := m.Called(ctx, config)
	return args.Error(0)
}

func (m *MockStorageProvider) GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
	args := m.Called(ctx, agentID, packageID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.AgentConfiguration), args.Error(1)
}

func (m *MockStorageProvider) QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentConfiguration), args.Error(1)
}

func (m *MockStorageProvider) UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	args := m.Called(ctx, config)
	return args.Error(0)
}

func (m *MockStorageProvider) DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error {
	args := m.Called(ctx, agentID, packageID)
	return args.Error(0)
}

func (m *MockStorageProvider) ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
	args := m.Called(ctx, agentID, packageID, config)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.ConfigurationValidationResult), args.Error(1)
}

func (m *MockStorageProvider) StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	args := m.Called(ctx, pkg)
	return args.Error(0)
}

func (m *MockStorageProvider) GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error) {
	args := m.Called(ctx, packageID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.AgentPackage), args.Error(1)
}

func (m *MockStorageProvider) QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentPackage), args.Error(1)
}

func (m *MockStorageProvider) UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	args := m.Called(ctx, pkg)
	return args.Error(0)
}

func (m *MockStorageProvider) DeleteAgentPackage(ctx context.Context, packageID string) error {
	args := m.Called(ctx, packageID)
	return args.Error(0)
}

func (m *MockStorageProvider) SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error) {
	args := m.Called(ctx, scope, scopeID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(<-chan types.MemoryChangeEvent), args.Error(1)
}

func (m *MockStorageProvider) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}

// Event operations
func (m *MockStorageProvider) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}

func (m *MockStorageProvider) GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	args := m.Called(ctx, filter)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.MemoryChangeEvent), args.Error(1)
}

// Distributed Lock operations
func (m *MockStorageProvider) AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error) {
	args := m.Called(ctx, key, timeout)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.DistributedLock), args.Error(1)
}

func (m *MockStorageProvider) ReleaseLock(ctx context.Context, lockID string) error {
	args := m.Called(ctx, lockID)
	return args.Error(0)
}

func (m *MockStorageProvider) RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error) {
	args := m.Called(ctx, lockID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.DistributedLock), args.Error(1)
}

func (m *MockStorageProvider) GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error) {
	args := m.Called(ctx, key)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.DistributedLock), args.Error(1)
}

// Execution event bus
func (m *MockStorageProvider) GetExecutionEventBus() *events.ExecutionEventBus {
	args := m.Called()
	if args.Get(0) == nil {
		return nil
	}
	return args.Get(0).(*events.ExecutionEventBus)
}

// DID Registry operations
func (m *MockStorageProvider) StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error {
	args := m.Called(ctx, did, didDocument, publicKey, privateKeyRef, derivationPath)
	return args.Error(0)
}

func (m *MockStorageProvider) GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error) {
	args := m.Called(ctx, did)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.DIDRegistryEntry), args.Error(1)
}

func (m *MockStorageProvider) ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.DIDRegistryEntry), args.Error(1)
}

// AgentField Server DID operations
func (m *MockStorageProvider) StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error {
	args := m.Called(ctx, agentfieldServerID, rootDID, masterSeed, createdAt, lastKeyRotation)
	return args.Error(0)
}

func (m *MockStorageProvider) GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error) {
	args := m.Called(ctx, agentfieldServerID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.AgentFieldServerDIDInfo), args.Error(1)
}

func (m *MockStorageProvider) ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentFieldServerDIDInfo), args.Error(1)
}

// Agent DID operations
func (m *MockStorageProvider) StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error {
	args := m.Called(ctx, agentID, agentDID, agentfieldServerDID, publicKeyJWK, derivationIndex)
	return args.Error(0)
}

func (m *MockStorageProvider) GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error) {
	args := m.Called(ctx, agentID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.AgentDIDInfo), args.Error(1)
}

func (m *MockStorageProvider) ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.AgentDIDInfo), args.Error(1)
}

// Component DID operations
func (m *MockStorageProvider) StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error {
	args := m.Called(ctx, componentID, componentDID, agentDID, componentType, componentName, derivationIndex)
	return args.Error(0)
}

func (m *MockStorageProvider) GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error) {
	args := m.Called(ctx, componentID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.ComponentDIDInfo), args.Error(1)
}

func (m *MockStorageProvider) ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	args := m.Called(ctx, agentDID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.ComponentDIDInfo), args.Error(1)
}

// Execution VC operations
func (m *MockStorageProvider) StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error {
	args := m.Called(ctx, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status, vcDocument, signature)
	return args.Error(0)
}

func (m *MockStorageProvider) GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error) {
	args := m.Called(ctx, vcID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.ExecutionVCInfo), args.Error(1)
}

func (m *MockStorageProvider) ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	args := m.Called(ctx, filters)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.ExecutionVCInfo), args.Error(1)
}

// Workflow VC operations
func (m *MockStorageProvider) StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error {
	args := m.Called(ctx, workflowVCID, workflowID, sessionID, componentVCIDs, status, startTime, endTime, totalSteps, completedSteps)
	return args.Error(0)
}

func (m *MockStorageProvider) GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error) {
	args := m.Called(ctx, workflowVCID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*types.WorkflowVCInfo), args.Error(1)
}

func (m *MockStorageProvider) ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error) {
	args := m.Called(ctx, workflowID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*types.WorkflowVCInfo), args.Error(1)
}

func setupTestRouter() (*gin.Engine, *MockStorageProvider) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	mockStorage := &MockStorageProvider{}
	configHandler := NewConfigHandler(mockStorage)

	v1 := router.Group("/api/ui/v1")
	{
		agents := v1.Group("/agents")
		{
			agents.GET("/:agentId/config/schema", configHandler.GetConfigSchemaHandler)
			agents.GET("/:agentId/config", configHandler.GetConfigHandler)
			agents.POST("/:agentId/config", configHandler.SetConfigHandler)
		}
	}

	return router, mockStorage
}

func TestGetConfigSchemaHandler(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		router, mockStorage := setupTestRouter()
		// Setup mock data
		schema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"api_key": map[string]interface{}{
					"type":        "string",
					"description": "API key for authentication",
				},
			},
		}
		schemaBytes, _ := json.Marshal(schema)

		agentPackage := &types.AgentPackage{
			ID:                  "test-package",
			Name:                "Test Package",
			Version:             "1.0.0",
			Description:         stringPtr("Test package description"),
			ConfigurationSchema: schemaBytes,
		}

		mockStorage.On("GetAgentPackage", "test-package").Return(agentPackage, nil)

		// Make request
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/config/schema?packageId=test-package", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Assert response
		assert.Equal(t, http.StatusOK, w.Code)

		var response map[string]interface{}
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "test-agent", response["agent_id"])
		assert.Equal(t, "test-package", response["package_id"])
		assert.NotNil(t, response["schema"])
		assert.NotNil(t, response["metadata"])

		mockStorage.AssertExpectations(t)
	})

	t.Run("Missing AgentId", func(t *testing.T) {
		router, _ := setupTestRouter()
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents//config/schema?packageId=test-package", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code) // Empty agentId should return 400
	})

	t.Run("Missing PackageId", func(t *testing.T) {
		router, _ := setupTestRouter()
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/config/schema", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "packageId query parameter is required", response.Error)
	})

	t.Run("Package Not Found", func(t *testing.T) {
		router, mockStorage := setupTestRouter()
		mockStorage.On("GetAgentPackage", "nonexistent-package").Return(nil, fmt.Errorf("package not found"))

		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/config/schema?packageId=nonexistent-package", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusNotFound, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "package not found", response.Error)

		mockStorage.AssertExpectations(t)
	})
}

func TestGetConfigHandler(t *testing.T) {
	t.Run("Success - Existing Configuration", func(t *testing.T) {
		router, mockStorage := setupTestRouter()
		config := &types.AgentConfiguration{
			ID:            1,
			AgentID:       "test-agent",
			PackageID:     "test-package",
			Configuration: map[string]interface{}{"api_key": "test-key"},
			Status:        types.ConfigurationStatusActive,
			Version:       1,
			CreatedAt:     time.Now(),
			UpdatedAt:     time.Now(),
		}

		mockStorage.On("GetAgentConfiguration", "test-agent", "test-package").Return(config, nil)

		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/config?packageId=test-package", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code)

		var response map[string]interface{}
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "test-agent", response["agent_id"])
		assert.Equal(t, "test-package", response["package_id"])
		assert.NotNil(t, response["configuration"])
		assert.Equal(t, "active", response["status"])
		assert.Equal(t, float64(1), response["version"])

		mockStorage.AssertExpectations(t)
	})

	t.Run("Success - No Existing Configuration", func(t *testing.T) {
		router, mockStorage := setupTestRouter()
		// Create a fresh mock for this test
		mockStorage.On("GetAgentConfiguration", "test-agent", "test-package").Return(nil, fmt.Errorf("configuration not found"))

		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/config?packageId=test-package", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code)

		var response map[string]interface{}
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "test-agent", response["agent_id"])
		assert.Equal(t, "test-package", response["package_id"])
		assert.Equal(t, map[string]interface{}{}, response["configuration"])
		assert.Equal(t, "draft", response["status"])
		assert.Equal(t, float64(0), response["version"])

		mockStorage.AssertExpectations(t)
	})

	t.Run("Missing PackageId", func(t *testing.T) {
		router, _ := setupTestRouter()
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/test-agent/config", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "packageId query parameter is required", response.Error)
	})
}

func TestSetConfigHandler(t *testing.T) {
	t.Run("Success - Create New Configuration", func(t *testing.T) {
		router, mockStorage := setupTestRouter()
		requestBody := SetConfigRequest{
			Configuration: map[string]interface{}{"api_key": "new-key"},
			Status:        stringPtr("active"),
		}
		bodyBytes, _ := json.Marshal(requestBody)

		validationResult := &types.ConfigurationValidationResult{
			Valid:  true,
			Errors: []string{},
		}

		mockStorage.On("ValidateAgentConfiguration", "test-agent", "test-package", requestBody.Configuration).Return(validationResult, nil)
		mockStorage.On("GetAgentConfiguration", "test-agent", "test-package").Return(nil, fmt.Errorf("configuration not found"))
		mockStorage.On("StoreAgentConfiguration", mock.AnythingOfType("*types.AgentConfiguration")).Return(nil)

		req, _ := http.NewRequest("POST", "/api/ui/v1/agents/test-agent/config?packageId=test-package", bytes.NewBuffer(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusCreated, w.Code)

		var response map[string]interface{}
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "test-agent", response["agent_id"])
		assert.Equal(t, "test-package", response["package_id"])
		assert.Equal(t, "configuration created successfully", response["message"])

		mockStorage.AssertExpectations(t)
	})

	t.Run("Success - Update Existing Configuration", func(t *testing.T) {
		router, mockStorage := setupTestRouter()
		requestBody := SetConfigRequest{
			Configuration: map[string]interface{}{"api_key": "updated-key"},
		}
		bodyBytes, _ := json.Marshal(requestBody)

		existingConfig := &types.AgentConfiguration{
			ID:            1,
			AgentID:       "test-agent",
			PackageID:     "test-package",
			Configuration: map[string]interface{}{"api_key": "old-key"},
			Status:        types.ConfigurationStatusActive,
			Version:       1,
		}

		validationResult := &types.ConfigurationValidationResult{
			Valid:  true,
			Errors: []string{},
		}

		mockStorage.On("ValidateAgentConfiguration", "test-agent", "test-package", requestBody.Configuration).Return(validationResult, nil)
		mockStorage.On("GetAgentConfiguration", "test-agent", "test-package").Return(existingConfig, nil)
		mockStorage.On("UpdateAgentConfiguration", mock.AnythingOfType("*types.AgentConfiguration")).Return(nil)

		req, _ := http.NewRequest("POST", "/api/ui/v1/agents/test-agent/config?packageId=test-package", bytes.NewBuffer(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code)

		var response map[string]interface{}
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "test-agent", response["agent_id"])
		assert.Equal(t, "test-package", response["package_id"])
		assert.Equal(t, "configuration updated successfully", response["message"])
		assert.Equal(t, float64(2), response["version"]) // Version should be incremented

		mockStorage.AssertExpectations(t)
	})

	t.Run("Validation Failed", func(t *testing.T) {
		router, mockStorage := setupTestRouter()
		requestBody := SetConfigRequest{
			Configuration: map[string]interface{}{"invalid_field": "value"},
		}
		bodyBytes, _ := json.Marshal(requestBody)

		validationResult := &types.ConfigurationValidationResult{
			Valid:  false,
			Errors: []string{"invalid_field is not allowed"},
		}

		mockStorage.On("ValidateAgentConfiguration", "test-agent", "test-package", requestBody.Configuration).Return(validationResult, nil)

		req, _ := http.NewRequest("POST", "/api/ui/v1/agents/test-agent/config?packageId=test-package", bytes.NewBuffer(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response map[string]interface{}
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "configuration validation failed", response["error"])
		assert.NotNil(t, response["validation_errors"])

		mockStorage.AssertExpectations(t)
	})

	t.Run("Invalid Request Body", func(t *testing.T) {
		router, _ := setupTestRouter()
		req, _ := http.NewRequest("POST", "/api/ui/v1/agents/test-agent/config?packageId=test-package", bytes.NewBuffer([]byte("invalid json")))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Contains(t, response.Error, "invalid request body")
	})

	t.Run("Missing PackageId", func(t *testing.T) {
		router, _ := setupTestRouter()
		requestBody := SetConfigRequest{
			Configuration: map[string]interface{}{"api_key": "test-key"},
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req, _ := http.NewRequest("POST", "/api/ui/v1/agents/test-agent/config", bytes.NewBuffer(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "packageId query parameter is required", response.Error)
	})
}

// Helper function to create string pointers
func stringPtr(s string) *string {
	return &s
}
