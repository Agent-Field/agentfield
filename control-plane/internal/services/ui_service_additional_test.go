package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type uiStorageStub struct {
	storage.StorageProvider
	listAgentsResp []*types.AgentNode
	listAgentsErr  error
	agentsByID     map[string]*types.AgentNode
	getAgentErr    error
	packages       []*types.AgentPackage
	queryPkgsErr   error
	updatedHealth  map[string]types.HealthStatus
	updatedLifecycle map[string]types.AgentLifecycleStatus
	updatedHeartbeat map[string]time.Time
}

func (s *uiStorageStub) ListAgents(_ context.Context, _ types.AgentFilters) ([]*types.AgentNode, error) {
	if s.listAgentsErr != nil {
		return nil, s.listAgentsErr
	}
	return s.listAgentsResp, nil
}

func (s *uiStorageStub) GetAgent(_ context.Context, nodeID string) (*types.AgentNode, error) {
	if s.getAgentErr != nil {
		return nil, s.getAgentErr
	}
	if agent, ok := s.agentsByID[nodeID]; ok {
		return agent, nil
	}
	return nil, errors.New("not found")
}

func (s *uiStorageStub) QueryAgentPackages(_ context.Context, _ types.PackageFilters) ([]*types.AgentPackage, error) {
	if s.queryPkgsErr != nil {
		return nil, s.queryPkgsErr
	}
	return s.packages, nil
}

func (s *uiStorageStub) UpdateAgentHealth(_ context.Context, nodeID string, health types.HealthStatus) error {
	if s.updatedHealth == nil {
		s.updatedHealth = make(map[string]types.HealthStatus)
	}
	s.updatedHealth[nodeID] = health
	if agent, ok := s.agentsByID[nodeID]; ok {
		agent.HealthStatus = health
	}
	return nil
}

func (s *uiStorageStub) UpdateAgentLifecycleStatus(_ context.Context, nodeID string, lifecycle types.AgentLifecycleStatus) error {
	if s.updatedLifecycle == nil {
		s.updatedLifecycle = make(map[string]types.AgentLifecycleStatus)
	}
	s.updatedLifecycle[nodeID] = lifecycle
	if agent, ok := s.agentsByID[nodeID]; ok {
		agent.LifecycleStatus = lifecycle
	}
	return nil
}

func (s *uiStorageStub) UpdateAgentHeartbeat(_ context.Context, nodeID, _ string, heartbeat time.Time) error {
	if s.updatedHeartbeat == nil {
		s.updatedHeartbeat = make(map[string]time.Time)
	}
	s.updatedHeartbeat[nodeID] = heartbeat
	if agent, ok := s.agentsByID[nodeID]; ok {
		agent.LastHeartbeat = heartbeat
	}
	return nil
}

type uiAgentServiceStub struct {
	statusByName map[string]*domain.AgentStatus
	errByName    map[string]error
}

func (s *uiAgentServiceStub) RunAgent(string, domain.RunOptions) (*domain.RunningAgent, error) {
	return nil, errors.New("not implemented")
}

func (s *uiAgentServiceStub) StopAgent(string) error {
	return errors.New("not implemented")
}

func (s *uiAgentServiceStub) GetAgentStatus(name string) (*domain.AgentStatus, error) {
	if err, ok := s.errByName[name]; ok {
		return nil, err
	}
	if status, ok := s.statusByName[name]; ok {
		return status, nil
	}
	return nil, errors.New("not found")
}

func (s *uiAgentServiceStub) ListRunningAgents() ([]domain.RunningAgent, error) {
	return nil, errors.New("not implemented")
}

func TestUIServiceStatusReconciliationAndSummaries(t *testing.T) {
	now := time.Date(2026, 4, 8, 10, 0, 0, 0, time.UTC)
	nodeFromCache := &types.AgentNode{
		ID:              "node-cache",
		TeamID:          "team-a",
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusOffline,
		LastHeartbeat:   now,
		Reasoners:       []types.ReasonerDefinition{{ID: "r1"}},
		Skills:          []types.SkillDefinition{{ID: "s1"}},
	}
	nodeRunning := &types.AgentNode{
		ID:              "node-running",
		TeamID:          "team-b",
		Version:         "1.1.0",
		HealthStatus:    types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusOffline,
		LastHeartbeat:   now.Add(time.Minute),
	}
	nodeFallback := &types.AgentNode{
		ID:              "node-fallback",
		TeamID:          "team-c",
		Version:         "1.2.0",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusOffline,
		LastHeartbeat:   now.Add(2 * time.Minute),
	}

	statusManager := &StatusManager{
		statusCache: map[string]*cachedAgentStatus{
			"node-cache": {
				Status: &types.AgentStatus{
					State:           types.AgentStateActive,
					HealthScore:     99,
					HealthStatus:    types.HealthStatusActive,
					LifecycleStatus: types.AgentStatusReady,
				},
				Timestamp: now,
			},
		},
	}

	serviceWithCache := &UIService{
		statusManager:  statusManager,
		clients:        syncMap(),
		lastEventCache: make(map[string]NodeEvent),
		stopHeartbeat:  make(chan struct{}),
	}
	lifecycle, health := serviceWithCache.getReconciledNodeStatus("node-cache", nodeFromCache)
	require.Equal(t, types.AgentStatusReady, lifecycle)
	require.Equal(t, types.HealthStatusActive, health)

	service := &UIService{
		storage: &uiStorageStub{listAgentsResp: []*types.AgentNode{nodeFromCache, nodeRunning, nodeFallback}},
		agentService: &uiAgentServiceStub{
			statusByName: map[string]*domain.AgentStatus{
				"node-cache":   {IsRunning: true},
				"node-running": {IsRunning: true},
			},
			errByName: map[string]error{
				"node-fallback": errors.New("agent service unavailable"),
			},
		},
		clients:        syncMap(),
		lastEventCache: make(map[string]NodeEvent),
		stopHeartbeat:  make(chan struct{}),
	}

	summaries, count, err := service.GetNodesSummary(context.Background())
	require.NoError(t, err)
	require.Equal(t, 3, count)

	cacheSummary := summaries[0]
	require.Equal(t, types.AgentStatusReady, cacheSummary.LifecycleStatus)
	require.Equal(t, types.HealthStatusActive, cacheSummary.HealthStatus)
	require.Equal(t, 1, cacheSummary.ReasonerCount)
	require.Equal(t, 1, cacheSummary.SkillCount)

	runningSummary := summaries[1]
	require.Equal(t, types.AgentStatusReady, runningSummary.LifecycleStatus)
	require.Equal(t, types.HealthStatusActive, runningSummary.HealthStatus)

	fallbackSummary := summaries[2]
	require.Equal(t, types.AgentStatusReady, fallbackSummary.LifecycleStatus)
	require.Equal(t, types.HealthStatusActive, fallbackSummary.HealthStatus)
}

func TestUIServiceNodeDetailsAndPackageLookup(t *testing.T) {
	node := &types.AgentNode{ID: "node-1", TeamID: "team-a"}
	schema, err := json.Marshal(map[string]any{"agent_node": map[string]any{"node_id": "node-1"}})
	require.NoError(t, err)

	storageStub := &uiStorageStub{
		agentsByID: map[string]*types.AgentNode{"node-1": node},
		packages: []*types.AgentPackage{
			{ID: "bad", ConfigurationSchema: json.RawMessage(`{`)},
			{ID: "pkg-1", Version: "2.0.0", Status: types.PackageStatusRunning, ConfigurationSchema: schema},
		},
	}
	service := &UIService{
		storage:        storageStub,
		clients:        syncMap(),
		lastEventCache: make(map[string]NodeEvent),
		stopHeartbeat:  make(chan struct{}),
	}

	details, err := service.GetNodeDetails(context.Background(), "node-1")
	require.NoError(t, err)
	require.Equal(t, "node-1", details.ID)

	withPackage, err := service.GetNodeDetailsWithPackageInfo(context.Background(), "node-1")
	require.NoError(t, err)
	require.Equal(t, "pkg-1", withPackage.PackageInfo.PackageID)
	require.Equal(t, "2.0.0", withPackage.PackageInfo.Version)
	require.Equal(t, string(types.PackageStatusRunning), withPackage.PackageInfo.Status)

	_, err = service.findPackageByNodeID(context.Background(), "missing")
	require.ErrorContains(t, err, "no package found")

	noPackageService := &UIService{
		storage: &uiStorageStub{
			agentsByID:   map[string]*types.AgentNode{"node-1": node},
			queryPkgsErr: errors.New("packages unavailable"),
		},
		clients:        syncMap(),
		lastEventCache: make(map[string]NodeEvent),
		stopHeartbeat:  make(chan struct{}),
	}
	withoutPackage, err := noPackageService.GetNodeDetailsWithPackageInfo(context.Background(), "node-1")
	require.NoError(t, err)
	require.Nil(t, withoutPackage.PackageInfo)
}

func TestUIServiceEventCallbacksAndErrorPaths(t *testing.T) {
	service := newTestUIService()
	client := service.RegisterClient()
	events, done := collectNodeEvents(client)

	node := &types.AgentNode{
		ID:              "node-1",
		TeamID:          "team-a",
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   time.Now().UTC(),
		Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}, {ID: "reasoner-b"}},
	}

	service.OnAgentRegistered(node)
	service.OnNodeStatusChanged(node)
	service.OnAgentRemoved(node.ID)

	require.Equal(t, "node_registered", requireNodeEvent(t, events, 200*time.Millisecond).Type)
	require.Equal(t, "node_status_changed", requireNodeEvent(t, events, 200*time.Millisecond).Type)
	require.Equal(t, "reasoner_status_changed", requireNodeEvent(t, events, 200*time.Millisecond).Type)
	require.Equal(t, "reasoner_status_changed", requireNodeEvent(t, events, 200*time.Millisecond).Type)
	require.Equal(t, "reasoners_refresh", requireNodeEvent(t, events, 200*time.Millisecond).Type)
	require.Equal(t, "node_removed", requireNodeEvent(t, events, 200*time.Millisecond).Type)

	service.DeregisterClient(client)
	<-done

	_, _, err := (&UIService{storage: &uiStorageStub{listAgentsErr: errors.New("list failed")}}).GetNodesSummary(context.Background())
	require.ErrorContains(t, err, "list failed")

	nilManagerService := &UIService{}
	require.ErrorContains(t, nilManagerService.RefreshNodeStatus(context.Background(), "node-1"), "status manager not available")
	_, err = nilManagerService.GetUnifiedNodeStatus(context.Background(), "node-1")
	require.ErrorContains(t, err, "status manager not available")
	_, err = nilManagerService.GetNodeUnifiedStatus(context.Background(), "node-1")
	require.ErrorContains(t, err, "status manager not available")
	_, err = nilManagerService.BulkNodeStatus(context.Background(), []string{"node-1"})
	require.ErrorContains(t, err, "status manager not available")
	_, err = nilManagerService.RefreshAllNodeStatus(context.Background())
	require.ErrorContains(t, err, "status manager not available")
}

func TestNewUIServiceInitializesHeartbeatState(t *testing.T) {
	service := NewUIService(&uiStorageStub{}, nil, nil, nil)
	require.NotNil(t, service.heartbeatTicker)
	require.NotNil(t, service.stopHeartbeat)
	require.NotNil(t, service.lastEventCache)
	service.StopHeartbeat()
}

func TestUIServiceUnifiedStatusAndRefreshFlows(t *testing.T) {
	now := time.Now().UTC()
	storageStub := &uiStorageStub{
		listAgentsResp: []*types.AgentNode{
			{ID: "node-1", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, LastHeartbeat: now},
			nil,
			{ID: "node-2", HealthStatus: types.HealthStatusInactive, LifecycleStatus: types.AgentStatusOffline, LastHeartbeat: now.Add(-time.Minute)},
		},
		agentsByID: map[string]*types.AgentNode{
			"node-1": {ID: "node-1", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, LastHeartbeat: now},
			"node-2": {ID: "node-2", HealthStatus: types.HealthStatusInactive, LifecycleStatus: types.AgentStatusOffline, LastHeartbeat: now.Add(-time.Minute)},
		},
	}
	statusManager := NewStatusManager(storageStub, StatusManagerConfig{}, nil, nil)
	service := &UIService{
		storage:        storageStub,
		statusManager:  statusManager,
		clients:        syncMap(),
		lastEventCache: make(map[string]NodeEvent),
		stopHeartbeat:  make(chan struct{}),
	}

	status, err := service.GetUnifiedNodeStatus(context.Background(), "node-1")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, status.State)

	refreshed, err := service.GetNodeUnifiedStatus(context.Background(), "node-2")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateInactive, refreshed.State)

	statuses, err := service.BulkNodeStatus(context.Background(), []string{"node-1", "missing", "node-2"})
	require.NoError(t, err)
	require.Len(t, statuses, 2)
	require.Contains(t, statuses, "node-1")
	require.Contains(t, statuses, "node-2")

	require.NoError(t, service.RefreshNodeStatus(context.Background(), "node-1"))
	require.Equal(t, types.HealthStatusInactive, storageStub.updatedHealth["node-1"])

	allStatuses, err := service.RefreshAllNodeStatus(context.Background())
	require.NoError(t, err)
	require.Len(t, allStatuses, 2)
	require.Contains(t, allStatuses, "node-1")
	require.Contains(t, allStatuses, "node-2")
}

func TestUIServiceCacheEventEvictsOldestEntry(t *testing.T) {
	service := newTestUIService()
	oldest := time.Now().Add(-time.Hour)
	for i := 0; i < 100; i++ {
		service.lastEventCache[fmt.Sprintf("event-%d", i)] = NodeEvent{Timestamp: oldest.Add(time.Duration(i) * time.Second)}
	}

	service.cacheEvent(NodeEvent{
		Type:      "node_registered",
		Node:      AgentNodeSummaryForUI{ID: "node-101"},
		Timestamp: time.Now(),
	})

	require.Len(t, service.lastEventCache, 100)
	require.NotContains(t, service.lastEventCache, "event-0")
	require.Contains(t, service.lastEventCache, "node_registered:node-101")
}

func TestStatusManagerCheckTransitionTimeoutsCancelsPendingApproval(t *testing.T) {
	storageStub := &uiStorageStub{
		agentsByID: map[string]*types.AgentNode{
			"node-1": {ID: "node-1", LifecycleStatus: types.AgentStatusPendingApproval},
		},
	}
	statusManager := NewStatusManager(storageStub, StatusManagerConfig{MaxTransitionTime: time.Second}, nil, nil)
	statusManager.activeTransitions["node-1"] = &types.StateTransition{
		From:      types.AgentStateStarting,
		To:        types.AgentStateActive,
		StartedAt: time.Now().Add(-2 * time.Second),
	}

	statusManager.checkTransitionTimeouts()

	require.Empty(t, statusManager.activeTransitions)
}

func TestStatusManagerCheckTransitionTimeoutsCompletesAndPersists(t *testing.T) {
	storageStub := &uiStorageStub{
		agentsByID: map[string]*types.AgentNode{
			"node-2": {ID: "node-2", LifecycleStatus: types.AgentStatusStarting},
		},
	}
	statusManager := NewStatusManager(storageStub, StatusManagerConfig{MaxTransitionTime: time.Second}, nil, nil)
	statusManager.activeTransitions["node-2"] = &types.StateTransition{
		From:      types.AgentStateStarting,
		To:        types.AgentStateActive,
		StartedAt: time.Now().Add(-2 * time.Second),
	}
	statusManager.statusCache["node-2"] = &cachedAgentStatus{
		Status: &types.AgentStatus{
			State:           types.AgentStateActive,
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
			StateTransition: &types.StateTransition{
				From:      types.AgentStateStarting,
				To:        types.AgentStateActive,
				StartedAt: time.Now().Add(-2 * time.Second),
			},
			LastSeen: time.Now().UTC(),
			Source:   types.StatusSourceManual,
		},
		Timestamp: time.Now(),
	}

	statusManager.checkTransitionTimeouts()

	require.Empty(t, statusManager.activeTransitions)
	require.Equal(t, types.AgentStatusReady, storageStub.updatedLifecycle["node-2"])
}

func TestStatusManagerPersistStatus(t *testing.T) {
	storageStub := &uiStorageStub{agentsByID: map[string]*types.AgentNode{"node-1": {ID: "node-1"}}}
	statusManager := NewStatusManager(storageStub, StatusManagerConfig{}, nil, nil)

	err := statusManager.persistStatus(context.Background(), "node-1", "v1", &types.AgentStatus{
		State:           types.AgentStateActive,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusOffline,
		LastSeen:        time.Now().UTC(),
		Source:          types.StatusSourceHeartbeat,
	})
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusActive, storageStub.updatedHealth["node-1"])
	require.Equal(t, types.AgentStatusReady, storageStub.updatedLifecycle["node-1"])
	require.False(t, storageStub.updatedHeartbeat["node-1"].IsZero())

	err = statusManager.persistStatus(context.Background(), "node-1", "v1", &types.AgentStatus{
		State:           types.AgentStateStarting,
		HealthStatus:    types.HealthStatusUnknown,
		LifecycleStatus: types.AgentStatusOffline,
		LastSeen:        time.Now().UTC(),
		Source:          types.StatusSourceManual,
	})
	require.NoError(t, err)
	require.Equal(t, types.AgentStatusStarting, storageStub.updatedLifecycle["node-1"])

	err = statusManager.persistStatus(context.Background(), "node-1", "v1", &types.AgentStatus{
		State:           types.AgentStateInactive,
		HealthStatus:    types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusReady,
		LastSeen:        time.Now().UTC(),
		Source:          types.StatusSourceManual,
	})
	require.NoError(t, err)
	require.Equal(t, types.AgentStatusOffline, storageStub.updatedLifecycle["node-1"])
}

func syncMap() sync.Map {
	return sync.Map{}
}
