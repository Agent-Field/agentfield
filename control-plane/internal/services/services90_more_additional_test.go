package services

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type observabilitySnapshotStore struct {
	*mockObservabilityStore
	agents  []*types.AgentNode
	listErr error
}

func (s *observabilitySnapshotStore) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	if s.listErr != nil {
		return nil, s.listErr
	}
	return s.agents, nil
}

func newTestObservabilityForwarder(store ObservabilityWebhookStore) *observabilityForwarder {
	ctx, cancel := context.WithCancel(context.Background())
	secret := "test-secret"
	return &observabilityForwarder{
		store:      store,
		cfg:        normalizeObservabilityConfig(ObservabilityForwarderConfig{QueueSize: 8, SnapshotInterval: time.Hour}),
		client:     nil,
		webhookCfg: &types.ObservabilityWebhookConfig{Enabled: true, URL: "https://example.test", Secret: &secret},
		eventQueue: make(chan types.ObservabilityEvent, 8),
		ctx:        ctx,
		cancel:     cancel,
	}
}

func TestObservabilityForwarderPublishSnapshotBuildsCounts(t *testing.T) {
	store := &observabilitySnapshotStore{
		mockObservabilityStore: newMockObservabilityStore(),
		agents: []*types.AgentNode{
			{
				ID:              "agent-1",
				BaseURL:         "http://agent-1",
				Version:         "1.0.0",
				HealthStatus:    types.HealthStatusActive,
				LifecycleStatus: types.AgentStatusReady,
				LastHeartbeat:   time.Unix(100, 0).UTC(),
				RegisteredAt:    time.Unix(90, 0).UTC(),
				Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-1", InputSchema: json.RawMessage(`{}`)}},
				Skills:          []types.SkillDefinition{{ID: "skill-1", Tags: []string{"approved"}}},
			},
			{
				ID:              "agent-2",
				BaseURL:         "http://agent-2",
				Version:         "2.0.0",
				HealthStatus:    types.HealthStatusInactive,
				LifecycleStatus: types.AgentStatusOffline,
				LastHeartbeat:   time.Unix(80, 0).UTC(),
				RegisteredAt:    time.Unix(70, 0).UTC(),
			},
		},
	}
	forwarder := newTestObservabilityForwarder(store)

	forwarder.publishSnapshot()

	select {
	case event := <-forwarder.eventQueue:
		require.Equal(t, string(events.SystemStateSnapshot), event.EventType)
		require.Equal(t, "system", event.EventSource)
		data, ok := event.Data.(map[string]interface{})
		require.True(t, ok)
		require.Equal(t, 2, data["total_agents"])
		require.Equal(t, 1, data["healthy_agents"])
		require.Equal(t, 1, data["unhealthy_agents"])
		require.Equal(t, 1, data["active_agents"])
		require.Equal(t, 1, data["inactive_agents"])
	case <-time.After(time.Second):
		t.Fatal("expected snapshot event")
	}

	forwarder.cancel()
}

func TestObservabilityForwarderSubscriptionsFilterHeartbeats(t *testing.T) {
	t.Run("execution events", func(t *testing.T) {
		forwarder := newTestObservabilityForwarder(newMockObservabilityStore())
		forwarder.wg.Add(1)
		go forwarder.subscribeExecutionEvents()
		require.Eventually(t, func() bool {
			return events.GlobalExecutionEventBus.GetSubscriberCount() > 0
		}, time.Second, 10*time.Millisecond)

		events.PublishExecutionStarted("exec-sub-1", "wf-sub-1", "agent-sub-1", map[string]interface{}{"step": 1})

		select {
		case event := <-forwarder.eventQueue:
			require.NotEmpty(t, event.EventType)
			require.Equal(t, "execution", event.EventSource)
		case <-time.After(2 * time.Second):
			t.Fatal("expected execution observability event")
		}

		forwarder.cancel()
		forwarder.wg.Wait()
	})

	t.Run("node events skip heartbeat", func(t *testing.T) {
		forwarder := newTestObservabilityForwarder(newMockObservabilityStore())
		forwarder.wg.Add(1)
		go forwarder.subscribeNodeEvents()
		require.Eventually(t, func() bool {
			return events.GlobalNodeEventBus.GetSubscriberCount() > 0
		}, time.Second, 10*time.Millisecond)

		events.PublishNodeHeartbeat()
		events.PublishNodeOffline("node-sub-1", map[string]interface{}{"reason": "test"})

		select {
		case event := <-forwarder.eventQueue:
			require.NotEqual(t, string(events.NodeHeartbeat), event.EventType)
			require.Equal(t, "node", event.EventSource)
		case <-time.After(2 * time.Second):
			t.Fatal("expected node observability event")
		}

		forwarder.cancel()
		forwarder.wg.Wait()
	})

	t.Run("reasoner events skip heartbeat", func(t *testing.T) {
		forwarder := newTestObservabilityForwarder(newMockObservabilityStore())
		forwarder.wg.Add(1)
		go forwarder.subscribeReasonerEvents()
		require.Eventually(t, func() bool {
			return events.GlobalReasonerEventBus.GetSubscriberCount() > 0
		}, time.Second, 10*time.Millisecond)

		events.PublishHeartbeat()
		events.PublishReasonerOffline("reasoner-sub-1", "node-sub-1", map[string]interface{}{"reason": "test"})

		select {
		case event := <-forwarder.eventQueue:
			require.NotEqual(t, string(events.Heartbeat), event.EventType)
			require.Equal(t, "reasoner", event.EventSource)
		case <-time.After(2 * time.Second):
			t.Fatal("expected reasoner observability event")
		}

		forwarder.cancel()
		forwarder.wg.Wait()
	})
}

func TestAccessPolicyAddPolicyErrorBranches(t *testing.T) {
	t.Run("create failure", func(t *testing.T) {
		service := NewAccessPolicyService(&mockAccessPolicyStorage{createErr: errors.New("create failed")})
		_, err := service.AddPolicy(context.Background(), &types.AccessPolicyRequest{
			Name:   "create-failure",
			Action: "allow",
		})
		require.ErrorContains(t, err, "failed to create access policy")
	})

	t.Run("cache reload failure", func(t *testing.T) {
		storage := &mockAccessPolicyStorage{getErr: errors.New("reload failed")}
		service := NewAccessPolicyService(storage)

		policy, err := service.AddPolicy(context.Background(), &types.AccessPolicyRequest{
			Name:   "reload-failure",
			Action: "allow",
		})
		require.ErrorContains(t, err, "policy created but cache reload failed")
		require.NotNil(t, policy)
		require.Len(t, storage.policies, 1)
	})
}

func TestWebhookDispatcherNotifyAndBackoffBranches(t *testing.T) {
	t.Run("returns context error when shutting down", func(t *testing.T) {
		store := newMockWebhookStore()
		store.webhooks["exec-shutdown"] = &types.ExecutionWebhook{
			ExecutionID: "exec-shutdown",
			Status:      types.ExecutionWebhookStatusPending,
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

			dispatcher := &webhookDispatcher{
				store: store,
				cfg:   normalizeWebhookConfig(WebhookDispatcherConfig{Timeout: time.Second, QueueSize: 1}),
				xctx:  ctx,
			}

		err := dispatcher.Notify(context.Background(), "exec-shutdown")
		require.ErrorIs(t, err, context.Canceled)
	})

	t.Run("refreshes schedule for non pending webhooks", func(t *testing.T) {
		store := newMockWebhookStore()
		lastAttempt := time.Now().Add(-time.Minute).UTC()
		lastErr := "previous failure"
		store.webhooks["exec-refresh"] = &types.ExecutionWebhook{
			ExecutionID:   "exec-refresh",
			Status:        types.ExecutionWebhookStatusDelivering,
			AttemptCount:  2,
			LastAttemptAt: &lastAttempt,
			LastError:     &lastErr,
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		dispatcher := &webhookDispatcher{
			store: store,
			cfg:   normalizeWebhookConfig(WebhookDispatcherConfig{Timeout: time.Second, QueueSize: 1}),
			xctx:  ctx,
			jobs:  make(chan webhookJob, 1),
		}

		require.NoError(t, dispatcher.Notify(context.Background(), "exec-refresh"))
		update, ok := store.stateUpdates["exec-refresh"]
		require.True(t, ok)
		require.Equal(t, types.ExecutionWebhookStatusPending, update.Status)
		require.NotNil(t, update.NextAttemptAt)
		require.NotNil(t, update.LastAttemptAt)
		require.NotNil(t, update.LastError)

		select {
		case job := <-dispatcher.jobs:
			require.Equal(t, "exec-refresh", job.ExecutionID)
		case <-time.After(time.Second):
			t.Fatal("expected refreshed webhook job")
		}
	})

	t.Run("compute backoff floors and caps", func(t *testing.T) {
		dispatcher := &webhookDispatcher{
			cfg: normalizeWebhookConfig(WebhookDispatcherConfig{
				RetryBackoff:    time.Second,
				MaxRetryBackoff: 3 * time.Second,
			}),
		}
		require.Equal(t, time.Second, dispatcher.computeBackoff(0))
		require.Equal(t, 2*time.Second, dispatcher.computeBackoff(2))
		require.Equal(t, 3*time.Second, dispatcher.computeBackoff(4))
	})
}

func TestStatusManagerUpdateFromHeartbeatLifecycleMappings(t *testing.T) {
	tests := []struct {
		name      string
		lifecycle types.AgentLifecycleStatus
		want      types.AgentLifecycleStatus
	}{
		{name: "ready maps active", lifecycle: types.AgentStatusReady, want: types.AgentStatusReady},
		{name: "starting preserved", lifecycle: types.AgentStatusStarting, want: types.AgentStatusStarting},
		{name: "offline preserved", lifecycle: types.AgentStatusOffline, want: types.AgentStatusOffline},
		{name: "degraded preserved", lifecycle: types.AgentStatusDegraded, want: types.AgentStatusDegraded},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, ctx := setupStatusManagerStorage(t)
			registerTestAgent(t, provider, ctx, "node-heartbeat-map")
			sm := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)

			require.NoError(t, sm.UpdateFromHeartbeat(ctx, "node-heartbeat-map", &tt.lifecycle, ""))

			agent, err := provider.GetAgent(ctx, "node-heartbeat-map")
			require.NoError(t, err)
			require.Equal(t, tt.want, agent.LifecycleStatus)
		})
	}
}

func TestObservabilityForwarderEnqueueEventBranches(t *testing.T) {
	t.Run("disabled config skips queueing", func(t *testing.T) {
		forwarder := newTestObservabilityForwarder(newMockObservabilityStore())
		forwarder.webhookCfg = &types.ObservabilityWebhookConfig{Enabled: false}
		forwarder.enqueueEvent(types.ObservabilityEvent{EventType: "disabled"})
		require.Len(t, forwarder.eventQueue, 0)
	})

	t.Run("full queue increments dropped", func(t *testing.T) {
		forwarder := newTestObservabilityForwarder(newMockObservabilityStore())
		forwarder.eventQueue = make(chan types.ObservabilityEvent, 1)
		forwarder.eventQueue <- types.ObservabilityEvent{EventType: "existing"}

		forwarder.enqueueEvent(types.ObservabilityEvent{EventType: "dropped"})
		require.Equal(t, int64(1), forwarder.dropped.Load())
	})
}

func TestWebhookDispatcherScanDueBranches(t *testing.T) {
	t.Run("no context is a no-op", func(t *testing.T) {
		dispatcher := &webhookDispatcher{store: newMockWebhookStore()}
		dispatcher.scanDue()
	})

	t.Run("closed context exits without enqueue", func(t *testing.T) {
		store := newMockWebhookStore()
		store.webhooks["exec-due-cancelled"] = &types.ExecutionWebhook{
			ExecutionID: "exec-due-cancelled",
			Status:      types.ExecutionWebhookStatusPending,
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		dispatcher := &webhookDispatcher{
			store: store,
			cfg:   normalizeWebhookConfig(WebhookDispatcherConfig{Timeout: time.Second, PollBatchSize: 10, QueueSize: 1}),
			xctx:  ctx,
		}
		dispatcher.scanDue()
		require.Equal(t, types.ExecutionWebhookStatusDelivering, store.webhooks["exec-due-cancelled"].Status)
	})
}

func TestVCServiceAndStorageTagListingBranches(t *testing.T) {
	service := NewVCService(&config.DIDConfig{Enabled: true}, nil, nil)
	records, err := service.ListAgentTagVCs()
	require.ErrorContains(t, err, "no storage provider configured")
	require.Nil(t, records)

	storage := NewVCStorageWithStorage(nil)
	workflowVC, err := storage.convertWorkflowVCInfo(nil)
	require.ErrorContains(t, err, "workflow VC info is nil")
	require.Nil(t, workflowVC)
}

func TestVCServiceWorkflowComplianceHelperBranch(t *testing.T) {
	service := NewVCService(&config.DIDConfig{Enabled: true}, nil, nil)
	require.False(t, service.checkWorkflowVCCompliance(&types.WorkflowVCDocument{
		Type: []string{"VerifiableCredential"},
	}))
}

func TestStatusManagerTransitionAndNotificationHelpers(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-helper")
	sm := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)

	require.True(t, sm.isValidTransition(types.AgentStateInactive, types.AgentStateActive))
	require.False(t, sm.isValidTransition(types.AgentStateActive, types.AgentStateStarting))

	done := make(chan string, 1)
	sm.AddEventHandler(&testStatusEventHandler{
		onStatusChanged: func(nodeID string, oldStatus, newStatus *types.AgentStatus) {
			done <- strings.Join([]string{nodeID, string(oldStatus.State), string(newStatus.State)}, ":")
		},
	})
	sm.AddEventHandler(&testStatusEventHandler{
		onStatusChanged: func(nodeID string, oldStatus, newStatus *types.AgentStatus) {
			panic("handler panic")
		},
	})

	sm.notifyStatusChanged("node-helper", &types.AgentStatus{State: types.AgentStateInactive}, &types.AgentStatus{State: types.AgentStateActive})

	select {
	case got := <-done:
		require.Equal(t, "node-helper:inactive:active", got)
	case <-time.After(time.Second):
		t.Fatal("expected status handler notification")
	}
}
