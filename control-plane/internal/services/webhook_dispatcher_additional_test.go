package services

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestWebhookDispatcherScanDueEnqueuesPendingJobs(t *testing.T) {
	store := newMockWebhookStore()
	store.webhooks["exec-due"] = &types.ExecutionWebhook{
		ExecutionID: "exec-due",
		Status:      types.ExecutionWebhookStatusPending,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	dispatcher := &webhookDispatcher{
		store: store,
		cfg: normalizeWebhookConfig(WebhookDispatcherConfig{
			Timeout:       time.Second,
			PollBatchSize: 10,
			QueueSize:     1,
		}),
		xctx: ctx,
		jobs: make(chan webhookJob, 1),
	}

	dispatcher.scanDue()

	select {
	case job := <-dispatcher.jobs:
		require.Equal(t, "exec-due", job.ExecutionID)
	case <-time.After(time.Second):
		t.Fatal("expected due webhook job to be enqueued")
	}
	require.Equal(t, types.ExecutionWebhookStatusDelivering, store.webhooks["exec-due"].Status)
}

func TestWebhookDispatcherHelpers(t *testing.T) {
	store := newMockWebhookStore()
	dispatcher := &webhookDispatcher{store: store}

	store.executions["exec-1"] = &types.Execution{ExecutionID: "exec-1", NodeID: "node-1", ReasonerID: "reasoner-1"}
	agent, err := setupWebhookHelperAgent()
	require.NoError(t, err)
	storeAgent := *agent
	storeAgent.ID = "node-1"

	t.Run("resolve target type prefers matching skill", func(t *testing.T) {
		storeWithAgent := &webhookHelperStore{mockWebhookStore: store, agent: &storeAgent}
		dispatcher.store = storeWithAgent

		targetType := dispatcher.resolveTargetType(context.Background(), &types.Execution{
			NodeID:     "node-1",
			ReasonerID: "skill-1",
		})
		require.Equal(t, "skill", targetType)
	})

	t.Run("decode execution payload variants", func(t *testing.T) {
		require.Nil(t, decodeExecutionPayload(nil))

		valid := decodeExecutionPayload(json.RawMessage(`{"ok":true}`))
		payloadMap, ok := valid.(map[string]interface{})
		require.True(t, ok)
		require.Equal(t, true, payloadMap["ok"])

		invalid := decodeExecutionPayload(json.RawMessage(`not-json`))
		require.Equal(t, "not-json", invalid)
	})
}

type webhookHelperStore struct {
	*mockWebhookStore
	agent *types.AgentNode
}

func (s *webhookHelperStore) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	return s.agent, nil
}

func setupWebhookHelperAgent() (*types.AgentNode, error) {
	return &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-1"}},
		Skills:    []types.SkillDefinition{{ID: "skill-1"}},
	}, nil
}
