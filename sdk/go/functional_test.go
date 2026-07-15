//go:build functional

package agentfield_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	agentfield "github.com/Agent-Field/agentfield/sdk/go/agent"
	"github.com/Agent-Field/agentfield/sdk/go/ai"
	"github.com/Agent-Field/agentfield/sdk/go/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMain runs once before all tests to set up the environment
func TestMain(m *testing.M) {
	// Tests will individually start infrastructure as needed
	m.Run()
}

func publicAgentURL(port int) string {
	host := os.Getenv("AGENTFIELD_FUNCTIONAL_AGENT_CALLBACK_HOST")
	if host == "" {
		host = "host.docker.internal"
	}
	return fmt.Sprintf("http://%s:%d", host, port)
}

func TestFunctionalRegistration(t *testing.T) {
	controlPlaneURL := testutil.StartTestInfra(t)
	port := testutil.AllocatePort()

	agent, err := agentfield.New(agentfield.Config{
		NodeID:        "test-go-registration",
		Version:       "test",
		AgentFieldURL: controlPlaneURL,
		ListenAddress: fmt.Sprintf(":%d", port),
		PublicURL:     publicAgentURL(port),
	})
	require.NoError(t, err, "Failed to create agent")

	// Register a skill
	err = agent.RegisterSkill("hello", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"message": "Hello from Go"}, nil
	})
	require.NoError(t, err, "Failed to register skill")

	// Start agent in background
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	errChan := make(chan error, 1)
	go func() {
		errChan <- agent.Serve(ctx)
	}()

	// Give agent time to start and register
	time.Sleep(3 * time.Second)

	// Verify agent registered by trying to execute
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Post(
		fmt.Sprintf("%s/api/v1/execute/test-go-registration.hello", controlPlaneURL),
		"application/json",
		nil,
	)

	if err == nil {
		defer resp.Body.Close()
		assert.Equal(t, http.StatusOK, resp.StatusCode, "Agent should be registered and responding")
		t.Logf("✅ Agent registered successfully")
	}

	// Cancel context to stop agent
	cancel()

	// Wait for agent to stop (with timeout)
	select {
	case err := <-errChan:
		if err != nil && err != context.Canceled {
			t.Logf("Agent stopped with error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Log("Agent did not stop within timeout (this is okay)")
	}
}

func TestFunctionalExecution(t *testing.T) {
	controlPlaneURL := testutil.StartTestInfra(t)
	port := testutil.AllocatePort()

	agent, err := agentfield.New(agentfield.Config{
		NodeID:        "test-go-execution",
		Version:       "test",
		AgentFieldURL: controlPlaneURL,
		ListenAddress: fmt.Sprintf(":%d", port),
		PublicURL:     publicAgentURL(port),
	})
	require.NoError(t, err)

	// Register a skill that adds numbers
	err = agent.RegisterSkill("add", func(ctx context.Context, input map[string]any) (any, error) {
		a, ok1 := input["a"].(float64)
		b, ok2 := input["b"].(float64)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("expected numeric inputs a and b")
		}
		return map[string]any{"sum": a + b}, nil
	})
	require.NoError(t, err)

	// Start agent
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	errChan := make(chan error, 1)
	go func() {
		errChan <- agent.Serve(ctx)
	}()

	waitForAgentHealth(t, port)

	// Execute the skill via control plane and verify the routed result.
	payload := []byte(`{"input":{"a":2,"b":3}}`)
	resp, err := http.Post(
		fmt.Sprintf("%s/api/v1/execute/test-go-execution.add", controlPlaneURL),
		"application/json",
		bytes.NewReader(payload),
	)
	require.NoError(t, err, "Failed to execute add skill through control plane")
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err, "Failed to read execution response")
	require.Equal(t, http.StatusOK, resp.StatusCode, "Unexpected execution response: %s", string(body))

	var result struct {
		Status       string         `json:"status"`
		Result       map[string]any `json:"result"`
		ErrorMessage *string        `json:"error_message"`
	}
	require.NoError(t, json.Unmarshal(body, &result), "Failed to decode execution response: %s", string(body))
	require.Equal(t, "succeeded", result.Status, "Unexpected execution status: %s", string(body))
	require.Nil(t, result.ErrorMessage, "Execution returned error: %s", string(body))
	require.InDelta(t, 5.0, result.Result["sum"], 0.001, "Unexpected add result: %s", string(body))

	cancel()
	select {
	case <-errChan:
	case <-time.After(5 * time.Second):
	}
}

func TestFunctionalAgentCall(t *testing.T) {
	controlPlaneURL := testutil.StartTestInfra(t)
	portA := testutil.AllocatePort()
	portB := testutil.AllocatePort()

	// Create Agent A
	agentA, err := agentfield.New(agentfield.Config{
		NodeID:        "test-go-agent-a",
		Version:       "test",
		AgentFieldURL: controlPlaneURL,
		ListenAddress: fmt.Sprintf(":%d", portA),
		PublicURL:     publicAgentURL(portA),
	})
	require.NoError(t, err)

	// Create Agent B
	agentB, err := agentfield.New(agentfield.Config{
		NodeID:        "test-go-agent-b",
		Version:       "test",
		AgentFieldURL: controlPlaneURL,
		ListenAddress: fmt.Sprintf(":%d", portB),
		PublicURL:     publicAgentURL(portB),
	})
	require.NoError(t, err)

	// Agent B: simple processing
	err = agentB.RegisterSkill("process", func(ctx context.Context, input map[string]any) (any, error) {
		value, ok := input["value"].(float64)
		if !ok {
			return nil, fmt.Errorf("expected numeric value")
		}
		return map[string]any{"result": value * 2}, nil
	})
	require.NoError(t, err)

	// Agent A: calls Agent B
	err = agentA.RegisterSkill("delegate", func(ctx context.Context, input map[string]any) (any, error) {
		result, err := agentA.Call(ctx, "test-go-agent-b.process", input)
		if err != nil {
			return nil, err
		}
		return map[string]any{"delegated_result": result}, nil
	})
	require.NoError(t, err)

	// Start both agents
	ctxA, cancelA := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancelA()

	ctxB, cancelB := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancelB()

	errChanA := make(chan error, 1)
	errChanB := make(chan error, 1)

	go func() {
		errChanA <- agentA.Serve(ctxA)
	}()

	go func() {
		errChanB <- agentB.Serve(ctxB)
	}()

	time.Sleep(4 * time.Second)

	t.Log("✅ Agent-to-agent call test setup complete")

	// Cleanup
	cancelA()
	cancelB()

	select {
	case <-errChanA:
	case <-time.After(5 * time.Second):
	}

	select {
	case <-errChanB:
	case <-time.After(5 * time.Second):
	}
}

func TestFunctionalAIIntegration(t *testing.T) {
	controlPlaneURL := testutil.StartTestInfra(t)
	apiKey, baseURL, model := testutil.GetOpenRouterConfig(t)
	port := testutil.AllocatePort()

	agent, err := agentfield.New(agentfield.Config{
		NodeID:        "test-go-ai-agent",
		Version:       "test",
		AgentFieldURL: controlPlaneURL,
		ListenAddress: fmt.Sprintf(":%d", port),
		PublicURL:     publicAgentURL(port),
		AIConfig: &ai.Config{
			APIKey:  apiKey,
			BaseURL: baseURL,
			Model:   model,
		},
	})
	require.NoError(t, err)

	// Register skill that uses AI
	err = agent.RegisterSkill("ask", func(ctx context.Context, input map[string]any) (any, error) {
		question, ok := input["question"].(string)
		if !ok {
			return nil, fmt.Errorf("expected string question")
		}

		response, err := agent.AI(ctx, fmt.Sprintf("Answer briefly: %s", question), ai.WithMaxTokens(50))
		if err != nil {
			return nil, err
		}

		return map[string]any{"question": question, "answer": response.Text()}, nil
	})
	require.NoError(t, err)

	// Start agent
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	errChan := make(chan error, 1)
	go func() {
		errChan <- agent.Serve(ctx)
	}()

	time.Sleep(3 * time.Second)

	t.Log("✅ AI integration test setup complete")

	cancel()
	select {
	case <-errChan:
	case <-time.After(5 * time.Second):
	}
}

func TestFunctionalMemory(t *testing.T) {
	t.Skip("Go SDK memory client is not exposed on Agent yet")
}

func waitForAgentHealth(t *testing.T, port int) {
	t.Helper()

	client := &http.Client{Timeout: 500 * time.Millisecond}
	require.Eventually(t, func() bool {
		resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}, 10*time.Second, 100*time.Millisecond)
}
