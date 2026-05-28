package did

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVCGenerator_GenerateExecutionVC_Success(t *testing.T) {
	var receivedReq VCGenerationRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/execution/vc", r.URL.Path)
		json.NewDecoder(r.Body).Decode(&receivedReq)

		vc := ExecutionVC{
			VCID:        "vc-gen-1",
			ExecutionID: "exec-100",
			WorkflowID:  "wf-200",
			IssuerDID:   "did:web:localhost:agents:caller",
			TargetDID:   "did:web:localhost:agents:target",
			Status:      "completed",
			InputHash:   "sha256:abc",
			OutputHash:  "sha256:def",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vc)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)

	mgr := NewManager(client, logger)
	mgr.SetIdentityFromCredentials("did:web:localhost:agents:test", "key")

	gen := NewVCGenerator(client, mgr, logger)
	gen.SetEnabled(true)

	execCtx := ExecutionContext{
		ExecutionID: "exec-100",
		WorkflowID:  "wf-200",
		SessionID:   "sess-300",
	}

	input := map[string]any{"query": "test"}
	output := map[string]any{"result": "ok"}

	vc, err := gen.GenerateExecutionVC(
		context.Background(),
		execCtx,
		input,
		output,
		"succeeded",
		"",
		150,
	)

	require.NoError(t, err)
	require.NotNil(t, vc)
	assert.Equal(t, "vc-gen-1", vc.VCID)
	assert.Equal(t, "exec-100", vc.ExecutionID)

	// Verify request was properly constructed.
	assert.Equal(t, "exec-100", receivedReq.ExecutionContext.ExecutionID)
	assert.Equal(t, "wf-200", receivedReq.ExecutionContext.WorkflowID)
	assert.Equal(t, "succeeded", receivedReq.Status)
	assert.Equal(t, int64(150), receivedReq.DurationMS)

	// AgentNodeDID should be filled from manager; CallerDID comes from
	// X-Caller-DID header (not auto-filled to avoid misattribution).
	assert.Equal(t, "did:web:localhost:agents:test", receivedReq.ExecutionContext.AgentNodeDID)
	assert.Empty(t, receivedReq.ExecutionContext.CallerDID)

	// Input/output should be base64-encoded JSON.
	assert.NotEmpty(t, receivedReq.InputData)
	inputBytes, err := base64.StdEncoding.DecodeString(receivedReq.InputData)
	require.NoError(t, err)
	var decodedInput map[string]any
	json.Unmarshal(inputBytes, &decodedInput)
	assert.Equal(t, "test", decodedInput["query"])
}

func TestVCGenerator_Disabled(t *testing.T) {
	gen := NewVCGenerator(NewClient("http://unused"), nil, newTestLogger())
	// Not enabled — should return nil, nil
	vc, err := gen.GenerateExecutionVC(
		context.Background(),
		ExecutionContext{ExecutionID: "exec-1"},
		nil, nil, "succeeded", "", 0,
	)
	assert.NoError(t, err)
	assert.Nil(t, vc)
}

func TestVCGenerator_EnableDisable(t *testing.T) {
	gen := NewVCGenerator(NewClient("http://unused"), nil, newTestLogger())
	assert.False(t, gen.IsEnabled())

	gen.SetEnabled(true)
	assert.True(t, gen.IsEnabled())

	gen.SetEnabled(false)
	assert.False(t, gen.IsEnabled())
}

func TestVCGenerator_GenerateExecutionVC_WithError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req VCGenerationRequest
		json.NewDecoder(r.Body).Decode(&req)

		assert.Equal(t, "failed", req.Status)
		assert.Equal(t, "division by zero", req.ErrorMessage)

		vc := ExecutionVC{
			VCID:   "vc-error",
			Status: "failed",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vc)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	gen := NewVCGenerator(client, nil, newTestLogger())
	gen.SetEnabled(true)

	vc, err := gen.GenerateExecutionVC(
		context.Background(),
		ExecutionContext{ExecutionID: "exec-fail"},
		map[string]any{"x": 1},
		nil,
		"failed",
		"division by zero",
		50,
	)

	require.NoError(t, err)
	assert.Equal(t, "vc-error", vc.VCID)
}

func TestVCGenerator_GenerateExecutionVC_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "vc generation failed"}`))
	}))
	defer server.Close()

	client := NewClient(server.URL)
	gen := NewVCGenerator(client, nil, newTestLogger())
	gen.SetEnabled(true)

	vc, err := gen.GenerateExecutionVC(
		context.Background(),
		ExecutionContext{ExecutionID: "exec-1"},
		nil, nil, "succeeded", "", 0,
	)

	assert.Error(t, err)
	assert.Nil(t, vc)
	assert.Contains(t, err.Error(), "VC generation failed")
}

func TestVCGenerator_ExportWorkflowVCChain(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/did/workflow/wf-export/vc-chain", r.URL.Path)

		chain := WorkflowVCChain{
			WorkflowID: "wf-export",
			ExecutionVCs: []ExecutionVC{
				{VCID: "vc-1"},
				{VCID: "vc-2"},
				{VCID: "vc-3"},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chain)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	gen := NewVCGenerator(client, nil, newTestLogger())

	chain, err := gen.ExportWorkflowVCChain(context.Background(), "wf-export")
	require.NoError(t, err)
	assert.Equal(t, "wf-export", chain.WorkflowID)
	assert.Len(t, chain.ExecutionVCs, 3)
}

func TestVCGenerator_NilInput(t *testing.T) {
	var receivedReq VCGenerationRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedReq)
		vc := ExecutionVC{VCID: "vc-nil"}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vc)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	gen := NewVCGenerator(client, nil, newTestLogger())
	gen.SetEnabled(true)

	vc, err := gen.GenerateExecutionVC(
		context.Background(),
		ExecutionContext{ExecutionID: "exec-nil"},
		nil, nil, "succeeded", "", 0,
	)

	require.NoError(t, err)
	assert.Equal(t, "vc-nil", vc.VCID)
	// Nil input/output should produce empty strings.
	assert.Empty(t, receivedReq.InputData)
	assert.Empty(t, receivedReq.OutputData)
}

func TestEncodeData(t *testing.T) {
	tests := []struct {
		name  string
		input any
		empty bool
	}{
		{"nil", nil, true},
		{"map", map[string]any{"key": "value"}, false},
		{"string", "hello", false},
		{"number", 42, false},
		{"slice", []int{1, 2, 3}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := encodeData(tt.input)
			if tt.empty {
				assert.Empty(t, result)
				return
			}

			// Should be valid base64
			decoded, err := base64.StdEncoding.DecodeString(result)
			require.NoError(t, err)
			assert.NotEmpty(t, decoded)

			// Should be valid JSON
			var parsed any
			err = json.Unmarshal(decoded, &parsed)
			assert.NoError(t, err)
		})
	}
}
