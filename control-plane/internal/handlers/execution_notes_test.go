package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestAddExecutionNoteHandler_AppendsNoteAndPublishesEvent(t *testing.T) {
	gin.SetMode(gin.TestMode)

	executionID := "exec-1"
	runID := "wf-1" // run_id is the workflow ID equivalent

	storage := newTestExecutionStorage(nil)
	exec := &types.Execution{
		ExecutionID: executionID,
		RunID:       runID,
		Notes:       []types.ExecutionNote{},
		UpdatedAt:   time.Now(),
	}
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), exec))

	// Subscribe to event bus to ensure event emitted
	subscriber := storage.GetExecutionEventBus().Subscribe("test-subscriber")
	defer storage.GetExecutionEventBus().Unsubscribe("test-subscriber")

	router := gin.New()
	router.POST("/api/v1/executions/note", func(c *gin.Context) {
		c.Set("execution_id", executionID)
		AddExecutionNoteHandler(storage)(c)
	})

	reqBody := `{"message":"This is a note","tags":["debug"]}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/note", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload AddNoteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.True(t, payload.Success)
	require.Equal(t, "Note added successfully", payload.Message)
	require.Equal(t, []string{"debug"}, payload.Note.Tags)

	// Verify execution updated
	updated, err := storage.GetExecutionRecord(context.Background(), executionID)
	require.NoError(t, err)
	require.Len(t, updated.Notes, 1)
	require.Equal(t, "This is a note", updated.Notes[0].Message)

	// Ensure event published
	select {
	case evt := <-subscriber:
		require.Equal(t, runID, evt.WorkflowID)
		require.Equal(t, executionID, evt.ExecutionID)
		require.Equal(t, "note_added", evt.Status)
	case <-time.After(time.Second):
		t.Fatal("expected workflow note event")
	}
}

func TestGetExecutionNotesHandler_ReturnsFilteredNotes(t *testing.T) {
	gin.SetMode(gin.TestMode)

	executionID := "exec-2"
	storage := newTestExecutionStorage(nil)
	exec := &types.Execution{
		ExecutionID: executionID,
		Notes: []types.ExecutionNote{
			{Message: "note-one", Tags: []string{"debug"}},
			{Message: "note-two", Tags: []string{"info"}},
		},
	}
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), exec))

	router := gin.New()
	router.GET("/api/v1/executions/:execution_id/notes", GetExecutionNotesHandler(storage))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/exec-2/notes?tags=debug", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload GetNotesResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, executionID, payload.ExecutionID)
	require.Equal(t, 1, payload.Total)
	require.Equal(t, "note-one", payload.Notes[0].Message)
}
