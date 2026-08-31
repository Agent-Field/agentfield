package agentic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

type runOverviewMetadataStorage struct {
	*handlerTestStorage
	run *types.WorkflowRun
}

func (s *runOverviewMetadataStorage) GetWorkflowRun(context.Context, string) (*types.WorkflowRun, error) {
	return s.run, nil
}

func TestRunOverviewRunMetadataPresenceAndEnvelopeLocation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	execution := []*types.Execution{{ExecutionID: "exec", RunID: "run-1", AgentNodeID: "node", Status: "succeeded"}}
	for _, test := range []struct {
		name    string
		store   *runOverviewMetadataStorage
		present bool
	}{
		{"absent", &runOverviewMetadataStorage{handlerTestStorage: &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}}, false},
		{"present", &runOverviewMetadataStorage{handlerTestStorage: &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}, run: &types.WorkflowRun{RunID: "run-1", Metadata: json.RawMessage(`{"run":{"display_name":"Release","labels":["smoke"]}}`)}}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			test.store.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return(execution, nil)
			router := gin.New()
			router.GET("/runs/:run_id", RunOverviewHandler(test.store))
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/runs/run-1", nil))
			require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
			var envelope map[string]interface{}
			require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &envelope))
			_, topLevel := envelope["run_metadata"]
			require.False(t, topLevel)
			data := envelope["data"].(map[string]interface{})
			metadata, exists := data["run_metadata"]
			require.Equal(t, test.present, exists)
			if test.present {
				require.Equal(t, "Release", metadata.(map[string]interface{})["display_name"])
			}
		})
	}
}
