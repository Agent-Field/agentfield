package services

import "testing"

func TestExecutionsUIServiceSortExecutionsNoOp(t *testing.T) {
	service := &ExecutionsUIService{}
	executions := []ExecutionSummaryForUI{
		{ExecutionID: "b"},
		{ExecutionID: "a"},
	}

	service.sortExecutions(executions, "time", "desc")
}
