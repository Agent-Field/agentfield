package workflowstatus

import (
	"github.com/your-org/agentfield/control-plane/pkg/types"
	"strings"
	"time"
)

// AggregatedWorkflowStatus captures lifecycle information derived from execution nodes and pending steps.
type AggregatedWorkflowStatus struct {
	Status           string
	RootExecution    *types.WorkflowExecution
	Counts           map[string]int
	TotalExecutions  int
	ActiveExecutions int
	ActiveSteps      int
	Terminal         bool
	EarliestStart    time.Time
	LatestUpdate     time.Time
	LatestCompletion *time.Time
}

var (
	statusPriority = []string{
		string(types.ExecutionStatusFailed),
		string(types.ExecutionStatusTimeout),
		string(types.ExecutionStatusCancelled),
		string(types.ExecutionStatusRunning),
		string(types.ExecutionStatusQueued),
		string(types.ExecutionStatusPending),
		string(types.ExecutionStatusSucceeded),
	}

	terminalStatuses = map[string]struct{}{
		string(types.ExecutionStatusSucceeded): {},
		string(types.ExecutionStatusFailed):    {},
		string(types.ExecutionStatusTimeout):   {},
		string(types.ExecutionStatusCancelled): {},
	}

	activeStepStatuses = map[string]struct{}{
		"pending": {},
		"running": {},
	}
)

// AggregateExecutions inspects execution rows and workflow steps to produce a lifecycle summary for the workflow.
func AggregateExecutions(executions []*types.WorkflowExecution, steps []*types.WorkflowStep) AggregatedWorkflowStatus {
	result := AggregatedWorkflowStatus{
		Status:          string(types.ExecutionStatusUnknown),
		Counts:          make(map[string]int),
		TotalExecutions: len(executions),
		Terminal:        true,
	}

	if len(executions) == 0 && len(steps) == 0 {
		return result
	}

	var (
		earliestStart *time.Time
		latestUpdate  *time.Time
		latestDone    *time.Time
	)

	for _, exec := range executions {
		normalized := types.NormalizeExecutionStatus(exec.Status)
		result.Counts[normalized]++

		if earliestStart == nil || exec.StartedAt.Before(*earliestStart) {
			t := exec.StartedAt
			earliestStart = &t
		}

		if latestUpdate == nil || exec.UpdatedAt.After(*latestUpdate) {
			t := exec.UpdatedAt
			latestUpdate = &t
		}

		if exec.CompletedAt != nil {
			if latestDone == nil || exec.CompletedAt.After(*latestDone) {
				t := *exec.CompletedAt
				latestDone = &t
			}
		}

		if exec.ParentExecutionID == nil && result.RootExecution == nil {
			result.RootExecution = exec
		}

		switch normalized {
		case string(types.ExecutionStatusRunning), string(types.ExecutionStatusQueued), string(types.ExecutionStatusPending):
			result.ActiveExecutions++
		}

		if _, ok := terminalStatuses[normalized]; !ok {
			result.Terminal = false
		}
	}

	activeSteps := 0
	for _, step := range steps {
		status := strings.ToLower(strings.TrimSpace(step.Status))
		if _, ok := activeStepStatuses[status]; ok {
			activeSteps++
			result.Terminal = false
		}
		if status == "failed" {
			result.Counts[string(types.ExecutionStatusFailed)]++
		}
		if status == "cancelled" {
			result.Counts[string(types.ExecutionStatusCancelled)]++
		}
	}

	if earliestStart != nil {
		result.EarliestStart = *earliestStart
	}
	if latestUpdate != nil {
		result.LatestUpdate = *latestUpdate
	}
	if latestDone != nil {
		result.LatestCompletion = new(time.Time)
		*result.LatestCompletion = *latestDone
	}

	if activeSteps > 0 {
		result.ActiveSteps = activeSteps
	}

	if result.ActiveSteps > 0 {
		result.ActiveExecutions += result.ActiveSteps
	}

	baseStatus := string(types.ExecutionStatusUnknown)
	if result.RootExecution != nil {
		baseStatus = types.NormalizeExecutionStatus(result.RootExecution.Status)
		if baseStatus == "" {
			baseStatus = string(types.ExecutionStatusUnknown)
		}
	}

	if baseStatus == string(types.ExecutionStatusUnknown) {
		for _, candidate := range statusPriority {
			if result.Counts[candidate] > 0 {
				baseStatus = candidate
				break
			}
		}
	}

	if result.ActiveExecutions > 0 {
		switch baseStatus {
		case string(types.ExecutionStatusQueued), string(types.ExecutionStatusPending), string(types.ExecutionStatusRunning):
			result.Status = baseStatus
		default:
			result.Status = string(types.ExecutionStatusRunning)
		}
		result.Terminal = false
	} else if baseStatus != string(types.ExecutionStatusUnknown) {
		result.Status = baseStatus
	} else if result.Terminal && result.Counts[string(types.ExecutionStatusSucceeded)] == result.TotalExecutions && result.TotalExecutions > 0 {
		result.Status = string(types.ExecutionStatusSucceeded)
	} else {
		result.Status = string(types.ExecutionStatusUnknown)
	}

	return result
}
