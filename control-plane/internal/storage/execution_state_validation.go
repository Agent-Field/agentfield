package storage

import (
	"fmt"

	"github.com/your-org/haxen/control-plane/pkg/types"
)

type InvalidExecutionStateTransitionError struct {
	ExecutionID  string
	CurrentState string
	NewState     string
	Reason       string
}

func (e *InvalidExecutionStateTransitionError) Error() string {
	return fmt.Sprintf("invalid execution state transition for %s: cannot change from %s to %s - %s",
		e.ExecutionID, e.CurrentState, e.NewState, e.Reason)
}

func validateExecutionStateTransition(currentStatus, newStatus string) error {
	currentStatus = types.NormalizeExecutionStatus(currentStatus)
	newStatus = types.NormalizeExecutionStatus(newStatus)

	validTransitions := map[string][]string{
		string(types.ExecutionStatusUnknown):   {string(types.ExecutionStatusPending)},
		string(types.ExecutionStatusPending):   {string(types.ExecutionStatusQueued), string(types.ExecutionStatusRunning), string(types.ExecutionStatusCancelled)},
		string(types.ExecutionStatusQueued):    {string(types.ExecutionStatusRunning), string(types.ExecutionStatusCancelled)},
		string(types.ExecutionStatusRunning):   {string(types.ExecutionStatusSucceeded), string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout)},
		string(types.ExecutionStatusSucceeded): {},
		string(types.ExecutionStatusFailed):    {},
		string(types.ExecutionStatusCancelled): {},
		string(types.ExecutionStatusTimeout):   {},
	}

	allowedStates, exists := validTransitions[currentStatus]
	if !exists {
		return fmt.Errorf("unknown current status: %s", currentStatus)
	}

	for _, allowed := range allowedStates {
		if newStatus == allowed {
			return nil
		}
	}

	if currentStatus == newStatus {
		return nil
	}

	return &InvalidExecutionStateTransitionError{
		CurrentState: currentStatus,
		NewState:     newStatus,
		Reason:       "transition not allowed by workflow execution state machine",
	}
}
