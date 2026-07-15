package agent

// ReasonerFailed is returned from a reasoner handler to report that the work
// ran but failed, while preserving a structured result on the execution
// record.
//
// Returning a value from a reasoner — even one whose payload says the work did
// not succeed — makes the execution handler record the execution as
// "succeeded": it only distinguishes "returned" from "errored", never
// inspecting the value. A build that completed zero issues and merged nothing
// would therefore surface as green.
//
// Return &ReasonerFailed{...} when the reasoner has determined its own work
// failed but you still want the structured Result preserved on the execution
// record. The async handler posts status="failed" to the control plane while
// also sending Result and ErrorDetails (the control plane stores the result
// payload regardless of terminal status), so the rich outcome — debt, DAG
// state, any PR that was opened — is not lost behind a bare error string.
//
// Mirrors the Python SDK's ReasonerFailed exception (agentfield.exceptions).
type ReasonerFailed struct {
	// Message is the human-readable failure summary; it becomes the execution
	// error string.
	Message string

	// Result is an optional structured result to preserve on the execution
	// record (e.g. a BuildResult). JSON-encoded by the handler before posting.
	Result any

	// ErrorDetails is optional structured error metadata mirrored onto the
	// status payload's error_details field.
	ErrorDetails any
}

// Error implements the error interface, returning the failure message.
func (e *ReasonerFailed) Error() string {
	return e.Message
}
