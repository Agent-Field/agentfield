package cli

import "errors"

// Node lifecycle commands (af list/stop/logs) gain a --json flag that emits
// the same {ok,data,error:{code,message,hint}} envelope agent mode uses.
// Default human output is unchanged; --json guarantees stdout carries exactly
// one JSON envelope and errors exit non-zero.

// nodeJSONSuccess emits an {ok:true,data:...} envelope on stdout.
func nodeJSONSuccess(data interface{}) error {
	return outputAgentJSON(AgentResponse{OK: true, Data: data})
}

// nodeJSONError emits an {ok:false,error:{...}} envelope on stdout and returns
// a cliExitError so the process exits non-zero with the message on stderr.
func nodeJSONError(code, message, hint string) error {
	_ = outputAgentJSON(AgentResponse{
		OK:    false,
		Error: &AgentError{Code: code, Message: message, Hint: hint},
	})
	return cliExitError{Code: 1, Err: errors.New(message)}
}
