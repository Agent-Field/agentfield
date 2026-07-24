package cli

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/spf13/cobra"
)

// newAgentExecCmd groups execution steering verbs for agent mode. Each verb is
// a thin wrapper over the existing /api/v1/executions/:execution_id/* REST
// endpoints and emits the standard AgentResponse JSON envelope.
func newAgentExecCmd() *cobra.Command {
	execCmd := &cobra.Command{
		Use:   "exec",
		Short: "Steer executions: pause, resume, cancel, restart, approvals",
		Run: func(cmd *cobra.Command, args []string) {
			agentOutput(map[string]interface{}{
				"message": "Use an exec subcommand",
				"available": []string{
					"pause",
					"resume",
					"cancel",
					"restart",
					"approval-status",
					"approve",
				},
			})
		},
	}

	execCmd.AddCommand(newAgentExecActionCmd("pause", "Pause a running execution", true))
	execCmd.AddCommand(newAgentExecActionCmd("resume", "Resume a paused execution", false))
	execCmd.AddCommand(newAgentExecActionCmd("cancel", "Cancel an execution", true))
	execCmd.AddCommand(newAgentExecRestartCmd())
	execCmd.AddCommand(newAgentExecApprovalStatusCmd())
	execCmd.AddCommand(newAgentExecApproveCmd())

	return execCmd
}

// requireExecutionID trims and validates the --id flag shared by exec verbs.
// On failure it emits an error envelope and exits non-zero via agentError.
func requireExecutionID(executionID, verb string) string {
	trimmed := strings.TrimSpace(executionID)
	if trimmed == "" {
		agentError(
			"missing_required_flag",
			"--id is required",
			fmt.Sprintf("Provide an execution ID, for example: af agent exec %s --id exec_123", verb),
		)
	}
	return trimmed
}

// newAgentExecActionCmd builds pause/resume/cancel verbs. withReason controls
// whether an optional --reason flag is sent in the request body.
func newAgentExecActionCmd(action, short string, withReason bool) *cobra.Command {
	var executionID string
	var reason string

	cmd := &cobra.Command{
		Use:   action,
		Short: short,
		Run: func(cmd *cobra.Command, args []string) {
			id := requireExecutionID(executionID, action)

			var body interface{}
			if withReason {
				payload := map[string]string{}
				if v := strings.TrimSpace(reason); v != "" {
					payload["reason"] = v
				}
				body = payload
			}

			proxyToServer(http.MethodPost, "/api/v1/executions/"+url.PathEscape(id)+"/"+action, body)
		},
	}

	cmd.Flags().StringVar(&executionID, "id", "", "Execution ID")
	if withReason {
		cmd.Flags().StringVar(&reason, "reason", "", "Reason for the action")
	}
	return cmd
}

func newAgentExecRestartCmd() *cobra.Command {
	var executionID string
	opts := executionActionOptions{
		scope: "workflow",
		reuse: "succeeded-before",
	}

	cmd := &cobra.Command{
		Use:   "restart",
		Short: "Restart a workflow from an execution point",
		Run: func(cmd *cobra.Command, args []string) {
			id := requireExecutionID(executionID, "restart")

			body, err := buildRestartExecutionBody(opts)
			if err != nil {
				agentError(
					"invalid_flag_value",
					err.Error(),
					"Pass valid JSON to --input (inline or @path) and check the restart flags.",
				)
			}

			proxyToServer(http.MethodPost, "/api/v1/executions/"+url.PathEscape(id)+"/restart", body)
		},
	}

	cmd.Flags().StringVar(&executionID, "id", "", "Execution ID")
	cmd.Flags().StringVar(&opts.scope, "scope", opts.scope, "Restart scope: workflow or execution")
	cmd.Flags().StringVar(&opts.reuse, "reuse", opts.reuse, "Replay reuse mode: succeeded-before, all-succeeded, or none")
	cmd.Flags().BoolVar(&opts.fork, "fork", false, "Mark this restart as a fork with intentional changes")
	cmd.Flags().StringVar(&opts.model, "model", "", "Model override to send in restart context")
	cmd.Flags().StringVar(&opts.input, "input", "", "JSON input override or @path to a JSON file")
	cmd.Flags().StringVar(&opts.reason, "reason", "", "Reason for restarting the execution")
	return cmd
}

func newAgentExecApprovalStatusCmd() *cobra.Command {
	var executionID string

	cmd := &cobra.Command{
		Use:   "approval-status",
		Short: "Get the approval status for an execution",
		Run: func(cmd *cobra.Command, args []string) {
			id := requireExecutionID(executionID, "approval-status")
			proxyToServer(http.MethodGet, "/api/v1/executions/"+url.PathEscape(id)+"/approval-status", nil)
		},
	}

	cmd.Flags().StringVar(&executionID, "id", "", "Execution ID")
	return cmd
}

func newAgentExecApproveCmd() *cobra.Command {
	var executionID string
	var decision string
	var reason string

	cmd := &cobra.Command{
		Use:   "approve",
		Short: "Resolve a pending approval for an execution",
		Run: func(cmd *cobra.Command, args []string) {
			id := requireExecutionID(executionID, "approve --decision approved")

			decision = strings.TrimSpace(decision)
			if decision == "" {
				agentError(
					"missing_required_flag",
					"--decision is required",
					"Set --decision to approved, rejected, or request_changes.",
				)
			}
			switch decision {
			case "approved", "rejected", "request_changes":
			default:
				agentError(
					"invalid_flag_value",
					fmt.Sprintf("invalid --decision %q", decision),
					"Set --decision to approved, rejected, or request_changes.",
				)
			}

			payload := map[string]string{"decision": decision}
			if v := strings.TrimSpace(reason); v != "" {
				payload["reason"] = v
			}

			proxyToServer(http.MethodPost, "/api/v1/executions/"+url.PathEscape(id)+"/approval-response", payload)
		},
	}

	cmd.Flags().StringVar(&executionID, "id", "", "Execution ID")
	cmd.Flags().StringVar(&decision, "decision", "", "Approval decision: approved, rejected, or request_changes")
	cmd.Flags().StringVar(&reason, "reason", "", "Optional feedback recorded with the decision")
	return cmd
}
