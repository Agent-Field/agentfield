package capabilities

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/sentry"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

type Runtime struct {
	Config config.Config
	Sentry *sentry.Client
}

func Register(agent *afagent.Agent, rt Runtime) {
	agent.RegisterReasoner("health", rt.health, afagent.WithDescription("Report Sentry node connectivity and configuration"))
	agent.RegisterReasoner("list_issues", rt.listIssues, afagent.WithDescription("List Sentry issues for a project"))
	agent.RegisterReasoner("get_issue", rt.getIssue, afagent.WithDescription("Read one Sentry issue by issue ID"))
	agent.RegisterReasoner("list_issue_events", rt.listIssueEvents, afagent.WithDescription("List events captured for a Sentry issue"))
	agent.RegisterReasoner("get_event", rt.getEvent, afagent.WithDescription("Read one event captured for a Sentry issue"))
	agent.RegisterReasoner("update_issue", rt.updateIssue, afagent.WithDescription("Update Sentry issue fields"))
	agent.RegisterReasoner("resolve_issue", rt.resolveIssue, afagent.WithDescription("Mark a Sentry issue resolved"))
	agent.RegisterReasoner("assign_issue", rt.assignIssue, afagent.WithDescription("Assign a Sentry issue to a user or team identifier"))
}

func (rt Runtime) health(ctx context.Context, _ map[string]any) (any, error) {
	out, err := rt.Sentry.Health(ctx)
	if err != nil {
		return nil, err
	}
	return map[string]any{"status": "ok", "node_id": rt.Config.NodeID, "sentry": out}, nil
}

func (rt Runtime) listIssues(ctx context.Context, input map[string]any) (any, error) {
	project, err := requiredString(input, "project")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.ListIssues(ctx, project, stringInput(input, "query"), intInput(input, "limit"))
}

func (rt Runtime) getIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := requiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.GetIssue(ctx, issueID)
}

func (rt Runtime) listIssueEvents(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := requiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.ListIssueEvents(ctx, issueID, stringInput(input, "query"), intInput(input, "limit"))
}

func (rt Runtime) getEvent(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := requiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	eventID, err := requiredString(input, "event_id")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.GetEvent(ctx, issueID, eventID)
}

func (rt Runtime) updateIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := requiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	fields, err := objectInput(input, "input")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.UpdateIssue(ctx, issueID, fields)
}

func (rt Runtime) resolveIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := requiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.ResolveIssue(ctx, issueID)
}

func (rt Runtime) assignIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := requiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	assignee, err := requiredString(input, "assignee")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.AssignIssue(ctx, issueID, assignee)
}

func requiredString(input map[string]any, key string) (string, error) {
	value := stringInput(input, key)
	if value == "" {
		return "", fmt.Errorf("%s is required", key)
	}
	return value, nil
}

func stringInput(input map[string]any, key string) string {
	if input == nil {
		return ""
	}
	value, ok := input[key].(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(value)
}

func intInput(input map[string]any, key string) int {
	if input == nil {
		return 0
	}
	switch value := input[key].(type) {
	case int:
		return value
	case float64:
		return int(value)
	case json.Number:
		n, _ := value.Int64()
		return int(n)
	default:
		return 0
	}
}

func objectInput(input map[string]any, key string) (map[string]any, error) {
	if input == nil {
		return nil, errors.New(key + " is required")
	}
	value, ok := input[key].(map[string]any)
	if !ok || len(value) == 0 {
		return nil, errors.New(key + " is required")
	}
	return value, nil
}
