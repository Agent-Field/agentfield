package capabilities

import (
	"context"

	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/sentry"
	"github.com/Agent-Field/agentfield/sdk/go/inputs"
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
	project, err := inputs.RequiredString(input, "project")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.ListIssues(ctx, project, inputs.String(input, "query"), inputs.Int(input, "limit"))
}

func (rt Runtime) getIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := inputs.RequiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.GetIssue(ctx, issueID)
}

func (rt Runtime) listIssueEvents(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := inputs.RequiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.ListIssueEvents(ctx, issueID, inputs.String(input, "query"), inputs.Int(input, "limit"))
}

func (rt Runtime) getEvent(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := inputs.RequiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	eventID, err := inputs.RequiredString(input, "event_id")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.GetEvent(ctx, issueID, eventID)
}

func (rt Runtime) updateIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := inputs.RequiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	fields, err := inputs.Object(input, "input")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.UpdateIssue(ctx, issueID, fields)
}

func (rt Runtime) resolveIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := inputs.RequiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.ResolveIssue(ctx, issueID)
}

func (rt Runtime) assignIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := inputs.RequiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	assignee, err := inputs.RequiredString(input, "assignee")
	if err != nil {
		return nil, err
	}
	return rt.Sentry.AssignIssue(ctx, issueID, assignee)
}
