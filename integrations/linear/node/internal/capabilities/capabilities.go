package capabilities

import (
	"context"

	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/linear"
	"github.com/Agent-Field/agentfield/sdk/go/inputs"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

type Runtime struct {
	Config config.Config
	Linear *linear.Client
}

func Register(agent *afagent.Agent, rt Runtime) {
	agent.RegisterReasoner("health", rt.health, afagent.WithDescription("Report Linear node connectivity and configuration"))
	agent.RegisterReasoner("get_issue", rt.getIssue, afagent.WithDescription("Read one Linear issue by UUID or identifier"))
	agent.RegisterReasoner("list_issues", rt.listIssues, afagent.WithDescription("List recent Linear issues"))
	agent.RegisterReasoner("create_issue", rt.createIssue, afagent.WithDescription("Create a Linear issue with IssueCreateInput fields"))
	agent.RegisterReasoner("update_issue", rt.updateIssue, afagent.WithDescription("Update a Linear issue with IssueUpdateInput fields"))
	agent.RegisterReasoner("comment_issue", rt.commentIssue, afagent.WithDescription("Add a comment to a Linear issue"))
	agent.RegisterReasoner("list_teams", rt.listTeams, afagent.WithDescription("List Linear teams available to the token"))
	agent.RegisterReasoner("list_projects", rt.listProjects, afagent.WithDescription("List Linear projects available to the token"))
}

func (rt Runtime) health(ctx context.Context, _ map[string]any) (any, error) {
	out, err := rt.Linear.Health(ctx)
	if err != nil {
		return nil, err
	}
	return map[string]any{"status": "ok", "node_id": rt.Config.NodeID, "linear": out["data"]}, nil
}

func (rt Runtime) getIssue(ctx context.Context, input map[string]any) (any, error) {
	id, err := inputs.RequiredString(input, "id")
	if err != nil {
		return nil, err
	}
	return rt.Linear.GetIssue(ctx, id)
}

func (rt Runtime) listIssues(ctx context.Context, input map[string]any) (any, error) {
	return rt.Linear.ListIssues(ctx, inputs.Int(input, "limit"))
}

func (rt Runtime) createIssue(ctx context.Context, input map[string]any) (any, error) {
	fields, err := inputs.Object(input, "input")
	if err != nil {
		return nil, err
	}
	return rt.Linear.CreateIssue(ctx, fields)
}

func (rt Runtime) updateIssue(ctx context.Context, input map[string]any) (any, error) {
	id, err := inputs.RequiredString(input, "id")
	if err != nil {
		return nil, err
	}
	fields, err := inputs.Object(input, "input")
	if err != nil {
		return nil, err
	}
	return rt.Linear.UpdateIssue(ctx, id, fields)
}

func (rt Runtime) commentIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := inputs.RequiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	body, err := inputs.RequiredString(input, "body")
	if err != nil {
		return nil, err
	}
	return rt.Linear.CommentIssue(ctx, issueID, body)
}

func (rt Runtime) listTeams(ctx context.Context, input map[string]any) (any, error) {
	return rt.Linear.ListTeams(ctx, inputs.Int(input, "limit"))
}

func (rt Runtime) listProjects(ctx context.Context, input map[string]any) (any, error) {
	return rt.Linear.ListProjects(ctx, inputs.Int(input, "limit"))
}
