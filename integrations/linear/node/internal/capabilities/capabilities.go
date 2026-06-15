package capabilities

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/linear"
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
	id, err := requiredString(input, "id")
	if err != nil {
		return nil, err
	}
	return rt.Linear.GetIssue(ctx, id)
}

func (rt Runtime) listIssues(ctx context.Context, input map[string]any) (any, error) {
	return rt.Linear.ListIssues(ctx, intInput(input, "limit"))
}

func (rt Runtime) createIssue(ctx context.Context, input map[string]any) (any, error) {
	fields, err := objectInput(input, "input")
	if err != nil {
		return nil, err
	}
	return rt.Linear.CreateIssue(ctx, fields)
}

func (rt Runtime) updateIssue(ctx context.Context, input map[string]any) (any, error) {
	id, err := requiredString(input, "id")
	if err != nil {
		return nil, err
	}
	fields, err := objectInput(input, "input")
	if err != nil {
		return nil, err
	}
	return rt.Linear.UpdateIssue(ctx, id, fields)
}

func (rt Runtime) commentIssue(ctx context.Context, input map[string]any) (any, error) {
	issueID, err := requiredString(input, "issue_id")
	if err != nil {
		return nil, err
	}
	body, err := requiredString(input, "body")
	if err != nil {
		return nil, err
	}
	return rt.Linear.CommentIssue(ctx, issueID, body)
}

func (rt Runtime) listTeams(ctx context.Context, input map[string]any) (any, error) {
	return rt.Linear.ListTeams(ctx, intInput(input, "limit"))
}

func (rt Runtime) listProjects(ctx context.Context, input map[string]any) (any, error) {
	return rt.Linear.ListProjects(ctx, intInput(input, "limit"))
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
