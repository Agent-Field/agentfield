package linear

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Config struct {
	APIURL         string
	Token          string
	TimeoutSeconds int
	HTTPClient     *http.Client
}

type Client struct {
	cfg        Config
	httpClient *http.Client
}

func NewClient(cfg Config) (*Client, error) {
	cfg.APIURL = strings.TrimSpace(cfg.APIURL)
	if cfg.APIURL == "" {
		cfg.APIURL = "https://api.linear.app/graphql"
	}
	if strings.TrimSpace(cfg.Token) == "" {
		return nil, errors.New("linear API token is required")
	}
	if cfg.TimeoutSeconds <= 0 {
		cfg.TimeoutSeconds = 20
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: time.Duration(cfg.TimeoutSeconds) * time.Second}
	}
	return &Client{cfg: cfg, httpClient: httpClient}, nil
}

func (c *Client) Health(ctx context.Context) (map[string]any, error) {
	return c.graphql(ctx, `query AgentFieldLinearHealth { viewer { id name email } }`, nil)
}

func (c *Client) GetIssue(ctx context.Context, id string) (map[string]any, error) {
	if strings.TrimSpace(id) == "" {
		return nil, errors.New("id is required")
	}
	return c.graphql(ctx, `query AgentFieldGetIssue($id: String!) {
		issue(id: $id) {
			id identifier title description url priority estimate createdAt updatedAt
			state { id name type }
			team { id key name }
			assignee { id name email }
			project { id name }
			cycle { id name number }
		}
	}`, map[string]any{"id": id})
}

func (c *Client) ListIssues(ctx context.Context, first int) (map[string]any, error) {
	if first <= 0 || first > 100 {
		first = 25
	}
	return c.graphql(ctx, `query AgentFieldListIssues($first: Int!) {
		issues(first: $first) {
			nodes {
				id identifier title url priority createdAt updatedAt
				state { id name type }
				team { id key name }
				assignee { id name email }
			}
		}
	}`, map[string]any{"first": first})
}

func (c *Client) CreateIssue(ctx context.Context, input map[string]any) (map[string]any, error) {
	if len(input) == 0 {
		return nil, errors.New("input is required")
	}
	return c.graphql(ctx, `mutation AgentFieldCreateIssue($input: IssueCreateInput!) {
		issueCreate(input: $input) {
			success
			issue { id identifier title url createdAt updatedAt state { id name type } }
		}
	}`, map[string]any{"input": input})
}

func (c *Client) UpdateIssue(ctx context.Context, id string, input map[string]any) (map[string]any, error) {
	if strings.TrimSpace(id) == "" {
		return nil, errors.New("id is required")
	}
	if len(input) == 0 {
		return nil, errors.New("input is required")
	}
	return c.graphql(ctx, `mutation AgentFieldUpdateIssue($id: String!, $input: IssueUpdateInput!) {
		issueUpdate(id: $id, input: $input) {
			success
			issue { id identifier title url createdAt updatedAt state { id name type } }
		}
	}`, map[string]any{"id": id, "input": input})
}

func (c *Client) CommentIssue(ctx context.Context, issueID, body string) (map[string]any, error) {
	if strings.TrimSpace(issueID) == "" {
		return nil, errors.New("issue_id is required")
	}
	if strings.TrimSpace(body) == "" {
		return nil, errors.New("body is required")
	}
	return c.graphql(ctx, `mutation AgentFieldCommentIssue($input: CommentCreateInput!) {
		commentCreate(input: $input) {
			success
			comment { id body url createdAt issue { id identifier } }
		}
	}`, map[string]any{"input": map[string]any{"issueId": issueID, "body": body}})
}

func (c *Client) ListTeams(ctx context.Context, first int) (map[string]any, error) {
	if first <= 0 || first > 100 {
		first = 50
	}
	return c.graphql(ctx, `query AgentFieldListTeams($first: Int!) {
		teams(first: $first) {
			nodes { id key name description }
		}
	}`, map[string]any{"first": first})
}

func (c *Client) ListProjects(ctx context.Context, first int) (map[string]any, error) {
	if first <= 0 || first > 100 {
		first = 50
	}
	return c.graphql(ctx, `query AgentFieldListProjects($first: Int!) {
		projects(first: $first) {
			nodes { id name description state url createdAt updatedAt }
		}
	}`, map[string]any{"first": first})
}

func (c *Client) graphql(ctx context.Context, query string, variables map[string]any) (map[string]any, error) {
	payload, err := json.Marshal(map[string]any{"query": query, "variables": variables})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.APIURL, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", authHeader(c.cfg.Token))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("linear GraphQL status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	if rawErrors, ok := out["errors"]; ok {
		return nil, fmt.Errorf("linear GraphQL errors: %v", rawErrors)
	}
	return out, nil
}

// authHeader returns the Authorization header value for the given token.
// OAuth tokens (starting with "lin_oauth_") are prefixed with "Bearer ",
// while personal API keys are returned raw.
func authHeader(token string) string {
	if strings.HasPrefix(token, "lin_oauth_") {
		return "Bearer " + token
	}
	return token
}
