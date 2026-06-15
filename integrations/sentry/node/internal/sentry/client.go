package sentry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type Config struct {
	BaseURL        string
	Organization   string
	Token          string
	TimeoutSeconds int
	HTTPClient     *http.Client
}

type Client struct {
	cfg        Config
	httpClient *http.Client
}

func NewClient(cfg Config) (*Client, error) {
	cfg.BaseURL = strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://sentry.io"
	}
	cfg.Organization = strings.TrimSpace(cfg.Organization)
	if strings.TrimSpace(cfg.Token) == "" {
		return nil, errors.New("sentry auth token is required")
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

func (c *Client) Health(ctx context.Context) (any, error) {
	if c.cfg.Organization == "" {
		return map[string]any{"status": "configured", "organization_required_for": []string{"list_issues", "list_issue_events", "get_event"}}, nil
	}
	return c.get(ctx, fmt.Sprintf("/api/0/organizations/%s/projects/", url.PathEscape(c.cfg.Organization)), nil)
}

// TODO: surface the Link header for cursor-based pagination — current callers only get the first page.
func (c *Client) ListIssues(ctx context.Context, project string, query string, limit int) (any, error) {
	if c.cfg.Organization == "" {
		return nil, errors.New("organization is required")
	}
	if strings.TrimSpace(project) == "" {
		return nil, errors.New("project is required")
	}
	params := url.Values{}
	if strings.TrimSpace(query) != "" {
		params.Set("query", query)
	}
	if limit > 0 {
		params.Set("per_page", fmt.Sprintf("%d", limit))
	}
	// TODO: migrate to /api/0/organizations/{org}/issues/?project=<id> once we resolve slug→id
	return c.get(ctx, fmt.Sprintf("/api/0/projects/%s/%s/issues/", url.PathEscape(c.cfg.Organization), url.PathEscape(project)), params)
}

func (c *Client) GetIssue(ctx context.Context, issueID string) (any, error) {
	if c.cfg.Organization == "" {
		return nil, errors.New("organization is required")
	}
	if strings.TrimSpace(issueID) == "" {
		return nil, errors.New("issue_id is required")
	}
	return c.get(ctx, fmt.Sprintf("/api/0/organizations/%s/issues/%s/", url.PathEscape(c.cfg.Organization), url.PathEscape(issueID)), nil)
}

// TODO: surface the Link header for cursor-based pagination — current callers only get the first page.
func (c *Client) ListIssueEvents(ctx context.Context, issueID string, query string, limit int) (any, error) {
	if c.cfg.Organization == "" {
		return nil, errors.New("organization is required")
	}
	if strings.TrimSpace(issueID) == "" {
		return nil, errors.New("issue_id is required")
	}
	params := url.Values{}
	if strings.TrimSpace(query) != "" {
		params.Set("query", query)
	}
	if limit > 0 {
		params.Set("per_page", fmt.Sprintf("%d", limit))
	}
	return c.get(ctx, fmt.Sprintf("/api/0/organizations/%s/issues/%s/events/", url.PathEscape(c.cfg.Organization), url.PathEscape(issueID)), params)
}

func (c *Client) GetEvent(ctx context.Context, issueID, eventID string) (any, error) {
	if c.cfg.Organization == "" {
		return nil, errors.New("organization is required")
	}
	if strings.TrimSpace(issueID) == "" {
		return nil, errors.New("issue_id is required")
	}
	if strings.TrimSpace(eventID) == "" {
		return nil, errors.New("event_id is required")
	}
	return c.get(ctx, fmt.Sprintf("/api/0/organizations/%s/issues/%s/events/%s/", url.PathEscape(c.cfg.Organization), url.PathEscape(issueID), url.PathEscape(eventID)), nil)
}

func (c *Client) UpdateIssue(ctx context.Context, issueID string, fields map[string]any) (any, error) {
	if c.cfg.Organization == "" {
		return nil, errors.New("organization is required")
	}
	if strings.TrimSpace(issueID) == "" {
		return nil, errors.New("issue_id is required")
	}
	if len(fields) == 0 {
		return nil, errors.New("input is required")
	}
	return c.requestJSON(ctx, http.MethodPut, fmt.Sprintf("/api/0/organizations/%s/issues/%s/", url.PathEscape(c.cfg.Organization), url.PathEscape(issueID)), nil, fields)
}

func (c *Client) ResolveIssue(ctx context.Context, issueID string) (any, error) {
	return c.UpdateIssue(ctx, issueID, map[string]any{"status": "resolved"})
}

func (c *Client) AssignIssue(ctx context.Context, issueID, assignee string) (any, error) {
	if strings.TrimSpace(assignee) == "" {
		return nil, errors.New("assignee is required")
	}
	return c.UpdateIssue(ctx, issueID, map[string]any{"assignedTo": assignee})
}

func (c *Client) get(ctx context.Context, path string, params url.Values) (any, error) {
	return c.requestJSON(ctx, http.MethodGet, path, params, nil)
}

func (c *Client) requestJSON(ctx context.Context, method, path string, params url.Values, body any) (any, error) {
	var reader io.Reader
	if body != nil {
		payload, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(payload)
	}
	u := c.cfg.BaseURL + path
	if len(params) > 0 {
		u += "?" + params.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, method, u, reader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.cfg.Token)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("sentry API status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	if len(strings.TrimSpace(string(data))) == 0 {
		return map[string]any{"ok": true}, nil
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return out, nil
}
