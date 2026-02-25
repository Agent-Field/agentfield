package client

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"time"
)

// RequestApprovalRequest is the payload for requesting human approval.
type RequestApprovalRequest struct {
	Title          string                 `json:"title"`
	Description    string                 `json:"description,omitempty"`
	TemplateType   string                 `json:"template_type"`
	Payload        map[string]interface{} `json:"payload,omitempty"`
	ProjectID      string                 `json:"project_id"`
	ExpiresInHours int                    `json:"expires_in_hours,omitempty"`
}

// RequestApprovalResponse is returned after creating an approval request.
type RequestApprovalResponse struct {
	ApprovalRequestID  string `json:"approval_request_id"`
	ApprovalRequestURL string `json:"approval_request_url"`
}

// ApprovalStatusResponse is returned by the approval status endpoint.
type ApprovalStatusResponse struct {
	Status      string                 `json:"status"` // pending, approved, rejected, expired
	Response    map[string]interface{} `json:"response,omitempty"`
	RequestURL  string                 `json:"request_url,omitempty"`
	RequestedAt string                 `json:"requested_at,omitempty"`
	RespondedAt string                 `json:"responded_at,omitempty"`
}

// WaitForApprovalOptions configures the blocking WaitForApproval helper.
type WaitForApprovalOptions struct {
	// PollInterval is the initial polling interval (default: 5s).
	PollInterval time.Duration
	// MaxInterval is the maximum polling interval (default: 60s).
	MaxInterval time.Duration
	// BackoffFactor is the multiplier applied to the interval each iteration (default: 2).
	BackoffFactor float64
}

func (o *WaitForApprovalOptions) defaults() {
	if o.PollInterval == 0 {
		o.PollInterval = 5 * time.Second
	}
	if o.MaxInterval == 0 {
		o.MaxInterval = 60 * time.Second
	}
	if o.BackoffFactor == 0 {
		o.BackoffFactor = 2.0
	}
}

// RequestApproval requests human approval for an execution, transitioning it
// to the "waiting" state on the control plane.
//
// Calls POST /api/v1/agents/{nodeID}/executions/{executionID}/request-approval.
func (c *Client) RequestApproval(ctx context.Context, nodeID, executionID string, req RequestApprovalRequest) (*RequestApprovalResponse, error) {
	route := fmt.Sprintf("/api/v1/agents/%s/executions/%s/request-approval",
		url.PathEscape(nodeID), url.PathEscape(executionID))

	var resp RequestApprovalResponse
	if err := c.do(ctx, http.MethodPost, route, req, &resp); err != nil {
		return nil, fmt.Errorf("request approval: %w", err)
	}
	return &resp, nil
}

// GetApprovalStatus returns the current approval status for an execution.
//
// Calls GET /api/v1/agents/{nodeID}/executions/{executionID}/approval-status.
func (c *Client) GetApprovalStatus(ctx context.Context, nodeID, executionID string) (*ApprovalStatusResponse, error) {
	route := fmt.Sprintf("/api/v1/agents/%s/executions/%s/approval-status",
		url.PathEscape(nodeID), url.PathEscape(executionID))

	var resp ApprovalStatusResponse
	if err := c.do(ctx, http.MethodGet, route, nil, &resp); err != nil {
		return nil, fmt.Errorf("get approval status: %w", err)
	}
	return &resp, nil
}

// WaitForApproval polls the approval status endpoint with exponential backoff
// until the status is no longer "pending" or the context is cancelled.
func (c *Client) WaitForApproval(ctx context.Context, nodeID, executionID string, opts *WaitForApprovalOptions) (*ApprovalStatusResponse, error) {
	o := WaitForApprovalOptions{}
	if opts != nil {
		o = *opts
	}
	o.defaults()

	interval := o.PollInterval

	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("wait for approval: %w", ctx.Err())
		case <-time.After(interval):
		}

		resp, err := c.GetApprovalStatus(ctx, nodeID, executionID)
		if err != nil {
			// Transient failure — back off and retry.
			interval = minDuration(
				time.Duration(float64(interval)*o.BackoffFactor),
				o.MaxInterval,
			)
			continue
		}

		if resp.Status != "pending" {
			return resp, nil
		}

		interval = minDuration(
			time.Duration(math.Round(float64(interval)*o.BackoffFactor)),
			o.MaxInterval,
		)
	}
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}
