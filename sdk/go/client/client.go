package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/types"
)

// Client provides a thin wrapper over the AgentField control plane REST API.
type Client struct {
	baseURL          *url.URL
	httpClient       *http.Client
	token            string
	apiKey           string
	didAuthenticator *DIDAuthenticator
}

// Option mutates Client configuration.
type Option func(*Client)

// WithHTTPClient allows custom HTTP transport configuration.
func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) {
		if h != nil {
			c.httpClient = h
		}
	}
}

// WithBearerToken sets the Authorization header for each request.
func WithBearerToken(token string) Option {
	return func(c *Client) {
		c.token = token
	}
}

// WithAPIKey sets the X-API-Key header for each request.
func WithAPIKey(key string) Option {
	return func(c *Client) {
		c.apiKey = key
	}
}

// WithDIDAuth configures DID authentication for agent-to-agent calls.
// The did parameter should be the agent's DID identifier (e.g., "did:web:example.com:agents:my-agent").
// The privateKeyJWK should be the JWK-formatted Ed25519 private key for signing.
func WithDIDAuth(did, privateKeyJWK string) Option {
	return func(c *Client) {
		auth, err := NewDIDAuthenticator(did, privateKeyJWK)
		if err != nil {
			log.Printf("WARNING: DID auth disabled due to JWK parse error: %v", err)
			return
		}
		c.didAuthenticator = auth
	}
}

// New creates a new Client instance.
func New(baseURL string, opts ...Option) (*Client, error) {
	if baseURL == "" {
		return nil, fmt.Errorf("baseURL is required")
	}

	parsed, err := url.Parse(strings.TrimSuffix(baseURL, "/"))
	if err != nil {
		return nil, fmt.Errorf("invalid baseURL: %w", err)
	}

	c := &Client{
		baseURL: parsed,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}

	for _, opt := range opts {
		opt(c)
	}

	return c, nil
}

// SignHTTPRequest applies DID authentication headers to an existing HTTP request.
// This is useful for code paths that construct their own requests (e.g., execute calls)
// rather than going through the client's do() method.
// If DID auth is not configured, this is a no-op.
func (c *Client) SignHTTPRequest(req *http.Request, body []byte) {
	if c == nil || c.didAuthenticator == nil || !c.didAuthenticator.IsConfigured() {
		return
	}
	for key, value := range c.didAuthenticator.SignRequest(body) {
		req.Header.Set(key, value)
	}
}

// RegisterNode registers or updates the agent node with the control plane.
func (c *Client) RegisterNode(ctx context.Context, payload types.NodeRegistrationRequest) (*types.NodeRegistrationResponse, error) {
	payload.LastHeartbeat = payload.LastHeartbeat.UTC()
	payload.RegisteredAt = payload.RegisteredAt.UTC()

	var resp types.NodeRegistrationResponse
	if err := c.do(ctx, http.MethodPost, "/api/v1/nodes", payload, &resp); err != nil {
		if apiErr, ok := err.(*APIError); ok && apiErr.StatusCode == http.StatusNotFound {
			// Fallback to legacy registration endpoint for older servers.
			if fallbackErr := c.do(ctx, http.MethodPost, "/api/v1/nodes/register", payload, &resp); fallbackErr != nil {
				return nil, fallbackErr
			}
			return &resp, nil
		}
		return nil, err
	}
	return &resp, nil
}

// GetNode retrieves node information from the control plane.
func (c *Client) GetNode(ctx context.Context, nodeID string) (map[string]interface{}, error) {
	var resp map[string]interface{}
	route := fmt.Sprintf("/api/v1/nodes/%s", url.PathEscape(nodeID))
	if err := c.do(ctx, http.MethodGet, route, nil, &resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// UpdateStatus renews the node lease and optionally reports lifecycle changes.
func (c *Client) UpdateStatus(ctx context.Context, nodeID string, payload types.NodeStatusUpdate) (*types.LeaseResponse, error) {
	var resp types.LeaseResponse
	route := fmt.Sprintf("/api/v1/nodes/%s/status", url.PathEscape(nodeID))
	if err := c.do(ctx, http.MethodPatch, route, payload, &resp); err != nil {
		if apiErr, ok := err.(*APIError); ok && apiErr.StatusCode == http.StatusNotFound {
			return c.legacyHeartbeat(ctx, nodeID, payload)
		}
		return nil, err
	}
	return &resp, nil
}

// AcknowledgeAction notifies the control plane that a pushed action completed.
func (c *Client) AcknowledgeAction(ctx context.Context, nodeID string, payload types.ActionAckRequest) (*types.LeaseResponse, error) {
	var resp types.LeaseResponse
	route := fmt.Sprintf("/api/v1/nodes/%s/actions/ack", url.PathEscape(nodeID))
	if err := c.do(ctx, http.MethodPost, route, payload, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Shutdown informs the control plane that the node is shutting down gracefully.
func (c *Client) Shutdown(ctx context.Context, nodeID string, payload types.ShutdownRequest) (*types.LeaseResponse, error) {
	var resp types.LeaseResponse
	route := fmt.Sprintf("/api/v1/nodes/%s/shutdown", url.PathEscape(nodeID))
	if err := c.do(ctx, http.MethodPost, route, payload, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// PostExecutionLogs sends one structured execution log entry or a batch payload
// to the control plane execution-log ingestion API.
func (c *Client) PostExecutionLogs(ctx context.Context, executionID string, payload any) error {
	if strings.TrimSpace(executionID) == "" {
		return fmt.Errorf("executionID is required")
	}
	route := fmt.Sprintf("/api/v1/executions/%s/logs", url.PathEscape(executionID))
	return c.do(ctx, http.MethodPost, route, payload, nil)
}

func (c *Client) do(ctx context.Context, method string, endpoint string, body any, out any) error {
	u := *c.baseURL
	rel := strings.TrimPrefix(endpoint, "/")
	basePath := strings.TrimSuffix(c.baseURL.Path, "/")
	if basePath == "" {
		u.Path = "/" + rel
	} else {
		u.Path = path.Join(basePath, rel)
		if !strings.HasPrefix(u.Path, "/") {
			u.Path = "/" + u.Path
		}
	}

	var bodyBytes []byte
	var buf io.ReadWriter = &bytes.Buffer{}
	if body != nil {
		var err error
		bodyBytes, err = json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode request: %w", err)
		}
		buf = bytes.NewBuffer(bodyBytes)
	}

	req, err := http.NewRequestWithContext(ctx, method, u.String(), buf)
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}

	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json")

	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	if c.apiKey != "" {
		req.Header.Set("X-API-Key", c.apiKey)
	}

	// Add DID authentication headers if configured
	if c.didAuthenticator != nil && c.didAuthenticator.IsConfigured() {
		didHeaders := c.didAuthenticator.SignRequest(bodyBytes)
		for key, value := range didHeaders {
			req.Header.Set(key, value)
		}
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("perform request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return &APIError{
			StatusCode: resp.StatusCode,
			Body:       respBody,
		}
	}

	if out == nil || len(respBody) == 0 {
		return nil
	}

	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}

	return nil
}

func (c *Client) legacyHeartbeat(ctx context.Context, nodeID string, payload types.NodeStatusUpdate) (*types.LeaseResponse, error) {
	route := fmt.Sprintf("/api/v1/nodes/%s/heartbeat", url.PathEscape(nodeID))
	if err := c.do(ctx, http.MethodPost, route, payload, nil); err != nil {
		return nil, err
	}
	lease := 120 * time.Second
	return &types.LeaseResponse{
		LeaseSeconds:     int(lease.Seconds()),
		NextLeaseRenewal: time.Now().Add(lease).UTC().Format(time.RFC3339),
	}, nil
}

// SetDIDCredentials configures DID authentication credentials after client creation.
// Returns an error if the credentials are invalid.
func (c *Client) SetDIDCredentials(did, privateKeyJWK string) error {
	auth, err := NewDIDAuthenticator(did, privateKeyJWK)
	if err != nil {
		return err
	}
	c.didAuthenticator = auth
	return nil
}

// DIDAuthConfigured returns true if DID authentication is configured.
func (c *Client) DIDAuthConfigured() bool {
	return c.didAuthenticator != nil && c.didAuthenticator.IsConfigured()
}

// DID returns the configured DID identifier, or empty string if not configured.
func (c *Client) DID() string {
	if c.didAuthenticator == nil {
		return ""
	}
	return c.didAuthenticator.DID()
}

// SignBody returns DID authentication headers for the given request body.
// Returns nil if DID auth is not configured. This is used by the DID client
// to sign VC generation and other authenticated requests.
func (c *Client) SignBody(body []byte) map[string]string {
	if c == nil || c.didAuthenticator == nil || !c.didAuthenticator.IsConfigured() {
		return nil
	}
	return c.didAuthenticator.SignRequest(body)
}

// APIError captures non-success responses from the AgentField API.
type APIError struct {
	StatusCode int
	Body       []byte
}

func (e *APIError) Error() string {
	return fmt.Sprintf("agentfield api error (%d): %s", e.StatusCode, strings.TrimSpace(string(e.Body)))
}
