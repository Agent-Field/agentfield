package did

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// SignRequestFunc returns DID authentication headers for a request body.
// It is set after DID registration, once the agent has credentials.
type SignRequestFunc func(body []byte) map[string]string

// Client handles HTTP communication with the control plane's DID and VC endpoints.
type Client struct {
	baseURL    string
	httpClient *http.Client
	token      string
	signFn     SignRequestFunc
}

// ClientOption configures a Client.
type ClientOption func(*Client)

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(c *http.Client) ClientOption {
	return func(dc *Client) {
		if c != nil {
			dc.httpClient = c
		}
	}
}

// WithToken sets a bearer token for authenticated requests.
func WithToken(token string) ClientOption {
	return func(dc *Client) {
		dc.token = token
	}
}

// NewClient creates a DID client for the given control plane URL.
func NewClient(baseURL string, opts ...ClientOption) *Client {
	c := &Client{
		baseURL: strings.TrimSuffix(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// SetSignFunc configures DID request signing. Call this after DID registration
// once the agent has valid credentials.
func (c *Client) SetSignFunc(fn SignRequestFunc) {
	c.signFn = fn
}

// RegisterAgent registers the agent with the control plane's DID service
// and returns the identity package containing all generated DIDs and keys.
func (c *Client) RegisterAgent(ctx context.Context, req RegistrationRequest) (*RegistrationResponse, error) {
	var resp RegistrationResponse
	if err := c.do(ctx, http.MethodPost, "/api/v1/did/register", req, &resp); err != nil {
		return nil, fmt.Errorf("DID registration failed: %w", err)
	}
	if !resp.Success {
		msg := resp.Error
		if msg == "" {
			msg = "server returned success=false"
		}
		return nil, fmt.Errorf("DID registration failed: %s", msg)
	}
	return &resp, nil
}

// GenerateExecutionVC requests the control plane to generate a Verifiable
// Credential for a completed execution.
func (c *Client) GenerateExecutionVC(ctx context.Context, req VCGenerationRequest) (*ExecutionVC, error) {
	var resp ExecutionVC
	if err := c.do(ctx, http.MethodPost, "/api/v1/execution/vc", req, &resp); err != nil {
		return nil, fmt.Errorf("VC generation failed: %w", err)
	}
	return &resp, nil
}

// ExportWorkflowVCChain retrieves the complete VC chain for a workflow,
// suitable for offline verification and auditing.
func (c *Client) ExportWorkflowVCChain(ctx context.Context, workflowID string) (*WorkflowVCChain, error) {
	endpoint := fmt.Sprintf("/api/v1/did/workflow/%s/vc-chain", url.PathEscape(workflowID))
	var resp WorkflowVCChain
	if err := c.do(ctx, http.MethodGet, endpoint, nil, &resp); err != nil {
		return nil, fmt.Errorf("export VC chain failed: %w", err)
	}
	return &resp, nil
}

func (c *Client) do(ctx context.Context, method, endpoint string, body any, out any) error {
	fullURL := c.baseURL + endpoint

	var bodyBytes []byte
	var bodyReader io.Reader
	if body != nil {
		var err error
		bodyBytes, err = json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode request: %w", err)
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}

	req, err := http.NewRequestWithContext(ctx, method, fullURL, bodyReader)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json")

	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	// Apply DID authentication if configured.
	if c.signFn != nil && bodyBytes != nil {
		for k, v := range c.signFn(bodyBytes) {
			req.Header.Set(k, v)
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
		return fmt.Errorf("server returned %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	if out != nil && len(respBody) > 0 {
		if err := json.Unmarshal(respBody, out); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}

	return nil
}
