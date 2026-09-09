package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Client provides AI/LLM capabilities using OpenAI or OpenRouter API.
type Client struct {
	config      *Config
	httpClient  *http.Client
	rateLimiter *RateLimiter
}

// ClientOption configures an AI client during construction.
type ClientOption func(*Client)

// WithHTTPClient uses the supplied HTTP client for AI requests.
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(c *Client) {
		if httpClient != nil {
			c.httpClient = httpClient
		}
	}
}

// WithTransport uses the supplied round tripper while preserving the default
// HTTP client's timeout and other settings.
func WithTransport(transport http.RoundTripper) ClientOption {
	return func(c *Client) {
		if transport != nil {
			c.httpClient.Transport = transport
		}
	}
}

// NewClient creates a new AI client with the given configuration.
func NewClient(config *Config, opts ...ClientOption) (*Client, error) {
	if config == nil {
		config = DefaultConfig()
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	client := &Client{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}

	if config.RateLimitEnabled() {
		client.rateLimiter = NewRateLimiter(config.rateLimiterConfig())
	}

	for _, opt := range opts {
		if opt != nil {
			opt(client)
		}
	}

	return client, nil
}

// Model returns the client's configured default model slug.
func (c *Client) Model() string {
	return c.config.Model
}

// Complete makes a chat completion request.
func (c *Client) Complete(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	// Build base request
	req := &Request{
		Messages: []Message{
			{
				Role: "user",
				Content: []ContentPart{
					{Type: "text", Text: prompt},
				},
			},
		},
		Model:       c.config.Model,
		Temperature: &c.config.Temperature,
		MaxTokens:   &c.config.MaxTokens,
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(req); err != nil {
			return nil, fmt.Errorf("apply option: %w", err)
		}
	}

	// Make HTTP request
	return c.doRequest(ctx, req)
}

// CompleteWithMessages makes a chat completion request with custom messages.
func (c *Client) CompleteWithMessages(ctx context.Context, messages []Message, opts ...Option) (*Response, error) {
	req := &Request{
		Messages:    messages,
		Model:       c.config.Model,
		Temperature: &c.config.Temperature,
		MaxTokens:   &c.config.MaxTokens,
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(req); err != nil {
			return nil, fmt.Errorf("apply option: %w", err)
		}
	}

	return c.doRequest(ctx, req)
}

func (c *Client) doRequest(ctx context.Context, req *Request) (*Response, error) {
	// Apply provider-specific parameter rewrites before serialization.
	body, err := c.marshalRequest(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	if c.rateLimiter == nil {
		return c.sendRequest(ctx, req, body)
	}

	var resp *Response
	err = c.rateLimiter.Execute(ctx, func() error {
		var callErr error
		resp, callErr = c.sendRequest(ctx, req, body)
		return callErr
	})
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// sendRequest performs a single (non-retried) chat completion HTTP request.
func (c *Client) sendRequest(ctx context.Context, req *Request, body []byte) (*Response, error) {
	// Build URL
	url := strings.TrimSuffix(c.config.BaseURL, "/") + "/chat/completions"

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	apiKey := c.config.APIKey
	if strings.TrimSpace(req.APIKeyOverride) != "" {
		apiKey = req.APIKeyOverride
	}
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	// Add gateway-specific attribution headers if applicable
	switch {
	case c.config.IsOpenRouter():
		applyOpenRouterAttributionHeaders(httpReq.Header, c.config.SiteURL, c.config.SiteName)
	case c.config.IsInfron():
		applyInfronAttributionHeaders(httpReq.Header, c.config.SiteURL, c.config.SiteName)
	}

	// Execute request
	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer httpResp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	// Check for errors
	if httpResp.StatusCode < http.StatusOK || httpResp.StatusCode >= http.StatusMultipleChoices {
		return nil, newAPIError(httpResp.StatusCode, respBody)
	}

	// Parse response
	var response Response
	if err := json.Unmarshal(respBody, &response); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}
	response.normalizeNativeCost()

	return &response, nil
}

// StreamComplete makes a streaming chat completion request.
// Returns a channel of response chunks.
func (c *Client) StreamComplete(ctx context.Context, prompt string, opts ...Option) (<-chan StreamChunk, <-chan error) {
	chunkCh := make(chan StreamChunk)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		// Build request with streaming enabled
		opts = append(opts, WithStream())
		req := &Request{
			Messages: []Message{
				{
					Role: "user",
					Content: []ContentPart{
						{Type: "text", Text: prompt},
					},
				},
			},
			Model:       c.config.Model,
			Temperature: &c.config.Temperature,
			MaxTokens:   &c.config.MaxTokens,
			Stream:      true,
		}

		// Apply options
		for _, opt := range opts {
			if err := opt(req); err != nil {
				errCh <- fmt.Errorf("apply option: %w", err)
				return
			}
		}

		// Apply provider-specific parameter rewrites before serialization.
		body, err := c.marshalRequest(req)
		if err != nil {
			errCh <- fmt.Errorf("marshal request: %w", err)
			return
		}

		// Build URL
		url := strings.TrimSuffix(c.config.BaseURL, "/") + "/chat/completions"

		// Create HTTP request
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			errCh <- fmt.Errorf("create request: %w", err)
			return
		}

		// Set headers
		httpReq.Header.Set("Content-Type", "application/json")
		apiKey := c.config.APIKey
		if strings.TrimSpace(req.APIKeyOverride) != "" {
			apiKey = req.APIKeyOverride
		}
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
		httpReq.Header.Set("Accept", "text/event-stream")

		// Add gateway-specific attribution headers if applicable
		switch {
		case c.config.IsOpenRouter():
			applyOpenRouterAttributionHeaders(httpReq.Header, c.config.SiteURL, c.config.SiteName)
		case c.config.IsInfron():
			applyInfronAttributionHeaders(httpReq.Header, c.config.SiteURL, c.config.SiteName)
		}

		// Execute request
		httpResp, err := c.httpClient.Do(httpReq)
		if err != nil {
			errCh <- fmt.Errorf("execute request: %w", err)
			return
		}
		defer httpResp.Body.Close()

		// Check for errors
		if httpResp.StatusCode < http.StatusOK || httpResp.StatusCode >= http.StatusMultipleChoices {
			respBody, _ := io.ReadAll(httpResp.Body)
			errCh <- newAPIError(httpResp.StatusCode, respBody)
			return
		}

		// Parse SSE stream
		decoder := NewSSEDecoder(httpResp.Body)
		for {
			chunk, err := decoder.Decode()
			if err != nil {
				if err != io.EOF {
					errCh <- fmt.Errorf("decode stream: %w", err)
				}
				return
			}

			select {
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			case chunkCh <- chunk:
			}
		}
	}()

	return chunkCh, errCh
}

// SSEDecoder decodes Server-Sent Events from a stream.
type SSEDecoder struct {
	reader      io.Reader
	accumulated []byte
	buf         []byte
}

// NewSSEDecoder creates a new SSE decoder.
func NewSSEDecoder(r io.Reader) *SSEDecoder {
	return &SSEDecoder{
		reader: r,
		buf:    make([]byte, 8192),
	}
}

// Decode reads and decodes the next SSE chunk.
func (d *SSEDecoder) Decode() (StreamChunk, error) {
	for {
		// First check if we already have a complete message in accumulated buffer
		data := string(d.accumulated)
		if idx := strings.Index(data, "\n\n"); idx >= 0 {
			message := data[:idx]
			d.accumulated = []byte(data[idx+2:])

			// Parse SSE message
			if strings.HasPrefix(message, "data: ") {
				jsonData := strings.TrimPrefix(message, "data: ")
				jsonData = strings.TrimSpace(jsonData)

				// Check for stream end
				if jsonData == "[DONE]" {
					return StreamChunk{}, io.EOF
				}

				var chunk StreamChunk
				if err := json.Unmarshal([]byte(jsonData), &chunk); err != nil {
					continue // Skip malformed chunks
				}
				chunk.normalizeNativeCost()

				return chunk, nil
			}
			continue // Non-data message, try next
		}

		// Need more data. Per the io.Reader contract, a Read may return n > 0
		// alongside a non-nil error (commonly the final body bytes together
		// with io.EOF) — those bytes must be consumed before the error is
		// surfaced, otherwise the stream's trailing messages (e.g. a terminal
		// usage-accounting chunk) are silently dropped.
		n, err := d.reader.Read(d.buf)
		if n > 0 {
			d.accumulated = append(d.accumulated, d.buf[:n]...)
			continue
		}
		if err != nil {
			return StreamChunk{}, err
		}
	}
}

// Convenience functions for common patterns

// SimpleAI makes a simple AI call with just a prompt.
func SimpleAI(ctx context.Context, prompt string) (string, error) {
	client, err := NewClient(DefaultConfig())
	if err != nil {
		return "", err
	}

	resp, err := client.Complete(ctx, prompt)
	if err != nil {
		return "", err
	}

	return resp.Text(), nil
}

// StructuredAI makes an AI call and returns structured data.
func StructuredAI(ctx context.Context, prompt string, schema interface{}, dest interface{}) error {
	client, err := NewClient(DefaultConfig())
	if err != nil {
		return err
	}

	resp, err := client.Complete(ctx, prompt, WithSchema(schema))
	if err != nil {
		return err
	}

	return resp.Into(dest)
}
