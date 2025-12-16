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
	config     *Config
	httpClient *http.Client
}

// NewClient creates a new AI client with the given configuration.
func NewClient(config *Config) (*Client, error) {
	if config == nil {
		config = DefaultConfig()
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &Client{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}, nil
}

// Complete makes a chat completion request.
func (c *Client) Complete(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	// Build base request
	req := &Request{
		Messages: []Message{
			{Role: "user", Content: prompt},
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
	// Marshal request
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

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

	// Add OpenRouter-specific headers if applicable
	if c.config.IsOpenRouter() {
		if c.config.SiteURL != "" {
			httpReq.Header.Set("HTTP-Referer", c.config.SiteURL)
		}
		if c.config.SiteName != "" {
			httpReq.Header.Set("X-Title", c.config.SiteName)
		}
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
	if httpResp.StatusCode >= 400 {
		var errResp ErrorResponse
		if err := json.Unmarshal(respBody, &errResp); err != nil {
			return nil, fmt.Errorf("API error (%d): %s", httpResp.StatusCode, string(respBody))
		}
		return nil, fmt.Errorf("API error: %s", errResp.Error.Message)
	}

	// Parse response
	var response Response
	if err := json.Unmarshal(respBody, &response); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

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
				{Role: "user", Content: prompt},
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

		// Marshal request
		body, err := json.Marshal(req)
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

		// Add OpenRouter-specific headers if applicable
		if c.config.IsOpenRouter() {
			if c.config.SiteURL != "" {
				httpReq.Header.Set("HTTP-Referer", c.config.SiteURL)
			}
			if c.config.SiteName != "" {
				httpReq.Header.Set("X-Title", c.config.SiteName)
			}
		}

		// Execute request
		httpResp, err := c.httpClient.Do(httpReq)
		if err != nil {
			errCh <- fmt.Errorf("execute request: %w", err)
			return
		}
		defer httpResp.Body.Close()

		// Check for errors
		if httpResp.StatusCode >= 400 {
			respBody, _ := io.ReadAll(httpResp.Body)
			errCh <- fmt.Errorf("API error (%d): %s", httpResp.StatusCode, string(respBody))
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

				return chunk, nil
			}
			continue // Non-data message, try next
		}

		// Need more data
		n, err := d.reader.Read(d.buf)
		if err != nil {
			return StreamChunk{}, err
		}

		d.accumulated = append(d.accumulated, d.buf[:n]...)
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
