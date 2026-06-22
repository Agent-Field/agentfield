package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"
)

const defaultOpenAIEndpoint = "https://api.openai.com/v1/embeddings"

// OpenAIEmbedder calls OpenAI's embeddings API. It is pinned to a single model
// and dimension (see DefaultModel / Dimensions) so every chunk in the knowledge
// store is dimensionally compatible with the shared vector index.
type OpenAIEmbedder struct {
	apiKey   string
	model    string
	endpoint string
	dims     int
	client   *http.Client
}

// OpenAIOption customizes an OpenAIEmbedder.
type OpenAIOption func(*OpenAIEmbedder)

// WithModel overrides the embedding model. The dimension stays pinned at
// Dimensions; callers must ensure the model emits that dimension.
func WithModel(model string) OpenAIOption {
	return func(e *OpenAIEmbedder) {
		if model != "" {
			e.model = model
		}
	}
}

// WithEndpoint overrides the API endpoint (useful for proxies/tests).
func WithEndpoint(endpoint string) OpenAIOption {
	return func(e *OpenAIEmbedder) {
		if endpoint != "" {
			e.endpoint = endpoint
		}
	}
}

// WithHTTPClient overrides the HTTP client.
func WithHTTPClient(c *http.Client) OpenAIOption {
	return func(e *OpenAIEmbedder) {
		if c != nil {
			e.client = c
		}
	}
}

// NewOpenAIEmbedder builds an embedder using the given API key. The model
// defaults to DefaultModel and the dimension is pinned to Dimensions.
func NewOpenAIEmbedder(apiKey string, opts ...OpenAIOption) *OpenAIEmbedder {
	e := &OpenAIEmbedder{
		apiKey:   apiKey,
		model:    DefaultModel,
		endpoint: defaultOpenAIEndpoint,
		dims:     Dimensions,
		client:   &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// Dimensions returns the pinned embedding dimension.
func (e *OpenAIEmbedder) Dimensions() int { return e.dims }

type openAIEmbedRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type openAIEmbedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// Embed returns one vector per input string, in the same order as texts.
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	if e.apiKey == "" {
		return nil, fmt.Errorf("openai embedder: API key is not configured")
	}

	body, err := json.Marshal(openAIEmbedRequest{Input: texts, Model: e.model})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai embed request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return nil, fmt.Errorf("read embed response: %w", err)
	}

	var parsed openAIEmbedResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("decode embed response (status %d): %w", resp.StatusCode, err)
	}
	if resp.StatusCode != http.StatusOK {
		if parsed.Error != nil {
			return nil, fmt.Errorf("openai embed error (status %d): %s", resp.StatusCode, parsed.Error.Message)
		}
		return nil, fmt.Errorf("openai embed error: status %d", resp.StatusCode)
	}
	if len(parsed.Data) != len(texts) {
		return nil, fmt.Errorf("openai embed: expected %d vectors, got %d", len(texts), len(parsed.Data))
	}

	// The API documents results in request order, but it also returns an index
	// per item — sort defensively so output order always matches input order.
	sort.Slice(parsed.Data, func(i, j int) bool { return parsed.Data[i].Index < parsed.Data[j].Index })

	out := make([][]float32, len(parsed.Data))
	for i, d := range parsed.Data {
		if len(d.Embedding) != e.dims {
			return nil, fmt.Errorf("openai embed: vector %d has dimension %d, expected %d", i, len(d.Embedding), e.dims)
		}
		out[i] = d.Embedding
	}
	return out, nil
}
