package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

// validJobID restricts job IDs to safe characters (prevents SSRF via path traversal).
var validJobID = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

const (
	defaultOpenRouterBaseURL  = "https://openrouter.ai/api/v1"
	defaultVideoPollInterval  = 30 * time.Second
	defaultVideoTimeout       = 10 * time.Minute
)

// OpenRouterMediaProvider implements MediaProvider for OpenRouter's media APIs.
type OpenRouterMediaProvider struct {
	APIKey  string
	BaseURL string
	Client  *http.Client
}

// NewOpenRouterMediaProvider creates a provider. If apiKey is empty, reads OPENROUTER_API_KEY.
// Returns error if no API key is available.
func NewOpenRouterMediaProvider(apiKey string) (*OpenRouterMediaProvider, error) {
	if apiKey == "" {
		apiKey = os.Getenv("OPENROUTER_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("OpenRouter API key required: pass apiKey or set OPENROUTER_API_KEY")
	}
	return &OpenRouterMediaProvider{
		APIKey:  apiKey,
		BaseURL: defaultOpenRouterBaseURL,
		Client:  &http.Client{Timeout: 60 * time.Second},
	}, nil
}

func (p *OpenRouterMediaProvider) Name() string {
	return "openrouter"
}

func (p *OpenRouterMediaProvider) SupportedModalities() []string {
	return []string{"image", "audio", "video"}
}

func (p *OpenRouterMediaProvider) baseURL() string {
	if p.BaseURL != "" {
		return strings.TrimSuffix(p.BaseURL, "/")
	}
	return defaultOpenRouterBaseURL
}

// stripPrefix removes the "openrouter/" prefix from model names.
func stripPrefix(model string) string {
	return strings.TrimPrefix(model, "openrouter/")
}

// GenerateVideo submits a video job, polls until complete, downloads result.
func (p *OpenRouterMediaProvider) GenerateVideo(ctx context.Context, req VideoRequest) (*MediaResponse, error) {
	if strings.TrimSpace(req.Prompt) == "" {
		return nil, fmt.Errorf("video prompt must not be empty")
	}

	pollInterval := req.PollInterval
	if pollInterval == 0 {
		pollInterval = defaultVideoPollInterval
	}
	timeout := req.Timeout
	if timeout == 0 {
		timeout = defaultVideoTimeout
	}

	// Build submit payload
	payload := map[string]any{
		"model":  stripPrefix(req.Model),
		"prompt": req.Prompt,
	}
	if req.Duration > 0 {
		payload["duration"] = req.Duration
	}
	if req.Resolution != "" {
		payload["resolution"] = req.Resolution
	}
	if req.AspectRatio != "" {
		payload["aspect_ratio"] = req.AspectRatio
	}
	if req.GenerateAudio != nil {
		payload["generate_audio"] = *req.GenerateAudio
	}
	if req.Seed != nil {
		payload["seed"] = *req.Seed
	}
	if len(req.FrameImages) > 0 {
		payload["frame_images"] = req.FrameImages
	}
	if len(req.InputReferences) > 0 {
		payload["input_references"] = req.InputReferences
	}
	for k, v := range req.Extra {
		payload[k] = v
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal video request: %w", err)
	}

	// Submit job
	submitURL := p.baseURL() + "/videos"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, submitURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create submit request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.Client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("submit video job: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("read submit response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("video submit error (%d): %s", resp.StatusCode, string(respBody))
	}

	var submitResp struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(respBody, &submitResp); err != nil {
		return nil, fmt.Errorf("parse submit response: %w", err)
	}
	if submitResp.ID == "" {
		return nil, fmt.Errorf("no job ID in submit response: %s", string(respBody))
	}

	// Validate job ID to prevent SSRF via path traversal
	if !validJobID.MatchString(submitResp.ID) {
		return nil, fmt.Errorf("invalid job ID in submit response: %q", submitResp.ID)
	}

	// Derive a context with the video-specific timeout, but respect caller's deadline
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	pollURL := p.baseURL() + "/videos/" + submitResp.ID

	// Poll loop using context for deadline enforcement
	const maxTransientErrors = 3
	transientErrors := 0

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("video generation: %w", ctx.Err())
		case <-ticker.C:
		}

		status, err := p.pollVideoJob(ctx, pollURL)
		if err != nil {
			transientErrors++
			if transientErrors >= maxTransientErrors {
				return nil, fmt.Errorf("video poll failed after %d retries: %w", transientErrors, err)
			}
			continue // retry on next tick
		}
		transientErrors = 0

		switch status.Status {
		case "completed":
			return p.buildVideoResponse(ctx, status)
		case "failed":
			return nil, fmt.Errorf("video generation failed: %s", status.Error)
		}
		// pending/processing — continue polling
	}
}

type videoJobStatus struct {
	ID          string  `json:"id"`
	Status      string  `json:"status"`
	Error       string  `json:"error,omitempty"`
	UnsignedURL string  `json:"unsigned_url,omitempty"`
	Duration    float64 `json:"duration,omitempty"`
	CostUSD     float64 `json:"cost_usd,omitempty"`
}

func (p *OpenRouterMediaProvider) pollVideoJob(ctx context.Context, url string) (*videoJobStatus, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create poll request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.Client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("poll video job: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("read poll response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("poll error (%d): %s", resp.StatusCode, string(body))
	}

	var status videoJobStatus
	if err := json.Unmarshal(body, &status); err != nil {
		return nil, fmt.Errorf("parse poll response: %w", err)
	}
	return &status, nil
}

func (p *OpenRouterMediaProvider) buildVideoResponse(_ context.Context, status *videoJobStatus) (*MediaResponse, error) {
	video := VideoData{
		URL:      status.UnsignedURL,
		MimeType: "video/mp4",
		Filename: "generated_video.mp4",
		Duration: status.Duration,
		CostUSD:  status.CostUSD,
	}

	return &MediaResponse{
		Videos:      []VideoData{video},
		RawResponse: status,
	}, nil
}

// GenerateImage uses chat completions with image modality.
func (p *OpenRouterMediaProvider) GenerateImage(ctx context.Context, req ImageRequest) (*MediaResponse, error) {
	model := req.Model
	if model == "" {
		model = "openai/gpt-image-1"
	}
	model = stripPrefix(model)

	payload := map[string]any{
		"model": model,
		"messages": []map[string]any{
			{"role": "user", "content": req.Prompt},
		},
		"modalities": []string{"image", "text"},
	}
	if req.Size != "" {
		payload["size"] = req.Size
	}
	if req.Quality != "" {
		payload["quality"] = req.Quality
	}
	if req.ImageConfig != nil {
		payload["image_config"] = req.ImageConfig
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal image request: %w", err)
	}

	url := p.baseURL() + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create image request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.Client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("execute image request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("read image response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("image generation error (%d): %s", resp.StatusCode, string(respBody))
	}

	// Content can be null, a string, or an array of content parts depending on the model.
	// Some models (Gemini) return images in message.images[] instead of content.
	var chatResp struct {
		Choices []struct {
			Message struct {
				Content json.RawMessage `json:"content"`
				Images  []struct {
					Type     string `json:"type"`
					ImageURL struct {
						URL string `json:"url"`
					} `json:"image_url"`
				} `json:"images,omitempty"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("parse image response: %w", err)
	}

	type contentPart struct {
		Type    string `json:"type"`
		Text    string `json:"text,omitempty"`
		B64JSON string `json:"b64_json,omitempty"`
	}

	result := &MediaResponse{RawResponse: json.RawMessage(respBody)}
	for _, choice := range chatResp.Choices {
		raw := choice.Message.Content

		// Parse content field (can be null, string, or array of parts)
		if len(raw) > 0 && string(raw) != "null" {
			// Try array of content parts first
			var parts []contentPart
			if err := json.Unmarshal(raw, &parts); err == nil {
				for _, part := range parts {
					switch part.Type {
					case "text":
						result.Text = part.Text
					case "image_url", "image":
						result.Images = append(result.Images, ImageData{
							B64JSON: part.B64JSON,
						})
					}
				}
			} else {
				// Fall back to plain string (some models return content as a string with inline base64)
				var textContent string
				if err := json.Unmarshal(raw, &textContent); err == nil {
					result.Text = textContent
					if idx := strings.Index(textContent, "data:image/"); idx >= 0 {
						if b64Start := strings.Index(textContent[idx:], "base64,"); b64Start >= 0 {
							b64Data := textContent[idx+b64Start+7:]
							if end := strings.IndexAny(b64Data, ")\n\r\t "); end >= 0 {
								b64Data = b64Data[:end]
							}
							result.Images = append(result.Images, ImageData{B64JSON: b64Data})
						}
					}
				}
			}
		}

		// Handle images returned in message.images[] (Gemini-style: content=null, images=[...])
		for _, img := range choice.Message.Images {
			imgData := ImageData{}
			url := img.ImageURL.URL
			if strings.HasPrefix(url, "data:image/") {
				if b64Start := strings.Index(url, "base64,"); b64Start >= 0 {
					imgData.B64JSON = url[b64Start+7:]
				}
			} else if url != "" {
				imgData.URL = url
			}
			if imgData.B64JSON != "" || imgData.URL != "" {
				result.Images = append(result.Images, imgData)
			}
		}
	}

	return result, nil
}

// GenerateAudio uses streaming chat completions with audio modality.
func (p *OpenRouterMediaProvider) GenerateAudio(ctx context.Context, req AudioRequest) (*MediaResponse, error) {
	if strings.TrimSpace(req.Text) == "" {
		return nil, fmt.Errorf("audio text input must not be empty")
	}

	model := req.Model
	if model == "" {
		model = "openai/gpt-4o-audio-preview"
	}
	model = stripPrefix(model)

	payload := map[string]any{
		"model": model,
		"messages": []map[string]any{
			{"role": "user", "content": req.Text},
		},
		"modalities": []string{"text", "audio"},
		"stream":     true,
	}

	// When streaming, OpenAI only supports pcm16 format; use pcm16 as default.
	audioFormat := "pcm16"
	if req.Format != "" {
		audioFormat = req.Format
	}
	audioConfig := map[string]string{"format": audioFormat}
	if req.Voice != "" {
		audioConfig["voice"] = req.Voice
	} else {
		audioConfig["voice"] = "alloy"
	}
	payload["audio"] = audioConfig

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal audio request: %w", err)
	}

	url := p.baseURL() + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create audio request: %w", err)
	}
	p.setHeaders(httpReq)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.Client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("execute audio request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
		return nil, fmt.Errorf("audio generation error (%d): %s", resp.StatusCode, string(respBody))
	}

	// Parse SSE stream, collect audio chunks
	var audioChunks []string
	var textParts []string

	scanner := bufio.NewScanner(resp.Body)
	// SSE audio chunks can be large base64; set 1MB max line size
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content,omitempty"`
					Audio   *struct {
						Data string `json:"data,omitempty"`
					} `json:"audio,omitempty"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				textParts = append(textParts, choice.Delta.Content)
			}
			if choice.Delta.Audio != nil && choice.Delta.Audio.Data != "" {
				audioChunks = append(audioChunks, choice.Delta.Audio.Data)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read audio stream: %w", err)
	}

	// Concatenate base64 audio chunks
	outputFormat := "pcm16"
	if req.Format != "" {
		outputFormat = req.Format
	}

	var audioData string
	if len(audioChunks) > 0 {
		// Decode all chunks, concatenate raw bytes, re-encode
		var raw []byte
		for _, chunk := range audioChunks {
			decoded, err := base64.StdEncoding.DecodeString(chunk)
			if err != nil {
				// Try without padding
				decoded, err = base64.RawStdEncoding.DecodeString(chunk)
				if err != nil {
					return nil, fmt.Errorf("decode audio chunk: %w (chunk length: %d)", err, len(chunk))
				}
			}
			raw = append(raw, decoded...)
		}
		audioData = base64.StdEncoding.EncodeToString(raw)
	}

	return &MediaResponse{
		Text: strings.Join(textParts, ""),
		Audio: &AudioData{
			Data:   audioData,
			Format: outputFormat,
		},
	}, nil
}

func (p *OpenRouterMediaProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)
}
