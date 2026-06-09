package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"
)

// validJobID restricts job IDs to safe characters (prevents SSRF via path traversal).
var validJobID = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

const (
	defaultOpenRouterBaseURL    = "https://openrouter.ai/api/v1"
	defaultVideoPollInterval    = 30 * time.Second
	defaultVideoTimeout         = 10 * time.Minute
	defaultTTSSampleRate        = 24000
	defaultOpenRouterImageModel = "google/gemini-3.1-flash-image-preview"
	defaultOpenRouterTTSModel   = "hexgrad/kokoro-82m"
	openRouterModelMetaTTL      = 30 * time.Minute
)

// modelMeta holds the architecture metadata for an OpenRouter model.
type modelMeta struct {
	OutputModalities []string
	InputModalities  []string
}

// OpenRouterMediaProvider implements MediaProvider for OpenRouter's media APIs.
type OpenRouterMediaProvider struct {
	APIKey   string
	BaseURL  string
	Client   *http.Client
	SiteURL  string
	SiteName string

	metaMu    sync.Mutex
	metaCache map[string]modelMeta
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
	siteURL, siteName, _ := resolveOpenRouterAttribution("", "")
	return &OpenRouterMediaProvider{
		APIKey:   apiKey,
		BaseURL:  defaultOpenRouterBaseURL,
		Client:   &http.Client{Timeout: 60 * time.Second},
		SiteURL:  siteURL,
		SiteName: siteName,
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

func defaultVoiceForModel(model string) string {
	if stripPrefix(model) == defaultOpenRouterTTSModel {
		return "af_alloy"
	}
	return "alloy"
}

// SeedModelMeta lets callers (or tests) pre-populate the metadata cache for a
// model. Useful when running against test servers that don't expose
// `GET /models/{id}/endpoints`. Output modalities follow OpenRouter's
// convention — e.g. `[]string{"speech"}` for TTS-only or
// `[]string{"text","audio"}` for chat-audio models.
func (p *OpenRouterMediaProvider) SeedModelMeta(model string, outputModalities, inputModalities []string) {
	stripped := stripPrefix(model)
	p.metaMu.Lock()
	defer p.metaMu.Unlock()
	if p.metaCache == nil {
		p.metaCache = make(map[string]modelMeta)
	}
	p.metaCache[stripped] = modelMeta{
		OutputModalities: append([]string(nil), outputModalities...),
		InputModalities:  append([]string(nil), inputModalities...),
	}
}

// fetchModelMeta retrieves and caches a model's output_modalities so we can
// route audio/image requests to the right OpenRouter endpoint. Returns a
// zero-value meta on any failure so callers can fall back to defaults.
func (p *OpenRouterMediaProvider) fetchModelMeta(ctx context.Context, model string) modelMeta {
	stripped := stripPrefix(model)
	p.metaMu.Lock()
	if p.metaCache == nil {
		p.metaCache = make(map[string]modelMeta)
	}
	if cached, ok := p.metaCache[stripped]; ok {
		p.metaMu.Unlock()
		return cached
	}
	p.metaMu.Unlock()

	reqURL := p.baseURL() + "/models/" + stripped + "/endpoints"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return modelMeta{}
	}
	p.setHeaders(httpReq)
	resp, err := p.Client.Do(httpReq)
	if err != nil {
		return modelMeta{}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return modelMeta{}
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1*1024*1024))
	if err != nil {
		return modelMeta{}
	}
	var payload struct {
		Data struct {
			Architecture struct {
				OutputModalities []string `json:"output_modalities"`
				InputModalities  []string `json:"input_modalities"`
			} `json:"architecture"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return modelMeta{}
	}
	meta := modelMeta{
		OutputModalities: payload.Data.Architecture.OutputModalities,
		InputModalities:  payload.Data.Architecture.InputModalities,
	}
	p.metaMu.Lock()
	p.metaCache[stripped] = meta
	p.metaMu.Unlock()
	return meta
}

// wrapPCM16AsWAV wraps raw little-endian PCM16 mono bytes in a WAV (RIFF) container.
func wrapPCM16AsWAV(pcm []byte, sampleRate int) []byte {
	channels := uint16(1)
	bitsPerSample := uint16(16)
	byteRate := uint32(sampleRate) * uint32(channels) * uint32(bitsPerSample) / 8
	blockAlign := channels * bitsPerSample / 8
	dataSize := uint32(len(pcm))

	buf := bytes.NewBuffer(make([]byte, 0, 44+len(pcm)))
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16)) // PCM fmt chunk size
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))  // PCM format
	_ = binary.Write(buf, binary.LittleEndian, channels)
	_ = binary.Write(buf, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(buf, binary.LittleEndian, byteRate)
	_ = binary.Write(buf, binary.LittleEndian, blockAlign)
	_ = binary.Write(buf, binary.LittleEndian, bitsPerSample)
	buf.WriteString("data")
	_ = binary.Write(buf, binary.LittleEndian, dataSize)
	buf.Write(pcm)
	return buf.Bytes()
}

// containsString reports whether haystack contains s (case-sensitive).
func containsString(haystack []string, s string) bool {
	for _, h := range haystack {
		if h == s {
			return true
		}
	}
	return false
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
	if req.ImageURL != "" {
		payload["image_url"] = req.ImageURL
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
	ID           string   `json:"id"`
	Status       string   `json:"status"`
	Error        string   `json:"error,omitempty"`
	UnsignedURL  string   `json:"unsigned_url,omitempty"`  // legacy single-URL form
	UnsignedURLs []string `json:"unsigned_urls,omitempty"` // current API
	Duration     float64  `json:"duration,omitempty"`
	CostUSD      float64  `json:"cost_usd,omitempty"`
	Usage        struct {
		Cost float64 `json:"cost,omitempty"`
	} `json:"usage,omitempty"`
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

func (p *OpenRouterMediaProvider) buildVideoResponse(ctx context.Context, status *videoJobStatus) (*MediaResponse, error) {
	videoURL := ""
	if len(status.UnsignedURLs) > 0 {
		videoURL = status.UnsignedURLs[0]
	} else if status.UnsignedURL != "" {
		videoURL = status.UnsignedURL
	}

	cost := status.CostUSD
	if cost == 0 {
		cost = status.Usage.Cost
	}

	video := VideoData{
		URL:      videoURL,
		MimeType: "video/mp4",
		Filename: "generated_video.mp4",
		Duration: status.Duration,
		CostUSD:  cost,
	}

	// Download bytes when we have a URL. OpenRouter's "unsigned" URLs are
	// actually served from openrouter.ai itself and require the same Bearer
	// auth as the API; other hosts (CDNs) take the URL bare.
	if videoURL != "" {
		dlReq, err := http.NewRequestWithContext(ctx, http.MethodGet, videoURL, nil)
		if err == nil {
			if u, perr := url.Parse(videoURL); perr == nil {
				host := strings.ToLower(u.Hostname())
				if host == "openrouter.ai" || strings.HasSuffix(host, ".openrouter.ai") {
					dlReq.Header.Set("Authorization", "Bearer "+p.APIKey)
				}
			}
			dlResp, derr := p.Client.Do(dlReq)
			if derr == nil {
				defer dlResp.Body.Close()
				if dlResp.StatusCode == http.StatusOK {
					const maxVideoBytes = 500 * 1024 * 1024 // 500 MB
					raw, rerr := io.ReadAll(io.LimitReader(dlResp.Body, maxVideoBytes))
					if rerr == nil {
						video.Data = base64.StdEncoding.EncodeToString(raw)
					}
				}
			}
		}
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
		model = defaultOpenRouterImageModel
	}
	model = stripPrefix(model)

	// Request only image output — works for both image-only models (e.g.
	// x-ai/grok-imagine-image-quality) and dual-output models. Image-only
	// models return 404 when "text" is also requested.
	var userContent any = req.Prompt
	if len(req.ImageURLs) > 0 {
		parts := []map[string]any{{"type": "text", "text": req.Prompt}}
		for _, u := range req.ImageURLs {
			parts = append(parts, map[string]any{
				"type":      "image_url",
				"image_url": map[string]string{"url": u},
			})
		}
		userContent = parts
	}
	payload := map[string]any{
		"model": model,
		"messages": []map[string]any{
			{"role": "user", "content": userContent},
		},
		"modalities": []string{"image"},
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
	for k, v := range req.Extra {
		payload[k] = v
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

// GenerateAudio auto-routes to the right OpenRouter endpoint based on the
// model's output_modalities:
//   - ["speech"] (e.g. hexgrad/kokoro-82m)  → POST /audio/speech
//   - contains "audio" (e.g. openai/gpt-audio*) → chat-completions SSE
//   - unknown                                → POST /audio/speech (broader compat)
func (p *OpenRouterMediaProvider) GenerateAudio(ctx context.Context, req AudioRequest) (*MediaResponse, error) {
	if strings.TrimSpace(req.Text) == "" {
		return nil, fmt.Errorf("audio text input must not be empty")
	}

	model := req.Model
	if model == "" {
		model = defaultOpenRouterTTSModel
	}
	model = stripPrefix(model)
	voice := req.Voice
	if voice == "" {
		voice = defaultVoiceForModel(model)
	}

	requestedFormat := req.Format
	if requestedFormat == "" {
		requestedFormat = "wav"
	}

	meta := p.fetchModelMeta(ctx, model)
	useSpeech := len(meta.OutputModalities) == 0 ||
		containsString(meta.OutputModalities, "speech") ||
		!containsString(meta.OutputModalities, "audio")

	if useSpeech {
		return p.generateAudioViaSpeechEndpoint(ctx, model, req.Text, voice, requestedFormat, &req)
	}

	// Chat-completions audio modality (gpt-audio family). Streaming on OpenAI
	// is locked to pcm16 — wire that, then re-wrap to caller's format below.
	wireFormat := requestedFormat
	if requestedFormat == "wav" {
		wireFormat = "pcm16"
	}
	audioConfig := map[string]string{"format": wireFormat}
	audioConfig["voice"] = voice
	payload := map[string]any{
		"model": model,
		"messages": []map[string]any{
			{"role": "user", "content": req.Text},
		},
		"modalities": []string{"text", "audio"},
		"stream":     true,
		"audio":      audioConfig,
	}

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

	outputFormat := requestedFormat

	var audioData string
	if len(audioChunks) > 0 {
		// Decode all chunks, concatenate raw bytes, re-encode (with WAV
		// header when caller asked for wav).
		var raw []byte
		for _, chunk := range audioChunks {
			decoded, err := base64.StdEncoding.DecodeString(chunk)
			if err != nil {
				decoded, err = base64.RawStdEncoding.DecodeString(chunk)
				if err != nil {
					return nil, fmt.Errorf("decode audio chunk: %w (chunk length: %d)", err, len(chunk))
				}
			}
			raw = append(raw, decoded...)
		}
		if outputFormat == "wav" {
			raw = wrapPCM16AsWAV(raw, defaultTTSSampleRate)
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
	applyOpenRouterAttributionHeaders(req.Header, p.SiteURL, p.SiteName)
}

// generateAudioViaSpeechEndpoint calls POST /api/v1/audio/speech (OpenAI-compat
// TTS). Returns raw bytes for the caller's requested format; wraps PCM → WAV
// client-side when requestedFormat == "wav".
func (p *OpenRouterMediaProvider) generateAudioViaSpeechEndpoint(
	ctx context.Context, model, text, voice, requestedFormat string, req *AudioRequest,
) (*MediaResponse, error) {
	wireFormat := requestedFormat
	switch requestedFormat {
	case "wav", "pcm", "pcm16":
		wireFormat = "pcm"
	}

	if voice == "" {
		voice = "alloy"
	}

	payload := map[string]any{
		"model":           model,
		"input":           text,
		"voice":           voice,
		"response_format": wireFormat,
	}
	if req != nil && req.Speed != nil {
		payload["speed"] = *req.Speed
	}
	if req != nil {
		for k, v := range req.Extra {
			payload[k] = v
		}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal speech request: %w", err)
	}

	endpoint := p.baseURL() + "/audio/speech"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create speech request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.Client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("execute speech request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
		return nil, fmt.Errorf("audio/speech error (%d): %s", resp.StatusCode, string(errBody))
	}

	const maxAudioBytes = 100 * 1024 * 1024
	audioBytes, err := io.ReadAll(io.LimitReader(resp.Body, maxAudioBytes))
	if err != nil {
		return nil, fmt.Errorf("read speech body: %w", err)
	}

	if requestedFormat == "wav" {
		audioBytes = wrapPCM16AsWAV(audioBytes, defaultTTSSampleRate)
	}

	return &MediaResponse{
		Text: text,
		Audio: &AudioData{
			Data:   base64.StdEncoding.EncodeToString(audioBytes),
			Format: requestedFormat,
		},
		RawResponse: map[string]string{
			"endpoint":  "audio/speech",
			"model":     model,
			"mime_type": resp.Header.Get("Content-Type"),
		},
	}, nil
}
