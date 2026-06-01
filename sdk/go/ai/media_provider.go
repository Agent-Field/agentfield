package ai

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"
)

// VideoRequest holds parameters for video generation.
type VideoRequest struct {
	Prompt          string           `json:"prompt"`
	Model           string           `json:"model"`
	Duration        int              `json:"duration,omitempty"`
	Resolution      string           `json:"resolution,omitempty"`
	AspectRatio     string           `json:"aspect_ratio,omitempty"`
	GenerateAudio   *bool            `json:"generate_audio,omitempty"`
	Seed            *int             `json:"seed,omitempty"`
	// ImageURL is a single input image for image-to-video models (convenience
	// alternative to FrameImages with frame_type=first_frame).
	ImageURL        string           `json:"image_url,omitempty"`
	// FrameImages is per-frame guidance — first_frame / last_frame. Items
	// follow OpenRouter's shape: {type, image_url:{url}, frame_type}.
	FrameImages     []map[string]any `json:"frame_images,omitempty"`
	// InputReferences supplies style/subject reference images (Veo
	// "reference-to-video").
	InputReferences []map[string]any `json:"input_references,omitempty"`
	PollInterval    time.Duration    `json:"-"`
	Timeout         time.Duration    `json:"-"`
	// Extra passes through additional model-specific parameters (e.g. Veo's
	// `personGeneration`).
	Extra           map[string]any   `json:"-"`
}

// ImageRequest holds parameters for image generation.
type ImageRequest struct {
	Prompt      string         `json:"prompt"`
	Model       string         `json:"model,omitempty"`
	Size        string         `json:"size,omitempty"`
	Quality     string         `json:"quality,omitempty"`
	// ImageURLs are reference / source images for image+text→image models
	// (e.g. x-ai/grok-imagine-image-quality). Each may be an http(s) or
	// data: URL.
	ImageURLs   []string       `json:"-"`
	ImageConfig *ImageConfig   `json:"image_config,omitempty"`
	// Extra passes through additional model-specific parameters.
	Extra       map[string]any `json:"-"`
}

// ImageConfig holds OpenRouter-specific image configuration.
type ImageConfig struct {
	AspectRatio               string     `json:"aspect_ratio,omitempty"`
	ImageSize                 string     `json:"image_size,omitempty"`
	// Strength is the image-to-image blend (0–1, model-dependent).
	Strength                  *float64   `json:"strength,omitempty"`
	// Style hint (e.g. Recraft V3 styles).
	Style                     string     `json:"style,omitempty"`
	// RgbColors is a color palette — array of [r,g,b].
	RgbColors                 [][3]int   `json:"rgb_colors,omitempty"`
	// BackgroundRgbColor is [r,g,b].
	BackgroundRgbColor        *[3]int    `json:"background_rgb_color,omitempty"`
	SuperResolutionReferences []string   `json:"super_resolution_references,omitempty"`
	FontInputs                []FontInput `json:"font_inputs,omitempty"`
}

// FontInput configures custom text rendering for compatible image models.
type FontInput struct {
	FontURL string `json:"font_url"`
	Text    string `json:"text"`
}

// AudioRequest holds parameters for audio generation.
type AudioRequest struct {
	Text   string         `json:"text"`
	Model  string         `json:"model,omitempty"`
	Voice  string         `json:"voice,omitempty"`
	Format string         `json:"format,omitempty"`
	// Speed multiplier (OpenAI TTS respects; other models ignore).
	Speed  *float64       `json:"speed,omitempty"`
	// Extra passes through additional model-specific parameters.
	Extra  map[string]any `json:"-"`
}

// MediaResponse holds the result of a media generation call.
type MediaResponse struct {
	Text        string      `json:"text"`
	Images      []ImageData `json:"images,omitempty"`
	Audio       *AudioData  `json:"audio,omitempty"`
	Files       []FileData  `json:"files,omitempty"`
	Videos      []VideoData `json:"videos,omitempty"`
	RawResponse any         `json:"raw_response,omitempty"`
}

// ImageData holds data for a generated image.
type ImageData struct {
	URL           string `json:"url,omitempty"`
	B64JSON       string `json:"b64_json,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

// AudioData holds data for generated audio.
type AudioData struct {
	Data   string `json:"data,omitempty"`
	Format string `json:"format"`
	URL    string `json:"url,omitempty"`
}

// FileData holds data for a generated file.
type FileData struct {
	URL      string `json:"url,omitempty"`
	Data     string `json:"data,omitempty"`
	MimeType string `json:"mime_type,omitempty"`
	Filename string `json:"filename,omitempty"`
}

// VideoData holds data for a generated video.
type VideoData struct {
	URL         string  `json:"url,omitempty"`
	Data        string  `json:"data,omitempty"`
	MimeType    string  `json:"mime_type,omitempty"`
	Filename    string  `json:"filename,omitempty"`
	Duration    float64 `json:"duration,omitempty"`
	Resolution  string  `json:"resolution,omitempty"`
	AspectRatio string  `json:"aspect_ratio,omitempty"`
	HasAudio    bool    `json:"has_audio,omitempty"`
	CostUSD     float64 `json:"cost_usd,omitempty"`
}

// MediaProvider defines the interface for media generation backends.
type MediaProvider interface {
	Name() string
	SupportedModalities() []string
	GenerateImage(ctx context.Context, req ImageRequest) (*MediaResponse, error)
	GenerateAudio(ctx context.Context, req AudioRequest) (*MediaResponse, error)
	GenerateVideo(ctx context.Context, req VideoRequest) (*MediaResponse, error)
}

// MediaRouter dispatches (model, capability) pairs to the correct MediaProvider.
type MediaRouter struct {
	providers []routerEntry
}

type routerEntry struct {
	prefix   string
	provider MediaProvider
}

// NewMediaRouter creates a new MediaRouter.
func NewMediaRouter() *MediaRouter {
	return &MediaRouter{}
}

// Register adds a provider with a model prefix. Longer prefixes match first.
func (r *MediaRouter) Register(prefix string, provider MediaProvider) {
	r.providers = append(r.providers, routerEntry{prefix: prefix, provider: provider})
	sort.Slice(r.providers, func(i, j int) bool {
		return len(r.providers[i].prefix) > len(r.providers[j].prefix)
	})
}

// Resolve finds the provider for a model and capability.
func (r *MediaRouter) Resolve(model, capability string) (MediaProvider, error) {
	for _, entry := range r.providers {
		if strings.HasPrefix(model, entry.prefix) {
			for _, mod := range entry.provider.SupportedModalities() {
				if mod == capability {
					return entry.provider, nil
				}
			}
		}
	}
	return nil, fmt.Errorf("no provider for model %q with %q capability", model, capability)
}
