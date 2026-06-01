// Package sources defines the trigger Source plugin interface and registry.
//
// A Source is a first-party or community-contributed adapter that converts an
// external signal — an HTTP webhook, a cron tick, a queue message — into a
// normalized inbound Event that the control plane dispatches to a reasoner.
//
// First-party Source impls live under sources/<name>/<name>.go and register
// themselves in their package init() via Register. The blank-import aggregator
// at sources/all wires every first-party source into a single import.
package sources

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"sort"
	"sync"
	"time"
)

// Kind distinguishes how a Source produces events.
type Kind int

const (
	// KindHTTP sources receive events through inbound HTTP requests.
	KindHTTP Kind = iota
	// KindLoop sources run a long-lived goroutine that emits events on a schedule
	// or by polling an external system.
	KindLoop
)

// String renders the Kind for logging and the JSON catalog.
func (k Kind) String() string {
	switch k {
	case KindHTTP:
		return "http"
	case KindLoop:
		return "loop"
	default:
		return "unknown"
	}
}

// RawRequest carries the unparsed inbound HTTP request for an HTTPSource to verify.
type RawRequest struct {
	Headers http.Header
	Body    []byte
	URL     *url.URL
	Method  string
}

// Event is the normalized record produced by a Source. The control plane
// persists every Event, mints a VC over it (when DID is enabled), and
// dispatches it to the trigger's target reasoner.
type Event struct {
	// Type is the source-specific event type (e.g. "payment_intent.succeeded",
	// "pull_request.opened", or "tick" for cron). Used for filtering.
	Type string
	// IdempotencyKey is the provider's globally unique event ID. The control
	// plane dedups on (source_name, idempotency_key) so retries from a provider
	// produce exactly one inbound_events row. Empty string disables dedup.
	IdempotencyKey string
	// Raw is the original payload bytes as received.
	Raw json.RawMessage
	// Normalized is a Source-curated subset suitable for downstream reasoners.
	// Sources should populate this even when it equals Raw, so the dispatcher
	// has a stable contract regardless of source.
	Normalized json.RawMessage
	// ReceivedAt is when the event entered the control plane. Sources should
	// not override this — leave zero and the registry will stamp it.
	ReceivedAt time.Time
}

// Source is the common contract every plugin satisfies.
//
// Implementations also satisfy exactly one of HTTPSource or LoopSource depending
// on Kind(). The dispatcher routes calls based on the satisfied interface.
type Source interface {
	// Name is the unique registry key (e.g. "stripe", "github", "cron").
	Name() string
	// Kind reports whether the source is HTTP-driven or loop-driven.
	Kind() Kind
	// ConfigSchema returns a JSON Schema describing the per-trigger config. The
	// UI uses this to render a dynamic form when creating a trigger instance.
	ConfigSchema() json.RawMessage
	// SecretRequired reports whether the trigger must have a non-empty
	// secret_env_var pointing at an environment variable holding the provider
	// secret (e.g. Stripe webhook secret, GitHub webhook secret).
	SecretRequired() bool
	// Validate checks the per-trigger config payload before persistence. Return
	// an error to reject the trigger.
	Validate(cfg json.RawMessage) error
}

// HTTPSource is implemented by Sources that ingest via inbound HTTP. It owns
// signature verification — verification failures should return a non-nil error
// so the registry can return 401.
type HTTPSource interface {
	Source
	HandleRequest(ctx context.Context, req *RawRequest, cfg json.RawMessage, secret string) ([]Event, error)
}

// LoopSource is implemented by Sources that emit events from a long-lived
// goroutine (cron schedules, polling adapters, queue consumers). Run blocks
// until ctx is cancelled and must call emit for each event produced.
type LoopSource interface {
	Source
	Run(ctx context.Context, cfg json.RawMessage, secret string, emit func(Event)) error
}

var (
	registryMu sync.RWMutex
	registry   = map[string]Source{}
)

// Register adds a Source to the global registry. Intended to be called from
// init() of each first-party source package. Panics if name is empty or
// already registered — registration conflicts are programmer errors.
func Register(s Source) {
	if s == nil {
		panic("sources.Register: nil source")
	}
	name := s.Name()
	if name == "" {
		panic("sources.Register: source has empty name")
	}
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, exists := registry[name]; exists {
		panic("sources.Register: duplicate source name " + name)
	}
	registry[name] = s
}

// Get returns the registered Source for the given name, or false if no source
// has been registered under that name.
func Get(name string) (Source, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	s, ok := registry[name]
	return s, ok
}

// Catalog is the public-facing description of a registered Source.
type Catalog struct {
	Name           string          `json:"name"`
	Kind           string          `json:"kind"`
	SecretRequired bool            `json:"secret_required"`
	ConfigSchema   json.RawMessage `json:"config_schema"`
}

// List returns the registered sources sorted by name. Used by the
// GET /api/v1/sources catalog endpoint and by the UI's "new trigger" form.
func List() []Catalog {
	registryMu.RLock()
	defer registryMu.RUnlock()
	out := make([]Catalog, 0, len(registry))
	for _, s := range registry {
		out = append(out, Catalog{
			Name:           s.Name(),
			Kind:           s.Kind().String(),
			SecretRequired: s.SecretRequired(),
			ConfigSchema:   s.ConfigSchema(),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// stamp populates ReceivedAt for any event that left it zero. The dispatcher
// uses this so individual Source impls do not need to think about clocks.
func stamp(events []Event, now time.Time) []Event {
	for i := range events {
		if events[i].ReceivedAt.IsZero() {
			events[i].ReceivedAt = now
		}
	}
	return events
}

// HandleHTTP is a registry-level helper that resolves a Source by name,
// verifies it implements HTTPSource, and invokes HandleRequest. It is used by
// the public ingest handler so the handler stays free of registry plumbing.
func HandleHTTP(ctx context.Context, name string, req *RawRequest, cfg json.RawMessage, secret string) ([]Event, error) {
	s, ok := Get(name)
	if !ok {
		return nil, ErrUnknownSource{Name: name}
	}
	hs, ok := s.(HTTPSource)
	if !ok {
		return nil, ErrSourceKindMismatch{Name: name, Want: "http"}
	}
	events, err := hs.HandleRequest(ctx, req, cfg, secret)
	if err != nil {
		return nil, err
	}
	return stamp(events, time.Now().UTC()), nil
}

// ErrUnknownSource is returned when a trigger references a source that is not
// registered. Surfaced to the UI when a stale trigger row outlives a removed
// plugin.
type ErrUnknownSource struct{ Name string }

func (e ErrUnknownSource) Error() string { return "sources: unknown source " + e.Name }

// ErrSourceKindMismatch is returned when a caller invokes the wrong dispatch
// path for a source (e.g. POSTing to a cron source).
type ErrSourceKindMismatch struct {
	Name string
	Want string
}

func (e ErrSourceKindMismatch) Error() string {
	return "sources: source " + e.Name + " is not a " + e.Want + " source"
}
