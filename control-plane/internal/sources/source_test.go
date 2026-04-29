package sources

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"
)

// fakeHTTPSource lets the registry tests run without depending on first-party
// source impls (which would create import cycles for tests in this package).
type fakeHTTPSource struct {
	name      string
	secretReq bool
	emit      []Event
	err       error
}

func (f *fakeHTTPSource) Name() string                  { return f.name }
func (f *fakeHTTPSource) Kind() Kind                    { return KindHTTP }
func (f *fakeHTTPSource) SecretRequired() bool          { return f.secretReq }
func (f *fakeHTTPSource) ConfigSchema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (f *fakeHTTPSource) Validate(json.RawMessage) error {
	return nil
}
func (f *fakeHTTPSource) HandleRequest(_ context.Context, _ *RawRequest, _ json.RawMessage, _ string) ([]Event, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.emit, nil
}

type fakeLoopSource struct{ name string }

func (f *fakeLoopSource) Name() string                                          { return f.name }
func (f *fakeLoopSource) Kind() Kind                                            { return KindLoop }
func (f *fakeLoopSource) SecretRequired() bool                                  { return false }
func (f *fakeLoopSource) ConfigSchema() json.RawMessage                         { return json.RawMessage(`{}`) }
func (f *fakeLoopSource) Validate(json.RawMessage) error                        { return nil }
func (f *fakeLoopSource) Run(_ context.Context, _ json.RawMessage, _ string, _ func(Event)) error {
	return nil
}

// withCleanRegistry isolates a test from any global Source registrations
// (including first-party init() side-effects) by saving and restoring the
// registry around the test body.
func withCleanRegistry(t *testing.T, fn func()) {
	t.Helper()
	registryMu.Lock()
	saved := registry
	registry = map[string]Source{}
	registryMu.Unlock()
	defer func() {
		registryMu.Lock()
		registry = saved
		registryMu.Unlock()
	}()
	fn()
}

func TestRegisterAndGet(t *testing.T) {
	withCleanRegistry(t, func() {
		Register(&fakeHTTPSource{name: "test"})
		s, ok := Get("test")
		if !ok {
			t.Fatal("expected source to be registered")
		}
		if s.Name() != "test" {
			t.Errorf("name=%q, want test", s.Name())
		}
	})
}

func TestRegister_PanicsOnNil(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on nil source")
		}
	}()
	Register(nil)
}

func TestRegister_PanicsOnDuplicate(t *testing.T) {
	withCleanRegistry(t, func() {
		Register(&fakeHTTPSource{name: "dup"})
		defer func() {
			if recover() == nil {
				t.Error("expected panic on duplicate name")
			}
		}()
		Register(&fakeHTTPSource{name: "dup"})
	})
}

func TestRegister_PanicsOnEmptyName(t *testing.T) {
	withCleanRegistry(t, func() {
		defer func() {
			if recover() == nil {
				t.Error("expected panic on empty name")
			}
		}()
		Register(&fakeHTTPSource{name: ""})
	})
}

func TestList_SortedByName(t *testing.T) {
	withCleanRegistry(t, func() {
		Register(&fakeHTTPSource{name: "zoo", secretReq: true})
		Register(&fakeLoopSource{name: "alpha"})
		Register(&fakeHTTPSource{name: "mid"})
		got := List()
		if len(got) != 3 {
			t.Fatalf("want 3 entries, got %d", len(got))
		}
		want := []string{"alpha", "mid", "zoo"}
		for i, c := range got {
			if c.Name != want[i] {
				t.Errorf("List()[%d].Name=%q, want %q", i, c.Name, want[i])
			}
		}
		if got[0].Kind != "loop" {
			t.Errorf("alpha.Kind=%q want loop", got[0].Kind)
		}
		if !got[2].SecretRequired {
			t.Error("zoo should require secret")
		}
	})
}

func TestHandleHTTP_DispatchesToRegisteredSource(t *testing.T) {
	withCleanRegistry(t, func() {
		Register(&fakeHTTPSource{
			name: "fake",
			emit: []Event{{Type: "x", IdempotencyKey: "k"}},
		})
		req := &RawRequest{
			Headers: http.Header{},
			Body:    []byte(`{}`),
			URL:     &url.URL{Path: "/sources/abc"},
			Method:  "POST",
		}
		events, err := HandleHTTP(context.Background(), "fake", req, nil, "")
		if err != nil {
			t.Fatalf("HandleHTTP err: %v", err)
		}
		if len(events) != 1 {
			t.Fatalf("want 1 event, got %d", len(events))
		}
		// Registry should have stamped ReceivedAt.
		if events[0].ReceivedAt.IsZero() {
			t.Error("expected ReceivedAt to be stamped")
		}
		if time.Since(events[0].ReceivedAt) > time.Minute {
			t.Errorf("ReceivedAt too old: %v", events[0].ReceivedAt)
		}
	})
}

func TestHandleHTTP_UnknownSource(t *testing.T) {
	withCleanRegistry(t, func() {
		_, err := HandleHTTP(context.Background(), "nope", &RawRequest{}, nil, "")
		var unknown ErrUnknownSource
		if !errors.As(err, &unknown) {
			t.Fatalf("want ErrUnknownSource, got %T (%v)", err, err)
		}
	})
}

func TestHandleHTTP_LoopSourceMismatch(t *testing.T) {
	withCleanRegistry(t, func() {
		Register(&fakeLoopSource{name: "ticker"})
		_, err := HandleHTTP(context.Background(), "ticker", &RawRequest{}, nil, "")
		var mismatch ErrSourceKindMismatch
		if !errors.As(err, &mismatch) {
			t.Fatalf("want ErrSourceKindMismatch, got %T (%v)", err, err)
		}
		if !strings.Contains(err.Error(), "ticker") {
			t.Errorf("expected source name in error, got %v", err)
		}
	})
}

func TestHandleHTTP_PropagatesSourceError(t *testing.T) {
	withCleanRegistry(t, func() {
		Register(&fakeHTTPSource{name: "boom", err: errors.New("verify failed")})
		_, err := HandleHTTP(context.Background(), "boom", &RawRequest{}, nil, "")
		if err == nil || !strings.Contains(err.Error(), "verify failed") {
			t.Fatalf("want propagated error, got %v", err)
		}
	})
}
