package codeaf

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Finding 7: a reserved origin can only be reached through the configured proxy.
// Each child gets a fresh ProxyFromEnvironment cache.
func TestEnsureProxyEnvironment(t *testing.T) {
	if os.Getenv("CODEAF_PROXY_CHILD") == "1" {
		err := Ensure(Options{Home: t.TempDir(), BaseURL: os.Getenv("CODEAF_PROXY_ORIGIN")})
		if err == nil || !strings.Contains(err.Error(), "Service Unavailable") {
			t.Fatalf("proxy response missing: %v", err)
		}
		return
	}
	for _, scheme := range []string{"http", "https"} {
		for _, lower := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/lower=%v", scheme, lower), func(t *testing.T) {
				var requests atomic.Int32
				proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					requests.Add(1)
					if scheme == "https" && r.Method != "CONNECT" {
						t.Errorf("method=%s", r.Method)
					}
					w.WriteHeader(503)
				}))
				defer proxy.Close()
				key := strings.ToUpper(scheme) + "_PROXY"
				if lower {
					key = strings.ToLower(key)
				}
				cmd := exec.Command(os.Args[0], "-test.run=^TestEnsureProxyEnvironment$")
				cmd.Env = append(proxyTestEnv(), "CODEAF_PROXY_CHILD=1", "CODEAF_PROXY_ORIGIN="+scheme+"://codeaf.invalid", key+"="+proxy.URL)
				out, err := cmd.CombinedOutput()
				if err != nil || requests.Load() != 1 {
					t.Fatalf("proxy requests=%d err=%v\n%s", requests.Load(), err, out)
				}
			})
		}
	}
}

func proxyTestEnv() []string {
	var env []string
	for _, pair := range os.Environ() {
		key := strings.ToUpper(strings.SplitN(pair, "=", 2)[0])
		switch key {
		case "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "AGENTFIELD_SKIP_CODEAF", "AGENTFIELD_SKIP_AFORGE":
			continue
		}
		env = append(env, pair)
	}
	return env
}

// A4: proxy redirects still install only a fixture with the embedded/test pin.
func TestEnsureProxyRedirectHashAndNoProxy(t *testing.T) {
	if mode := os.Getenv("CODEAF_PROXY_MODE"); mode != "" {
		opts := Options{Home: t.TempDir(), BaseURL: "http://codeaf.invalid/release", GOOS: "linux", GOARCH: "amd64"}
		sum := sha256.Sum256([]byte("verified fixture"))
		pin := fmt.Sprintf("%x", sum)
		if mode == "mismatch" {
			pin = strings.Repeat("0", 64)
		}
		err := ensure(opts, map[string]string{"codeaf-linux-amd64": pin})
		switch mode {
		case "bypass":
			if err == nil || !strings.Contains(err.Error(), "no such host") {
				t.Fatalf("NO_PROXY was ignored: %v", err)
			}
		case "mismatch":
			if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
				t.Fatalf("hash bypassed: %v", err)
			}
			if _, err := os.Stat(filepath.Join(opts.Home, "bin", "codeaf")); !os.IsNotExist(err) {
				t.Fatal("bad hash published")
			}
		default:
			if err != nil {
				t.Fatal(err)
			}
			if !alreadyInstalled(filepath.Join(opts.Home, "bin", "codeaf"), filepath.Join(opts.Home, "bin", versionMarker)) {
				t.Fatal("verified binary missing")
			}
		}
		return
	}
	for _, mode := range []string{"verified", "mismatch", "bypass"} {
		for _, lower := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/lower=%v", mode, lower), func(t *testing.T) {
				var requests atomic.Int32
				proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					requests.Add(1)
					if r.URL.Path == "/release/codeaf-linux-amd64" {
						http.Redirect(w, r, "http://codeaf.invalid/payload", http.StatusTemporaryRedirect)
						return
					}
					_, _ = io.WriteString(w, "verified fixture")
				}))
				defer proxy.Close()
				key := "NO_PROXY"
				if lower {
					key = "no_proxy"
				}
				env := append(proxyTestEnv(), "CODEAF_PROXY_MODE="+mode, "HTTP_PROXY="+proxy.URL)
				if mode == "bypass" {
					env = append(env, key+"=codeaf.invalid")
				}
				cmd := exec.Command(os.Args[0], "-test.run=^TestEnsureProxyRedirectHashAndNoProxy$")
				cmd.Env = env
				out, err := cmd.CombinedOutput()
				want := int32(2)
				if mode == "bypass" {
					want = 0
				}
				if err != nil || requests.Load() != want {
					t.Fatalf("requests=%d want=%d err=%v\n%s", requests.Load(), want, err, out)
				}
			})
		}
	}
}

func TestEnsureHonorsContextDuringDownload(t *testing.T) {
	noSkip(t)
	entered := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	opts := Options{Home: t.TempDir(), BaseURL: server.URL, Context: ctx}
	done := make(chan error, 1)
	go func() { done <- Ensure(opts) }()
	<-entered
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("context lost: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("download ignored cancellation")
	}
	entries, err := os.ReadDir(filepath.Join(opts.Home, "bin"))
	if err != nil || len(entries) != 1 || entries[0].Name() != ".codeaf.lock" {
		t.Fatalf("partial files after cancellation: %v %v", entries, err)
	}
}

func TestEnsureBestEffortRetryAfterFailure(t *testing.T) {
	noSkip(t)
	resetBestEffort()
	t.Cleanup(resetBestEffort)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { requests.Add(1); w.WriteHeader(503) }))
	defer server.Close()
	opts := Options{Home: t.TempDir(), BaseURL: server.URL}
	_ = EnsureBestEffort(opts, io.Discard)
	_ = EnsureBestEffort(opts, io.Discard)
	if requests.Load() != 1 {
		t.Fatal("failure retry ignored 60s backoff")
	}
	// Advance only the retry clock, without resetting the resolver/flight state.
	bestEffort.Lock()
	bestEffort.lastDownload = time.Now().Add(-time.Minute)
	bestEffort.Unlock()
	_ = EnsureBestEffort(opts, io.Discard)
	if requests.Load() != 2 {
		t.Fatal("failure permanently cached")
	}
}

func TestEnsureManagedHomeAndProgress(t *testing.T) {
	opts, pins, requests := fixture(t)
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	progress := 0
	opts.OnDownload = func() { progress++ }
	for i := 0; i < 2; i++ {
		if err := ensure(opts, pins); err != nil {
			t.Fatal(err)
		}
	}
	if progress != 1 || requests.Load() != 1 {
		t.Fatalf("progress=%d downloads=%d", progress, requests.Load())
	}
	bin := filepath.Join(opts.Home, "bin")
	if !alreadyInstalled(filepath.Join(bin, "codeaf"), filepath.Join(bin, versionMarker)) {
		t.Fatal("managed home placement missing")
	}
}
