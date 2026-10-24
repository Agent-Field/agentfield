package harness

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A4 / V6: fresh processes avoid net/http's process-wide proxy environment cache.
func TestCodeAFDownloadProxyEnvironment(t *testing.T) {
	if fixture := os.Getenv("AGENTFIELD_CODEAF_PROXY_TEST"); fixture != "" {
		runCodeAFProxyFixture(t, fixture)
		return
	}
	for _, fixture := range []string{"http upper", "http lower", "https upper", "https lower", "http bypass upper", "https bypass lower"} {
		t.Run(fixture, func(t *testing.T) {
			t.Setenv("AGENTFIELD_CODEAF_PROXY_TEST", fixture)
			binary, err := os.Executable()
			require.NoError(t, err)
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			output, err := exec.CommandContext(ctx, binary, "-test.run=^TestCodeAFDownloadProxyEnvironment$", "-test.count=1").CombinedOutput()
			require.NoError(t, err, "%s", output)
		})
	}
}

func runCodeAFProxyFixture(t *testing.T, fixture string) {
	t.Helper()
	isolateCodeAF(t)
	for _, key := range []string{"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy", "REQUEST_METHOD"} {
		t.Setenv(key, "")
	}
	data := "verified proxy fixture"
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(data)))
	var proxyRequests, assetRequests atomic.Int32
	serveAsset := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assetRequests.Add(1)
		if r.URL.Path != "/redirected" {
			http.Redirect(w, r, "/redirected", http.StatusTemporaryRedirect)
			return
		}
		_, _ = io.WriteString(w, data)
	})
	origin := httptest.NewTLSServer(serveAsset)
	defer origin.Close()
	httpOrigin := httptest.NewServer(serveAsset)
	defer httpOrigin.Close()
	originURL, err := url.Parse(origin.URL)
	require.NoError(t, err)
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyRequests.Add(1)
		if r.Method != http.MethodConnect {
			serveAsset.ServeHTTP(w, r)
			return
		}
		upstream, err := net.DialTimeout("tcp", originURL.Host, 5*time.Second)
		if err != nil {
			http.Error(w, "fixture dial failed", http.StatusBadGateway)
			return
		}
		defer upstream.Close()
		client, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer client.Close()
		_, _ = io.WriteString(client, "HTTP/1.1 200 Connection Established\r\n\r\n")
		go func() { _, _ = io.Copy(upstream, client) }()
		_, _ = io.Copy(client, upstream)
	}))
	defer proxy.Close()
	scheme := strings.Fields(fixture)[0]
	key := strings.ToUpper(scheme) + "_PROXY"
	if strings.Contains(fixture, "lower") {
		key = strings.ToLower(key)
	}
	t.Setenv(key, proxy.URL)
	if strings.Contains(fixture, "bypass") {
		key = "NO_PROXY"
		if strings.Contains(fixture, "lower") {
			key = strings.ToLower(key)
		}
		t.Setenv(key, "codeaf-proxy.invalid")
	}
	base := scheme + "://codeaf-proxy.invalid"
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", base)
	req, err := http.NewRequest(http.MethodGet, base, nil)
	require.NoError(t, err)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	defer transport.CloseIdleConnections()
	require.NotNil(t, transport.Proxy, "SDK default transport must retain ProxyFromEnvironment")
	selected, err := transport.Proxy(req)
	require.NoError(t, err)
	if strings.Contains(fixture, "bypass") {
		assert.Nil(t, selected)
		directURL := originURL
		if scheme == "http" {
			directURL, err = url.Parse(httpOrigin.URL)
			require.NoError(t, err)
		}
		// Route the reserved fixture hostname locally after proxy selection.
		transport.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", directURL.Host)
		}
	} else {
		require.NotNil(t, selected)
		assert.Equal(t, proxy.URL, selected.String())
	}
	// Trust only this test's TLS origin while retaining the default proxy function.
	transport.TLSClientConfig = origin.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
	transport.TLSClientConfig.ServerName = originURL.Hostname()
	previous := http.DefaultTransport
	http.DefaultTransport = transport
	defer func() { http.DefaultTransport = previous }()
	home := codeAFManagedPaths()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	path, err := installCodeAF(ctx, home, "asset", hash)
	require.NoError(t, err)
	assert.True(t, codeAFInstalled(path))
	if strings.Contains(fixture, "bypass") {
		assert.Zero(t, proxyRequests.Load())
	} else {
		assert.Positive(t, proxyRequests.Load())
	}
	assert.Equal(t, int32(2), assetRequests.Load(), "redirect must reach the verified asset")
	badPath := filepath.Join(t.TempDir(), "codeaf")
	_, err = installCodeAF(ctx, badPath, "asset", strings.Repeat("0", 64))
	require.ErrorContains(t, err, "sha256 mismatch")
	assert.NoFileExists(t, badPath)
	assert.NoFileExists(t, filepath.Join(filepath.Dir(badPath), codeAFVersionFilename))
}
