//go:build !windows

package services

import (
	"fmt"
	"net"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDevServiceDiscoverAgentPort(t *testing.T) {
	// discoverAgentPort scans the hardcoded 8001-8999 range, so we must bind
	// within that range. Find and hold a port atomically via startLocalServer.
	port := findPortInAgentRange(t)
	server := startLocalServer(t, port, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	service := &DefaultDevService{}
	discoveredPort, err := service.discoverAgentPort(2 * time.Second)
	require.NoError(t, err)
	assert.Equal(t, port, discoveredPort)
}

func TestDevServiceDiscoverAgentPortTimeout(t *testing.T) {
	service := &DefaultDevService{}
	port, err := service.discoverAgentPort(750 * time.Millisecond)
	require.Error(t, err)
	assert.Equal(t, 0, port)
	assert.Contains(t, err.Error(), "could not discover agent port")
}

func TestDevServiceWaitForAgent(t *testing.T) {
	var requestCount int32
	server, port := startLocalServerOnFreePort(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		if atomic.AddInt32(&requestCount, 1) < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	service := &DefaultDevService{}
	require.NoError(t, service.waitForAgent(port, 2*time.Second))
}

func TestDevServiceWaitForAgentTimeout(t *testing.T) {
	service := &DefaultDevService{}
	// Bind a listener so the port is deterministically occupied, then close it
	// right before calling waitForAgent. Using port 0 gives us an ephemeral port
	// that no test server is listening on.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	// Now nothing is listening on this port, so waitForAgent must time out.
	err = service.waitForAgent(port, 750*time.Millisecond)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "agent did not become ready")
}

func TestDevServiceDisplayDevCapabilities(t *testing.T) {
	server, port := startLocalServerOnFreePort(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/reasoners":
			_, _ = fmt.Fprint(w, `{"reasoners":[{"id":"reasoner-a"}]}`)
		case "/skills":
			_, _ = fmt.Fprint(w, `{"skills":[{"id":"skill-a"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	service := &DefaultDevService{}
	require.NoError(t, service.displayDevCapabilities(port))
}

func TestDevServiceDisplayDevCapabilitiesDecodeError(t *testing.T) {
	server, port := startLocalServerOnFreePort(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/reasoners":
			_, _ = fmt.Fprint(w, `{"reasoners":[`)
		case "/skills":
			_, _ = fmt.Fprint(w, `{"skills":[{"id":"skill-a"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	service := &DefaultDevService{}
	err := service.displayDevCapabilities(port)
	require.Error(t, err)
}

func TestAgentServiceDisplayCapabilities(t *testing.T) {
	server, port := startLocalServerOnFreePort(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/reasoners":
			_, _ = fmt.Fprint(w, `{"reasoners":[{"id":"reasoner-a"}]}`)
		case "/skills":
			_, _ = fmt.Fprint(w, `{"skills":[{"id":"skill-a"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	service := &DefaultAgentService{}
	require.NoError(t, service.displayCapabilities(packages.InstalledPackage{Name: "agent"}, port))
}

func TestAgentServiceDisplayCapabilitiesDecodeError(t *testing.T) {
	server, port := startLocalServerOnFreePort(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/reasoners":
			_, _ = fmt.Fprint(w, `{"reasoners":[{"id":"reasoner-a"}]}`)
		case "/skills":
			_, _ = fmt.Fprint(w, `{"skills":[`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	service := &DefaultAgentService{}
	err := service.displayCapabilities(packages.InstalledPackage{Name: "agent"}, port)
	require.Error(t, err)
}
