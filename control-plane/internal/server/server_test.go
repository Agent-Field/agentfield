package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"

	"github.com/gin-gonic/gin"
	"gopkg.in/yaml.v3"
)

func TestCheckStorageHealthOverride(t *testing.T) {
	srv := &AgentFieldServer{
		storageHealthOverride: func(context.Context) gin.H {
			return gin.H{"status": "healthy"}
		},
	}

	result := srv.checkStorageHealth(context.Background())
	if status, ok := result["status"].(string); !ok || status != "healthy" {
		t.Fatalf("expected healthy status, got %+v", result)
	}
}

func TestCheckStorageHealthWithoutStorage(t *testing.T) {
	srv := &AgentFieldServer{}
	result := srv.checkStorageHealth(context.Background())
	if status, ok := result["status"].(string); !ok || status != "healthy" {
		t.Fatalf("expected default healthy status when storage nil, got %+v", result)
	}
}

func TestCheckStorageHealthContextError(t *testing.T) {
	srv := &AgentFieldServer{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result := srv.checkStorageHealth(ctx)
	if status, ok := result["status"].(string); !ok || status != "unhealthy" {
		t.Fatalf("expected unhealthy status for cancelled context, got %+v", result)
	}
}

type fakeCache struct {
	mu     sync.RWMutex
	store  map[string]string
	setErr error
	getErr error
}

func newFakeCache() *fakeCache {
	return &fakeCache{store: make(map[string]string)}
}

func (c *fakeCache) Set(key string, value interface{}, ttl time.Duration) error {
	if c.setErr != nil {
		return c.setErr
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if s, ok := value.(string); ok {
		c.store[key] = s
	}
	return nil
}

func (c *fakeCache) Get(key string, dest interface{}) error {
	if c.getErr != nil {
		return c.getErr
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if ptr, ok := dest.(*string); ok {
		*ptr = c.store[key]
	}
	return nil
}

func (c *fakeCache) Delete(key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.store, key)
	return nil
}

func (c *fakeCache) Exists(key string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	_, ok := c.store[key]
	return ok
}

func (c *fakeCache) Subscribe(channel string) (<-chan storage.CacheMessage, error) {
	return nil, nil
}

func (c *fakeCache) Publish(channel string, message interface{}) error {
	return nil
}

func TestCheckCacheHealthHealthy(t *testing.T) {
	cache := newFakeCache()
	srv := &AgentFieldServer{cache: cache}

	result := srv.checkCacheHealth(context.Background())
	if status, ok := result["status"].(string); !ok || status != "healthy" {
		t.Fatalf("expected healthy cache status, got %+v", result)
	}
}

func TestCheckCacheHealthSetError(t *testing.T) {
	cache := newFakeCache()
	cache.setErr = context.DeadlineExceeded
	srv := &AgentFieldServer{cache: cache}

	result := srv.checkCacheHealth(context.Background())
	if status, ok := result["status"].(string); !ok || status != "unhealthy" {
		t.Fatalf("expected unhealthy cache status, got %+v", result)
	}
}

func TestCheckCacheHealthGetError(t *testing.T) {
	cache := newFakeCache()
	cache.getErr = context.DeadlineExceeded
	srv := &AgentFieldServer{cache: cache}

	result := srv.checkCacheHealth(context.Background())
	if status, ok := result["status"].(string); !ok || status != "unhealthy" {
		t.Fatalf("expected unhealthy cache status, got %+v", result)
	}
}

func TestHealthCheckHandlerHealthy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{
		storageHealthOverride: func(context.Context) gin.H { return gin.H{"status": "healthy"} },
		cacheHealthOverride:   func(context.Context) gin.H { return gin.H{"status": "healthy"} },
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req, _ := http.NewRequest(http.MethodGet, "/health", nil)
	c.Request = req

	srv.healthCheckHandler(c)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 status, got %d", w.Code)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload["status"] != "healthy" {
		t.Fatalf("expected response status healthy, got %+v", payload)
	}
}

func TestHealthCheckHandlerCacheOptional(t *testing.T) {
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{
		storageHealthOverride: func(context.Context) gin.H { return gin.H{"status": "healthy"} },
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req, _ := http.NewRequest(http.MethodGet, "/health", nil)
	c.Request = req

	srv.healthCheckHandler(c)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 status, got %d", w.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	checks := payload["checks"].(map[string]any)
	cacheCheck := checks["cache"].(map[string]any)
	if cacheCheck["message"] != "cache not configured (optional)" {
		t.Fatalf("expected optional cache message, got %+v", cacheCheck)
	}
}

func TestHealthCheckHandlerUnhealthyStorage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{
		storageHealthOverride: func(context.Context) gin.H { return gin.H{"status": "unhealthy"} },
		cacheHealthOverride:   func(context.Context) gin.H { return gin.H{"status": "healthy"} },
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req, _ := http.NewRequest(http.MethodGet, "/health", nil)
	c.Request = req

	srv.healthCheckHandler(c)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 status, got %d", w.Code)
	}
}

func TestHealthCheckHandlerWithoutStorage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req, _ := http.NewRequest(http.MethodGet, "/health", nil)
	c.Request = req

	srv.healthCheckHandler(c)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 status, got %d", w.Code)
	}
}

func TestGenerateAgentFieldServerIDDeterministic(t *testing.T) {
	dir1 := filepath.Join("/tmp", "agentfield-test-1")
	dir2 := filepath.Join("/tmp", "agentfield-test-2")

	id1 := generateAgentFieldServerID(dir1)
	id1Again := generateAgentFieldServerID(dir1)
	id2 := generateAgentFieldServerID(dir2)

	if id1 != id1Again {
		t.Fatal("expected deterministic ID for same path")
	}
	if id1 == id2 {
		t.Fatal("expected different IDs for different paths")
	}
}

func TestUnregisterAgentFromMonitoring_NoNodeID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{}

	router := gin.New()
	router.DELETE("/nodes/:node_id/monitoring", srv.unregisterAgentFromMonitoring)

	req := httptest.NewRequest(http.MethodDelete, "/nodes//monitoring", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 status, got %d", resp.Code)
	}
}

func TestUnregisterAgentFromMonitoring_NoMonitor(t *testing.T) {
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{}

	router := gin.New()
	router.DELETE("/nodes/:node_id/monitoring", srv.unregisterAgentFromMonitoring)

	req := httptest.NewRequest(http.MethodDelete, "/nodes/node-1/monitoring", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 status, got %d", resp.Code)
	}
}

func TestSyncPackagesFromRegistry(t *testing.T) {
	storage := newStubPackageStorage()

	agentfieldHome := t.TempDir()
	pkgDir := filepath.Join(agentfieldHome, "packages", "mypkg")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatalf("failed to create package dir: %v", err)
	}

	packageContent := []byte(`name: Test Package\nversion: 1.0.0`)
	if err := os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), packageContent, 0o644); err != nil {
		t.Fatalf("failed to write agentfield-package.yaml: %v", err)
	}

	installedContent := []byte("installed:\n  test-package:\n    name: Test Package\n    version: \"1.0.0\"\n    description: Test description\n    path: \"" + pkgDir + "\"\n    source: local\n")
	if err := os.WriteFile(filepath.Join(agentfieldHome, "installed.yaml"), installedContent, 0o644); err != nil {
		t.Fatalf("failed to write installed.yaml: %v", err)
	}
	var reg InstallationRegistry
	if data, err := os.ReadFile(filepath.Join(agentfieldHome, "installed.yaml")); err == nil {
		_ = yaml.Unmarshal(data, &reg)
	}
	if len(reg.Installed) == 0 {
		t.Fatal("expected registry to contain installed package")
	}

	if err := SyncPackagesFromRegistry(agentfieldHome, storage); err != nil {
		t.Fatalf("SyncPackagesFromRegistry returned error: %v", err)
	}

	if len(storage.getCalls) == 0 {
		t.Fatalf("expected GetAgentPackage to be called, got %d", len(storage.getCalls))
	}
}

func TestSyncPackagesFromRegistryMissingFile(t *testing.T) {
	storage := newStubPackageStorage()
	agentfieldHome := t.TempDir()

	if err := SyncPackagesFromRegistry(agentfieldHome, storage); err != nil {
		t.Fatalf("expected nil error when registry file missing, got %v", err)
	}
	if len(storage.packages) != 0 {
		t.Fatalf("expected no packages to be stored, found %d", len(storage.packages))
	}
}
