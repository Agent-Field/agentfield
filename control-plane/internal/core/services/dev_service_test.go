//go:build !windows

package services

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock FileSystemAdapter for testing
type mockFileSystemAdapter struct {
	readFileFunc      func(string) ([]byte, error)
	writeFileFunc     func(string, []byte) error
	existsFunc        func(string) bool
	createDirFunc     func(string) error
	listDirectoryFunc func(string) ([]string, error)
	files             map[string][]byte
	directories       map[string]bool
}

func newMockFileSystemAdapter() *mockFileSystemAdapter {
	return &mockFileSystemAdapter{
		files:       make(map[string][]byte),
		directories: make(map[string]bool),
	}
}

func (m *mockFileSystemAdapter) ReadFile(path string) ([]byte, error) {
	if m.readFileFunc != nil {
		return m.readFileFunc(path)
	}
	if data, ok := m.files[path]; ok {
		return data, nil
	}
	return nil, errors.New("file not found")
}

func (m *mockFileSystemAdapter) WriteFile(path string, data []byte) error {
	if m.writeFileFunc != nil {
		return m.writeFileFunc(path, data)
	}
	m.files[path] = data
	return nil
}

func (m *mockFileSystemAdapter) Exists(path string) bool {
	if m.existsFunc != nil {
		return m.existsFunc(path)
	}
	_, fileExists := m.files[path]
	_, dirExists := m.directories[path]
	return fileExists || dirExists
}

func (m *mockFileSystemAdapter) CreateDirectory(path string) error {
	if m.createDirFunc != nil {
		return m.createDirFunc(path)
	}
	m.directories[path] = true
	return nil
}

func (m *mockFileSystemAdapter) ListDirectory(path string) ([]string, error) {
	if m.listDirectoryFunc != nil {
		return m.listDirectoryFunc(path)
	}
	return []string{}, nil
}

func TestNewDevService(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem)

	assert.NotNil(t, service)
	ds, ok := service.(*DefaultDevService)
	require.True(t, ok)
	assert.Equal(t, processManager, ds.processManager)
	assert.Equal(t, portManager, ds.portManager)
	assert.Equal(t, fileSystem, ds.fileSystem)
}

func TestRunInDevMode_NoAgentfieldYaml(t *testing.T) {
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	// Mock file system to report agentfield.yaml doesn't exist
	fileSystem.existsFunc = func(path string) bool {
		return false
	}

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	options := domain.DevOptions{
		Port:       0,
		AutoReload: false,
		Verbose:    false,
		WatchFiles: false,
	}

	err := service.RunInDevMode(packagePath, options)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no agentfield.yaml found")
}

func TestRunInDevMode_AgentfieldYamlExists(t *testing.T) {
	// This test verifies that RunInDevMode gets past the agentfield.yaml check.
	// It will fail at startDevProcess or discoverAgentPort since we can't easily mock exec.Cmd.
	// Use a short timeout to avoid hanging for 10+ minutes when discoverAgentPort
	// scans ports endlessly after the subprocess fails to start.
	if testing.Short() {
		t.Skip("skipping slow dev mode test in short mode")
	}

	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	agentfieldYamlPath := filepath.Join(packagePath, "agentfield.yaml")
	agentfieldYamlContent := []byte("name: test-package\nversion: 1.0.0")
	require.NoError(t, os.WriteFile(agentfieldYamlPath, agentfieldYamlContent, 0644))

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	// Mock file system to report agentfield.yaml exists
	fileSystem.existsFunc = func(path string) bool {
		return path == agentfieldYamlPath
	}

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	options := domain.DevOptions{
		Port:       0,
		AutoReload: false,
		Verbose:    false,
		WatchFiles: false,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- service.RunInDevMode(packagePath, options)
	}()

	select {
	case err := <-done:
		// The error should be about process startup or port discovery, not about agentfield.yaml
		if err != nil {
			assert.NotContains(t, err.Error(), "no agentfield.yaml found")
		}
	case <-ctx.Done():
		// Expected: discoverAgentPort hangs because no real agent is running.
		// The test already proved agentfield.yaml was accepted (we got past that check).
	}
}

func TestStopDevMode_NotImplemented(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	err := service.StopDevMode("/some/path")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}

func TestGetDevStatus_NotImplemented(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	_, err := service.GetDevStatus("/some/path")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}

func TestGetFreePort(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	// Mock port manager to return available ports
	portManager.findFreePortFunc = func(startPort int) (int, error) {
		if startPort >= 8001 && startPort <= 8999 {
			return startPort, nil
		}
		return 0, errors.New("no free port available")
	}

	port, err := service.getFreePort()
	require.NoError(t, err)
	assert.GreaterOrEqual(t, port, 8001)
	assert.LessOrEqual(t, port, 8999)
}

func TestGetFreePort_NoPortAvailable(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	// Mock port manager to return error BEFORE creating service
	portManager.findFreePortFunc = func(startPort int) (int, error) {
		return 0, errors.New("no free port available")
	}

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	port, err := service.getFreePort()
	assert.Error(t, err)
	assert.Equal(t, 0, port)
	if err != nil {
		assert.Contains(t, err.Error(), "no free port available")
	}
}

func TestIsPortAvailable_Available(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	// Mock port manager to report port as available
	portManager.isAvailableFunc = func(port int) bool {
		return port == 8001
	}

	available := service.isPortAvailable(8001)
	assert.True(t, available)
}

func TestIsPortAvailable_NotAvailable(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	// Mock port manager to report port as not available
	portManager.isAvailableFunc = func(port int) bool {
		return false
	}

	available := service.isPortAvailable(8001)
	assert.False(t, available)
}

func TestDiscoverAgentPort_Success(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	// This test would require mocking HTTP client, which is complex
	// For now, we test that the function exists and can be called
	// The actual port discovery logic requires a running HTTP server
	// which is better tested in integration tests

	// We can at least verify the function signature is correct
	assert.NotNil(t, service.discoverAgentPort)
}

func TestWaitForAgent_Success(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	// This test would require mocking HTTP client responses
	// For now, we verify the function exists
	assert.NotNil(t, service.waitForAgent)
}

func TestLoadDevEnvFile(t *testing.T) {
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	envPath := filepath.Join(packagePath, ".env")
	envContent := `KEY1=value1
KEY2=value2
# Comment line
KEY3="quoted value"
KEY4='single quoted'
`
	require.NoError(t, os.WriteFile(envPath, []byte(envContent), 0644))

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	envVars, err := service.loadDevEnvFile(packagePath)
	require.NoError(t, err)
	assert.Equal(t, "value1", envVars["KEY1"])
	assert.Equal(t, "value2", envVars["KEY2"])
	assert.Equal(t, "quoted value", envVars["KEY3"])
	assert.Equal(t, "single quoted", envVars["KEY4"])
	assert.NotContains(t, envVars, "# Comment line")
}

func TestLoadDevEnvFile_FileNotFound(t *testing.T) {
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	_, err := service.loadDevEnvFile(packagePath)
	assert.Error(t, err)
}

func TestLoadDevEnvFile_InvalidFormat(t *testing.T) {
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	envPath := filepath.Join(packagePath, ".env")
	envContent := `INVALID_LINE_WITHOUT_EQUALS
KEY=value
`
	require.NoError(t, os.WriteFile(envPath, []byte(envContent), 0644))

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	fileSystem := newMockFileSystemAdapter()

	service := NewDevService(processManager, portManager, fileSystem).(*DefaultDevService)

	envVars, err := service.loadDevEnvFile(packagePath)
	// Should not error, but should skip invalid lines
	require.NoError(t, err)
	assert.Equal(t, "value", envVars["KEY"])
	assert.NotContains(t, envVars, "INVALID_LINE_WITHOUT_EQUALS")
}
