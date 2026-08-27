package services

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// Mock implementations for testing

type mockProcessManager struct {
	startFunc   func(interfaces.ProcessConfig) (int, error)
	stopFunc    func(int) error
	statusFunc  func(int) (interfaces.ProcessInfo, error)
	startedPIDs map[int]bool
	stoppedPIDs map[int]bool
}

func newMockProcessManager() *mockProcessManager {
	return &mockProcessManager{
		startedPIDs: make(map[int]bool),
		stoppedPIDs: make(map[int]bool),
	}
}

func (m *mockProcessManager) Start(config interfaces.ProcessConfig) (int, error) {
	if m.startFunc != nil {
		return m.startFunc(config)
	}
	pid := 12345
	m.startedPIDs[pid] = true
	return pid, nil
}

func (m *mockProcessManager) Stop(pid int) error {
	if m.stopFunc != nil {
		return m.stopFunc(pid)
	}
	m.stoppedPIDs[pid] = true
	return nil
}

func (m *mockProcessManager) Status(pid int) (interfaces.ProcessInfo, error) {
	if m.statusFunc != nil {
		return m.statusFunc(pid)
	}
	if m.startedPIDs[pid] && !m.stoppedPIDs[pid] {
		return interfaces.ProcessInfo{
			PID:     pid,
			Status:  "running",
			Command: "python",
		}, nil
	}
	return interfaces.ProcessInfo{}, errors.New("process not found")
}

type mockPortManager struct {
	findFreePortFunc func(int) (int, error)
	isAvailableFunc  func(int) bool
	reserveFunc      func(int) error
	releaseFunc      func(int) error
	availablePorts   map[int]bool
}

func newMockPortManager() *mockPortManager {
	return &mockPortManager{
		availablePorts: make(map[int]bool),
	}
}

func (m *mockPortManager) FindFreePort(startPort int) (int, error) {
	if m.findFreePortFunc != nil {
		return m.findFreePortFunc(startPort)
	}
	// Default: return startPort if available
	if m.availablePorts[startPort] || len(m.availablePorts) == 0 {
		m.availablePorts[startPort] = true
		return startPort, nil
	}
	return 0, errors.New("no free port available")
}

func (m *mockPortManager) IsPortAvailable(port int) bool {
	if m.isAvailableFunc != nil {
		return m.isAvailableFunc(port)
	}
	return m.availablePorts[port] || len(m.availablePorts) == 0
}

func (m *mockPortManager) ReservePort(port int) error {
	if m.reserveFunc != nil {
		return m.reserveFunc(port)
	}
	m.availablePorts[port] = true
	return nil
}

func (m *mockPortManager) ReleasePort(port int) error {
	if m.releaseFunc != nil {
		return m.releaseFunc(port)
	}
	delete(m.availablePorts, port)
	return nil
}

type mockRegistryStorage struct {
	loadRegistryFunc func() (*domain.InstallationRegistry, error)
	saveRegistryFunc func(*domain.InstallationRegistry) error
	getPackageFunc   func(string) (*domain.InstalledPackage, error)
	savePackageFunc  func(string, *domain.InstalledPackage) error
	registry         *domain.InstallationRegistry
}

func newMockRegistryStorage() *mockRegistryStorage {
	return &mockRegistryStorage{
		registry: &domain.InstallationRegistry{
			Installed: make(map[string]domain.InstalledPackage),
		},
	}
}

func (m *mockRegistryStorage) LoadRegistry() (*domain.InstallationRegistry, error) {
	if m.loadRegistryFunc != nil {
		return m.loadRegistryFunc()
	}
	return m.registry, nil
}

func (m *mockRegistryStorage) SaveRegistry(registry *domain.InstallationRegistry) error {
	if m.saveRegistryFunc != nil {
		return m.saveRegistryFunc(registry)
	}
	m.registry = registry
	return nil
}

func (m *mockRegistryStorage) GetPackage(name string) (*domain.InstalledPackage, error) {
	if m.getPackageFunc != nil {
		return m.getPackageFunc(name)
	}
	if pkg, ok := m.registry.Installed[name]; ok {
		return &pkg, nil
	}
	return nil, errors.New("package not found")
}

func (m *mockRegistryStorage) SavePackage(name string, pkg *domain.InstalledPackage) error {
	if m.savePackageFunc != nil {
		return m.savePackageFunc(name, pkg)
	}
	if m.registry.Installed == nil {
		m.registry.Installed = make(map[string]domain.InstalledPackage)
	}
	m.registry.Installed[name] = *pkg
	return nil
}

type mockAgentClient struct {
	shutdownFunc func(context.Context, string, bool, int) (*interfaces.AgentShutdownResponse, error)
}

func newMockAgentClient() *mockAgentClient {
	return &mockAgentClient{}
}

func (m *mockAgentClient) ShutdownAgent(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*interfaces.AgentShutdownResponse, error) {
	if m.shutdownFunc != nil {
		return m.shutdownFunc(ctx, nodeID, graceful, timeoutSeconds)
	}
	return &interfaces.AgentShutdownResponse{
		Status:   "shutting_down",
		Graceful: graceful,
		Message:  "Shutdown requested",
	}, nil
}

func (m *mockAgentClient) GetAgentStatus(ctx context.Context, nodeID string) (*interfaces.AgentStatusResponse, error) {
	return nil, errors.New("not implemented")
}

// Helper function to create a test registry file
func createTestRegistry(t *testing.T, dir string, registry *packages.InstallationRegistry) string {
	registryPath := filepath.Join(dir, "installed.yaml")
	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	err = os.WriteFile(registryPath, data, 0644)
	require.NoError(t, err)
	return registryPath
}

func TestNewAgentService(t *testing.T) {
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()
	agentfieldHome := "/tmp/test-agentfield"

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	)

	assert.NotNil(t, service)
	as, ok := service.(*DefaultAgentService)
	require.True(t, ok)
	assert.Equal(t, processManager, as.processManager)
	assert.Equal(t, portManager, as.portManager)
	assert.Equal(t, registryStorage, as.registryStorage)
	assert.Equal(t, agentClient, as.agentClient)
	assert.Equal(t, agentfieldHome, as.agentfieldHome)
}

func TestRunAgent_Success(t *testing.T) {
	// Setup
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	// Create test registry with an installed agent
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-agent": {
				Name:    "test-agent",
				Version: "1.0.0",
				Path:    "/tmp/test-agent-path",
				Status:  "stopped",
				Runtime: packages.RuntimeInfo{
					Port:      nil,
					PID:       nil,
					StartedAt: nil,
					LogFile:   "/tmp/test-agent.log",
				},
			},
		},
	}
	createTestRegistry(t, agentfieldHome, registry)

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	).(*DefaultAgentService)

	// Mock port manager to return a port nothing listens on: the readiness
	// wait probes it for real, so a fixed 8001 breaks whenever a node is
	// serving there on the developer machine.
	portManager.findFreePortFunc = func(startPort int) (int, error) {
		return closedPort(t), nil
	}

	// Mock process manager to start successfully
	processManager.startFunc = func(config interfaces.ProcessConfig) (int, error) {
		// Don't assert on exact command since it may be python3 or system python with full path
		assert.True(t, config.Command == "python" || config.Command == "python3" ||
			strings.Contains(config.Command, "python3"),
			"Expected python command, got: %s", config.Command)
		assert.Equal(t, []string{"main.py"}, config.Args)
		return 12345, nil
	}

	options := domain.RunOptions{
		Port:   0, // Let it find a free port
		Detach: false,
	}

	// This will fail at waitForAgentNode since we can't easily mock HTTP client
	// The test verifies the earlier steps (registry loading, port allocation, process start) work
	_, err := service.RunAgent("test-agent", options)
	// We expect it to fail at waitForAgentNode since we can't easily mock HTTP
	// But we can verify the earlier steps worked
	if err != nil {
		// Verify the error is about waiting for agent (not about registry, port, etc.)
		assert.Contains(t, err.Error(), "agent node did not become ready")
	} else {
		// If it somehow succeeded, that's also fine - it means all steps worked
		// This can happen if there's actually a process running on the test port
		t.Logf("RunAgent succeeded (unexpected but acceptable)")
	}
}

func TestRunAgent_AgentNotInstalled(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	).(*DefaultAgentService)

	options := domain.RunOptions{Port: 0}

	_, err := service.RunAgent("nonexistent-agent", options)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not installed")
}

func TestRunAgent_AlreadyRunning(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	port := findFreePortInRange(t)
	pid := os.Getpid()
	startedAt := time.Now().Format(time.RFC3339)
	startTime := packages.CurrentProcessStartTime(pid)

	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-agent": {
				Name:    "test-agent",
				Version: "1.0.0",
				Path:    "/tmp/test-agent-path",
				Status:  "running",
				Runtime: packages.RuntimeInfo{
					Port:      &port,
					PID:       &pid,
					StartedAt: &startedAt,
					StartTime: startTime,
					LogFile:   "/tmp/test-agent.log",
				},
			},
		},
	}
	createTestRegistry(t, agentfieldHome, registry)

	processManager := newMockProcessManager()
	// Mock process manager to report process as running
	processManager.statusFunc = func(pid int) (interfaces.ProcessInfo, error) {
		return interfaces.ProcessInfo{
			PID:     pid,
			Status:  "running",
			Command: "python",
		}, nil
	}
	processManager.startedPIDs[pid] = true

	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	).(*DefaultAgentService)

	options := domain.RunOptions{Port: 0}

	_, err := service.RunAgent("test-agent", options)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already running")
}

func TestStopAgent_Success(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	port := closedPort(t)
	pid := 1<<22 - 1 // above pid_max: guaranteed dead, never a real process this test could signal
	startedAt := time.Now().Format(time.RFC3339)

	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-agent": {
				Name:    "test-agent",
				Version: "1.0.0",
				Path:    "/tmp/test-agent-path",
				Status:  "running",
				Runtime: packages.RuntimeInfo{
					Port:      &port,
					PID:       &pid,
					StartedAt: &startedAt,
					LogFile:   "/tmp/test-agent.log",
				},
			},
		},
	}
	createTestRegistry(t, agentfieldHome, registry)

	processManager := newMockProcessManager()
	processManager.statusFunc = func(pid int) (interfaces.ProcessInfo, error) {
		return interfaces.ProcessInfo{
			PID:     pid,
			Status:  "running",
			Command: "python",
		}, nil
	}
	processManager.startedPIDs[pid] = true

	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	// Mock successful HTTP shutdown
	agentClient.shutdownFunc = func(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*interfaces.AgentShutdownResponse, error) {
		return &interfaces.AgentShutdownResponse{
			Status:   "shutting_down",
			Graceful: graceful,
			Message:  "Shutdown requested",
		}, nil
	}

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	).(*DefaultAgentService)

	err := service.StopAgent("test-agent")
	assert.NoError(t, err, "stopping an already-dead installed agent is idempotent")
}

func TestE3C1RunAgentReplacesARecycledRecordedPIDWithOrWithoutStartTime(t *testing.T) {
	for _, test := range []struct {
		name      string
		startTime string
	}{
		{name: "mismatched identity", startTime: "a-different-process"},
		{name: "legacy unknown identity"},
	} {
		t.Run(test.name, func(t *testing.T) {
			home := t.TempDir()
			pkgDir := filepath.Join(home, "packages", "demo")
			writeManifest(t, pkgDir, "name: demo\nversion: 1.0.0\nagent_node:\n  node_id: demo\n")
			stalePID := os.Getpid()
			var startedAt *string
			if test.startTime == "" {
				processStarted, ok := packages.ProcessStartWallClock(stalePID)
				require.True(t, ok)
				recorded := processStarted.Add(-time.Minute).Format(time.RFC3339)
				startedAt = &recorded
			}
			stalePort := findFreePortInRange(t)
			createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
				"demo": {
					Name: "demo", Path: pkgDir, Status: "running", DesiredState: packages.DesiredStateRunning,
					Runtime: packages.RuntimeInfo{PID: &stalePID, Port: &stalePort, StartTime: test.startTime, StartedAt: startedAt},
				},
			}})

			server, newPort := startLocalServerOnFreePort(t, contractNodeHandler("demo"))
			defer server.Close()
			processManager := newMockProcessManager()
			starts := 0
			var started *exec.Cmd
			processManager.startFunc = func(interfaces.ProcessConfig) (int, error) {
				starts++
				started = exec.Command("sleep", "60")
				if err := started.Start(); err != nil {
					return 0, err
				}
				return started.Process.Pid, nil
			}
			t.Cleanup(func() {
				if started != nil && started.Process != nil {
					_ = started.Process.Kill()
					_, _ = started.Process.Wait()
				}
			})
			ports := newMockPortManager()
			ports.findFreePortFunc = func(int) (int, error) { return newPort, nil }
			service := NewAgentService(processManager, ports, newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)

			running, err := service.RunAgent("demo", domain.RunOptions{})
			require.NoError(t, err)
			require.Equal(t, 1, starts)
			require.Equal(t, newPort, running.Port)
			require.NotEqual(t, stalePort, running.Port)
			registry, err := service.loadRegistryDirect()
			require.NoError(t, err)
			entry := registry.Installed["demo"]
			require.NotNil(t, entry.Runtime.PID)
			require.Equal(t, running.PID, *entry.Runtime.PID)
			require.NotEmpty(t, entry.Runtime.StartTime)
		})
	}
}

func TestC2RunAgentKeepsAnEquivalentOrAnonymousHealthyNode(t *testing.T) {
	for _, nodeID := range []string{"demo_node", ""} {
		t.Run("node_id="+nodeID, func(t *testing.T) {
			home := t.TempDir()
			pkgDir := filepath.Join(home, "packages", "demo-node")
			writeManifest(t, pkgDir, "name: demo-node\nversion: 1.0.0\nagent_node:\n  node_id: demo-node\n")
			server, port := startLocalServerOnFreePort(t, contractNodeHandler(nodeID))
			defer server.Close()
			pid := os.Getpid()
			createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
				"demo-node": {
					Name: "demo-node", Path: pkgDir, Status: "running", DesiredState: packages.DesiredStateRunning,
					Runtime: packages.RuntimeInfo{PID: &pid, Port: &port, StartTime: "a-different-process"},
				},
			}})
			processManager := newMockProcessManager()
			starts := 0
			processManager.startFunc = func(interfaces.ProcessConfig) (int, error) { starts++; return 0, nil }
			service := NewAgentService(processManager, newMockPortManager(), newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)

			_, err := service.RunAgent("demo-node", domain.RunOptions{})
			require.Error(t, err)
			require.Contains(t, err.Error(), "already running")
			require.Zero(t, starts)
		})
	}
}

func TestC3RunAgentAvoidsAHealthyForeignNodeAndNeverSignalsItsPID(t *testing.T) {
	home := t.TempDir()
	pkgDir := filepath.Join(home, "packages", "demo")
	writeManifest(t, pkgDir, "name: demo\nversion: 1.0.0\nagent_node:\n  node_id: demo\n")
	foreignServer, foreignPort := startLocalServerOnFreePort(t, contractNodeHandler("somebody-else"))
	defer foreignServer.Close()
	recordedPID := os.Getpid()
	createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {
			Name: "demo", Path: pkgDir, Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{PID: &recordedPID, Port: &foreignPort, StartTime: "a-different-process"},
		},
	}})
	ownServer, ownPort := startLocalServerOnFreePort(t, contractNodeHandler("demo"))
	defer ownServer.Close()
	processManager := newMockProcessManager()
	var started *exec.Cmd
	processManager.startFunc = func(interfaces.ProcessConfig) (int, error) {
		started = exec.Command("sleep", "60")
		if err := started.Start(); err != nil {
			return 0, err
		}
		return started.Process.Pid, nil
	}
	t.Cleanup(func() {
		if started != nil && started.Process != nil {
			_ = started.Process.Kill()
			_, _ = started.Process.Wait()
		}
	})
	ports := newMockPortManager()
	ports.findFreePortFunc = func(int) (int, error) { return ownPort, nil }
	service := NewAgentService(processManager, ports, newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)

	running, err := service.RunAgent("demo", domain.RunOptions{})
	require.NoError(t, err)
	require.Equal(t, ownPort, running.Port)
	require.NotEqual(t, foreignPort, running.Port)
	require.True(t, packages.RuntimePIDAlive(packages.RuntimeInfo{PID: &recordedPID}))
}

func TestC4GetAgentStatusKeepsALiveOwnedButUnhealthyProcessRunningOnLinux(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("C4 is the Linux read-path contract")
	}
	home := t.TempDir()
	cmd := exec.Command("sleep", "60")
	require.NoError(t, cmd.Start())
	t.Cleanup(func() { _ = cmd.Process.Kill(); _, _ = cmd.Process.Wait() })
	pid := cmd.Process.Pid
	port := findFreePortInRange(t)
	createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {
			Name: "demo", Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{PID: &pid, Port: &port, StartTime: packages.CurrentProcessStartTime(pid)},
		},
	}})
	service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), newMockAgentClient(), home).(*DefaultAgentService)
	status, err := service.GetAgentStatus("demo")
	require.NoError(t, err)
	require.True(t, status.IsRunning)
	require.Equal(t, pid, status.PID)
}

func contractNodeHandler(nodeID string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/health":
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"node_id":%q}`, nodeID)
		case "/reasoners":
			_, _ = w.Write([]byte(`{"reasoners":[]}`))
		case "/skills":
			_, _ = w.Write([]byte(`{"skills":[]}`))
		default:
			http.NotFound(w, request)
		}
	})
}

func TestC5DarwinReadPathUsesCheapPIDProbe(t *testing.T) {
	service := newStartupTestService(t, newMockPortManager())
	pid := os.Getpid()
	port := 8123
	pkg := &packages.InstalledPackage{
		Name: "recycled", Status: "running",
		Runtime: packages.RuntimeInfo{PID: &pid, Port: &port, StartTime: "different-process"},
	}
	alive, reconciled := service.reconcileProcessStateWithProbes(
		pkg, "recycled", "darwin", false, make(map[int]packages.HealthIdentity),
		packages.RuntimeProcessStatus,
		func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} },
	)
	if !alive || reconciled {
		t.Fatalf("alive=%v reconciled=%v, want a cheap live projection without identity reconciliation", alive, reconciled)
	}
	if pkg.Status != "running" || pkg.DesiredState != packages.DesiredStateRunning || pkg.Runtime.PID == nil || pkg.Runtime.StartTime != "different-process" {
		t.Fatalf("status projection performed an expensive identity reconciliation: %+v", pkg)
	}
}

func TestWindowsProcessReconciliationUsesMemoizedHealthIdentityForEveryEntry(t *testing.T) {
	service := &DefaultAgentService{}
	port, pid := 8123, 42
	memo := make(map[int]packages.HealthIdentity)
	probeCalls := 0
	probe := func(context.Context, int, string) packages.HealthIdentity {
		probeCalls++
		return packages.HealthIdentity{Healthy: true, NodeID: "demo_node"}
	}
	for iteration := 0; iteration < 2; iteration++ {
		pkg := packages.InstalledPackage{
			Name: "demo-node", Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{Port: &port, PID: &pid, StartTime: "recorded-even-though-windows-status-must-probe"},
		}
		running, reconciled := service.reconcileProcessStateWithProbe(&pkg, "demo-node", "windows", memo, probe)
		if !running || reconciled || pkg.Status != "running" {
			t.Fatalf("iteration=%d running=%v reconciled=%v pkg=%+v", iteration, running, reconciled, pkg)
		}
	}
	if probeCalls != 1 {
		t.Fatalf("health probe calls=%d, want one per memoized status read", probeCalls)
	}

	for _, test := range []struct {
		name           string
		identity       packages.HealthIdentity
		wantRunning    bool
		wantReconciled bool
	}{
		{name: "silent port", identity: packages.HealthIdentity{}, wantReconciled: true},
		{name: "anonymous Go health", identity: packages.HealthIdentity{Healthy: true}, wantRunning: true},
		{name: "foreign node", identity: packages.HealthIdentity{Healthy: true, NodeID: "somebody-else"}, wantReconciled: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			missingPID := 99999999
			pkg := packages.InstalledPackage{
				Name: "demo-node", Status: "running", DesiredState: packages.DesiredStateRunning,
				Runtime: packages.RuntimeInfo{Port: &port, PID: &missingPID, StartTime: "recorded"},
			}
			running, reconciled := service.reconcileProcessStateWithProbe(
				&pkg, "demo-node", "windows", make(map[int]packages.HealthIdentity),
				func(context.Context, int, string) packages.HealthIdentity { return test.identity },
			)
			if running != test.wantRunning || reconciled != test.wantReconciled {
				t.Fatalf("identity=%+v running=%v reconciled=%v pkg=%+v", test.identity, running, reconciled, pkg)
			}
			if test.wantRunning {
				if pkg.Status != "running" || pkg.Runtime.PID == nil {
					t.Fatalf("anonymous healthy Windows status cleared runtime: %+v", pkg)
				}
			} else if pkg.Status != "stopped" || pkg.Runtime.PID != nil {
				t.Fatalf("dead/foreign Windows status was not reconciled: %+v", pkg)
			}
		})
	}
}

func TestStopAgentForUpdateStopsLegacyGoNodeWithAnonymousHealthBySignal(t *testing.T) {
	home := t.TempDir()
	pkgDir := filepath.Join(home, "packages", "demo")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte("name: demo\nversion: 1.0.0\nlanguage: go\nagent_node:\n  node_id: demo\n"), 0o600))
	cmd := exec.Command("sh", "-c", "trap 'exit 0' INT; while :; do sleep 1; done")
	require.NoError(t, cmd.Start())
	done := make(chan struct{})
	go func() { _, _ = cmd.Process.Wait(); close(done) }()
	t.Cleanup(func() {
		select {
		case <-done:
		default:
			_ = cmd.Process.Kill()
			<-done
		}
	})
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	var shutdowns atomic.Int32
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/health" {
			_, _ = w.Write([]byte(`{"status":"ok"}`))
			return
		}
		shutdowns.Add(1)
		http.NotFound(w, request)
	})}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })
	pid := cmd.Process.Pid
	createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {
			Name: "demo", Path: pkgDir, Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{Port: &port, PID: &pid},
		},
	}})
	service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
	require.NoError(t, service.StopAgentForUpdate("demo"))
	require.Equal(t, int32(0), shutdowns.Load())
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Go node was not interrupted after its unsupported shutdown route was skipped")
	}
	registry, err := service.loadRegistryDirect()
	require.NoError(t, err)
	entry := registry.Installed["demo"]
	require.Equal(t, "stopped", entry.Status)
	require.Equal(t, packages.DesiredStateRunning, entry.DesiredState)
	require.Nil(t, entry.Runtime.PID)
	require.NotNil(t, entry.Runtime.Port)
}

func TestStopAgentOwnershipAcrossExplicitAndUpdateStops(t *testing.T) {
	for _, test := range []struct {
		name      string
		stop      func(*DefaultAgentService, string) error
		wantState string
		keepPort  bool
	}{
		{name: "explicit stop", stop: func(service *DefaultAgentService, name string) error { return service.StopAgent(name) }, wantState: packages.DesiredStateStopped},
		{name: "update stop", stop: func(service *DefaultAgentService, name string) error { return service.StopAgentForUpdate(name) }, wantState: packages.DesiredStateRunning, keepPort: true},
	} {
		t.Run(test.name+" accepts equivalent node id and uses graceful HTTP", func(t *testing.T) {
			home := t.TempDir()
			cmd := exec.Command("sleep", "30")
			require.NoError(t, cmd.Start())
			done := make(chan struct{})
			go func() { _, _ = cmd.Process.Wait(); close(done) }()
			t.Cleanup(func() {
				select {
				case <-done:
				default:
					_ = cmd.Process.Kill()
					<-done
				}
			})

			listener, err := net.Listen("tcp", "127.0.0.1:0")
			require.NoError(t, err)
			port := listener.Addr().(*net.TCPAddr).Port
			var shutdowns atomic.Int32
			server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
				if request.URL.Path == "/health" {
					_, _ = w.Write([]byte(`{"node_id":"DEMO_NODE"}`))
					return
				}
				shutdowns.Add(1)
				_ = cmd.Process.Kill()
				w.WriteHeader(http.StatusOK)
			})}
			go func() { _ = server.Serve(listener) }()
			t.Cleanup(func() { _ = server.Close() })

			pid := cmd.Process.Pid
			createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
				"demo-node": {
					Name: "demo-node", Status: "running", DesiredState: packages.DesiredStateRunning,
					Runtime: packages.RuntimeInfo{Port: &port, PID: &pid, StartTime: packages.CurrentProcessStartTime(pid)},
				},
			}})
			service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
			require.NoError(t, test.stop(service, "demo-node"))
			require.Equal(t, int32(1), shutdowns.Load())
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("graceful HTTP path did not stop equivalent node")
			}
			registry, err := service.loadRegistryDirect()
			require.NoError(t, err)
			entry := registry.Installed["demo-node"]
			require.Equal(t, "stopped", entry.Status)
			require.Equal(t, test.wantState, entry.DesiredState)
			if test.keepPort {
				require.NotNil(t, entry.Runtime.Port)
			} else {
				require.Nil(t, entry.Runtime.Port)
			}
		})

		t.Run(test.name+" rejects foreign port without signalling PID", func(t *testing.T) {
			home := t.TempDir()
			cmd := exec.Command("sleep", "30")
			require.NoError(t, cmd.Start())
			t.Cleanup(func() {
				_ = cmd.Process.Kill()
				_, _ = cmd.Process.Wait()
			})
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			require.NoError(t, err)
			port := listener.Addr().(*net.TCPAddr).Port
			var shutdowns atomic.Int32
			server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
				if request.URL.Path == "/health" {
					_, _ = w.Write([]byte(`{"node_id":"somebody-else"}`))
					return
				}
				shutdowns.Add(1)
			})}
			go func() { _ = server.Serve(listener) }()
			t.Cleanup(func() { _ = server.Close() })

			pid := cmd.Process.Pid
			createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
				"demo-node": {
					Name: "demo-node", Status: "running", DesiredState: packages.DesiredStateRunning,
					Runtime: packages.RuntimeInfo{Port: &port, PID: &pid, StartTime: packages.CurrentProcessStartTime(pid)},
				},
			}})
			service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
			require.NoError(t, test.stop(service, "demo-node"))
			require.Zero(t, shutdowns.Load())
			require.True(t, packages.RuntimePIDAlive(packages.RuntimeInfo{PID: &pid}), "foreign port caused recorded PID to be signalled")
			registry, err := service.loadRegistryDirect()
			require.NoError(t, err)
			entry := registry.Installed["demo-node"]
			require.Equal(t, "stopped", entry.Status)
			require.Equal(t, test.wantState, entry.DesiredState)
		})
	}
}

func TestGetAgentStatusHidesPreservedPortWhenStopped(t *testing.T) {
	home := t.TempDir()
	port := 8123
	createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateRunning, Runtime: packages.RuntimeInfo{Port: &port}},
	}})
	service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
	status, err := service.GetAgentStatus("demo")
	if err != nil {
		t.Fatal(err)
	}
	if status.IsRunning || status.Port != 0 {
		t.Fatalf("status = %+v, want stopped with zero projected port", status)
	}
	registry, err := service.loadRegistryDirect()
	if err != nil || registry.Installed["demo"].Runtime.Port == nil || *registry.Installed["demo"].Runtime.Port != port {
		t.Fatalf("preferred registry port was not retained: registry=%+v err=%v", registry, err)
	}
}

func TestUpdateRuntimeInfoRecordsRunningIntentAndProcessIdentity(t *testing.T) {
	home := t.TempDir()
	createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {Name: "demo"},
	}})
	service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
	if err := service.updateRuntimeInfo("demo", 8123, os.Getpid()); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var registry packages.InstallationRegistry
	if err := yaml.Unmarshal(data, &registry); err != nil {
		t.Fatal(err)
	}
	if registry.Installed["demo"].Runtime.BootID != packages.CurrentBootID() {
		t.Fatalf("boot_id = %q, want %q", registry.Installed["demo"].Runtime.BootID, packages.CurrentBootID())
	}
	if registry.Installed["demo"].DesiredState != packages.DesiredStateRunning {
		t.Fatalf("desired_state = %q, want running", registry.Installed["demo"].DesiredState)
	}
	if got := registry.Installed["demo"].Runtime.StartTime; got == "" || got != packages.CurrentProcessStartTime(os.Getpid()) {
		t.Fatalf("start_time = %q", got)
	}
}

func TestExplicitStopPersistsStoppedIntentForDeadProcess(t *testing.T) {
	home := t.TempDir()
	pid := 99999999
	port := 8123
	createTestRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "running", DesiredState: packages.DesiredStateRunning, Runtime: packages.RuntimeInfo{PID: &pid, Port: &port, StartTime: "gone"}},
	}})
	service := NewAgentService(newMockProcessManager(), newMockPortManager(), newMockRegistryStorage(), nil, home).(*DefaultAgentService)
	_ = service.StopAgent("demo")
	data, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var registry packages.InstallationRegistry
	if err := yaml.Unmarshal(data, &registry); err != nil {
		t.Fatal(err)
	}
	entry := registry.Installed["demo"]
	if entry.DesiredState != packages.DesiredStateStopped {
		t.Fatalf("desired_state = %q, want stopped", entry.DesiredState)
	}
}

func TestStopAgent_NotInstalled(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	).(*DefaultAgentService)

	err := service.StopAgent("nonexistent-agent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not installed")
}

func TestGetAgentStatus_Success(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	port := closedPort(t)
	pid := 12345
	startedAt := time.Now().Format(time.RFC3339)

	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-agent": {
				Name:    "test-agent",
				Version: "1.0.0",
				Path:    "/tmp/test-agent-path",
				Status:  "running",
				Runtime: packages.RuntimeInfo{
					Port:      &port,
					PID:       &pid,
					StartedAt: &startedAt,
					LogFile:   "/tmp/test-agent.log",
				},
			},
		},
	}
	createTestRegistry(t, agentfieldHome, registry)

	processManager := newMockProcessManager()
	processManager.statusFunc = func(pid int) (interfaces.ProcessInfo, error) {
		return interfaces.ProcessInfo{
			PID:     pid,
			Status:  "running",
			Command: "python",
		}, nil
	}
	processManager.startedPIDs[pid] = true

	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	).(*DefaultAgentService)

	status, err := service.GetAgentStatus("test-agent")
	require.NoError(t, err)
	assert.Equal(t, "test-agent", status.Name)
	// Since PID 12345 doesn't exist, reconcileProcessState will mark it as stopped
	// The test verifies the agent is found in registry and basic fields are populated
	assert.False(t, status.IsRunning) // Process doesn't actually exist
	assert.Equal(t, 0, status.Port)   // Stopped status never advertises a live endpoint
	assert.Equal(t, 0, status.PID)    // Cleared by reconciliation
	refreshed, loadErr := service.loadRegistryDirect()
	require.NoError(t, loadErr)
	require.NotNil(t, refreshed.Installed["test-agent"].Runtime.Port)
	assert.Equal(t, port, *refreshed.Installed["test-agent"].Runtime.Port) // Restore preference is retained privately.
}

func TestGetAgentStatus_NotInstalled(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	).(*DefaultAgentService)

	_, err := service.GetAgentStatus("nonexistent-agent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not installed")
}

func TestReconcileProcessState_ProcessNotRunning(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	port := closedPort(t)
	pid := 12345
	startedAt := time.Now().Format(time.RFC3339)

	pkg := &packages.InstalledPackage{
		Name:    "test-agent",
		Version: "1.0.0",
		Path:    "/tmp/test-agent-path",
		Status:  "running",
		Runtime: packages.RuntimeInfo{
			Port:      &port,
			PID:       &pid,
			StartedAt: &startedAt,
			LogFile:   "/tmp/test-agent.log",
		},
	}

	processManager := newMockProcessManager()
	// Mock process not found
	processManager.statusFunc = func(pid int) (interfaces.ProcessInfo, error) {
		return interfaces.ProcessInfo{}, errors.New("process not found")
	}

	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	).(*DefaultAgentService)

	actuallyRunning, wasReconciled := service.reconcileProcessState(pkg, "test-agent")
	assert.False(t, actuallyRunning)
	assert.True(t, wasReconciled)
	assert.Equal(t, "stopped", pkg.Status)
	assert.Nil(t, pkg.Runtime.PID)
	require.NotNil(t, pkg.Runtime.Port)
	assert.Equal(t, port, *pkg.Runtime.Port)
	assert.Equal(t, packages.DesiredStateRunning, pkg.DesiredState)
}

func TestReconcileProcessState_ProcessRunning(t *testing.T) {
	port := closedPort(t)
	pid := 12345
	startedAt := time.Now().Format(time.RFC3339)

	pkg := &packages.InstalledPackage{
		Name:    "test-agent",
		Version: "1.0.0",
		Path:    "/tmp/test-agent-path",
		Status:  "running",
		Runtime: packages.RuntimeInfo{
			Port:      &port,
			PID:       &pid,
			StartedAt: &startedAt,
			LogFile:   "/tmp/test-agent.log",
		},
	}

	tmpDir := t.TempDir()
	processManager := newMockProcessManager()
	processManager.statusFunc = func(pid int) (interfaces.ProcessInfo, error) {
		return interfaces.ProcessInfo{
			PID:     pid,
			Status:  "running",
			Command: "python",
		}, nil
	}
	processManager.startedPIDs[pid] = true

	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		tmpDir,
	).(*DefaultAgentService)

	actuallyRunning, wasReconciled := service.reconcileProcessState(pkg, "test-agent")
	// Since PID 12345 doesn't exist, reconciliation will mark it as stopped
	assert.False(t, actuallyRunning)
	assert.True(t, wasReconciled)
	assert.Equal(t, "stopped", pkg.Status)
}

func TestReconcileProcessState_NoPID(t *testing.T) {
	pkg := &packages.InstalledPackage{
		Name:    "test-agent",
		Version: "1.0.0",
		Path:    "/tmp/test-agent-path",
		Status:  "running",
		Runtime: packages.RuntimeInfo{
			Port:      nil,
			PID:       nil,
			StartedAt: nil,
			LogFile:   "/tmp/test-agent.log",
		},
	}

	tmpDir := t.TempDir()
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		tmpDir,
	).(*DefaultAgentService)

	actuallyRunning, wasReconciled := service.reconcileProcessState(pkg, "test-agent")
	assert.False(t, actuallyRunning)
	assert.True(t, wasReconciled)
	assert.Equal(t, "stopped", pkg.Status)
}

func TestListRunningAgents(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	port1 := 8001
	pid1 := 12345
	port2 := 8002
	pid2 := 12346
	startedAt := time.Now().Format(time.RFC3339)

	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-agent-1": {
				Name:    "test-agent-1",
				Version: "1.0.0",
				Path:    "/tmp/test-agent-1-path",
				Status:  "running",
				Runtime: packages.RuntimeInfo{
					Port:      &port1,
					PID:       &pid1,
					StartedAt: &startedAt,
					LogFile:   "/tmp/test-agent-1.log",
				},
			},
			"test-agent-2": {
				Name:    "test-agent-2",
				Version: "1.0.0",
				Path:    "/tmp/test-agent-2-path",
				Status:  "stopped",
				Runtime: packages.RuntimeInfo{
					Port:      &port2,
					PID:       &pid2,
					StartedAt: &startedAt,
					LogFile:   "/tmp/test-agent-2.log",
				},
			},
		},
	}
	createTestRegistry(t, agentfieldHome, registry)

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		agentfieldHome,
	).(*DefaultAgentService)

	runningAgents, err := service.ListRunningAgents()
	require.NoError(t, err)
	assert.Len(t, runningAgents, 1)
	assert.Equal(t, "test-agent-1", runningAgents[0].Name)
}

func TestFindAgentInRegistry_ExactMatch(t *testing.T) {
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-agent": {
				Name:    "test-agent",
				Version: "1.0.0",
			},
		},
	}

	tmpDir := t.TempDir()
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		tmpDir,
	).(*DefaultAgentService)

	pkg, name, exists := service.findAgentInRegistry(registry, "test-agent")
	assert.True(t, exists)
	assert.Equal(t, "test-agent", name)
	assert.Equal(t, "test-agent", pkg.Name)
}

func TestFindAgentInRegistry_NormalizedMatch(t *testing.T) {
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"deep-research-agent": {
				Name:    "deep-research-agent",
				Version: "1.0.0",
			},
		},
	}

	tmpDir := t.TempDir()
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		tmpDir,
	).(*DefaultAgentService)

	pkg, name, exists := service.findAgentInRegistry(registry, "deepresearchagent")
	assert.True(t, exists)
	assert.Equal(t, "deep-research-agent", name)
	assert.Equal(t, "deep-research-agent", pkg.Name)
}

func TestFindAgentInRegistry_NotFound(t *testing.T) {
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{},
	}

	tmpDir := t.TempDir()
	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		tmpDir,
	).(*DefaultAgentService)

	_, _, exists := service.findAgentInRegistry(registry, "nonexistent")
	assert.False(t, exists)
}

func TestBuildProcessConfig(t *testing.T) {
	tmpDir := t.TempDir()
	agentPath := filepath.Join(tmpDir, "agent")
	require.NoError(t, os.MkdirAll(agentPath, 0755))

	agentNode := packages.InstalledPackage{
		Name:    "test-agent",
		Version: "1.0.0",
		Path:    agentPath,
		Runtime: packages.RuntimeInfo{
			LogFile: "/tmp/test-agent.log",
		},
	}

	processManager := newMockProcessManager()
	portManager := newMockPortManager()
	registryStorage := newMockRegistryStorage()
	agentClient := newMockAgentClient()

	service := NewAgentService(
		processManager,
		portManager,
		registryStorage,
		agentClient,
		tmpDir,
	).(*DefaultAgentService)

	config, err := service.buildProcessConfig(agentNode, 8001)
	require.NoError(t, err)
	// Check for any Python command (python, python3, or full path to python3)
	assert.True(t, config.Command == "python" || config.Command == "python3" ||
		strings.Contains(config.Command, "python3"),
		"Expected python command, got: %s", config.Command)
	assert.Equal(t, []string{"main.py"}, config.Args)
	assert.Equal(t, agentPath, config.WorkDir)
	assert.Equal(t, "/tmp/test-agent.log", config.LogFile)
	assert.Contains(t, config.Env, "PORT=8001")
	found := false
	for _, e := range config.Env {
		if strings.HasPrefix(e, "AGENTFIELD_SERVER_URL=") {
			found = true
			break
		}
	}
	assert.True(t, found, "Expected AGENTFIELD_SERVER_URL in env")
}

// closedPort returns a loopback port nothing is listening on. Fixtures that
// record a dead process must not depend on a fixed port being free: the read
// path now probes the recorded port, and an anonymous healthy listener there
// (a Go SDK node, or any dev server on the developer's machine) is legitimately
// treated as ours.
func closedPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	require.NoError(t, listener.Close())
	return port
}

// E24: an explicit start records the intent to run before the node is
// launched, so a container replacement afterwards restores it — regardless
// of what desired_state said before (a fresh install records "stopped").
func TestE24ExplicitStartRecordsRunningIntentBeforeLaunch(t *testing.T) {
	agentfieldHome := t.TempDir()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	node := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"healthy","node_id":"test-agent"}`))
	}))
	node.Listener = listener
	node.Start()
	t.Cleanup(node.Close)

	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-agent": {
				Name:         "test-agent",
				Version:      "1.0.0",
				Path:         t.TempDir(),
				Status:       "stopped",
				DesiredState: packages.DesiredStateStopped,
				Runtime:      packages.RuntimeInfo{LogFile: filepath.Join(agentfieldHome, "test-agent.log")},
			},
		},
	}
	createTestRegistry(t, agentfieldHome, registry)

	processManager := newMockProcessManager()
	processManager.startFunc = func(interfaces.ProcessConfig) (int, error) { return os.Getpid(), nil }
	processManager.startedPIDs[os.Getpid()] = true
	portManager := newMockPortManager()
	portManager.findFreePortFunc = func(int) (int, error) { return port, nil }
	service := NewAgentService(processManager, portManager, newMockRegistryStorage(), newMockAgentClient(), agentfieldHome).(*DefaultAgentService)

	_, err = service.RunAgent("test-agent", domain.RunOptions{Detach: true})
	require.NoError(t, err)

	refreshed, err := service.loadRegistryDirect()
	require.NoError(t, err)
	entry := refreshed.Installed["test-agent"]
	assert.Equal(t, packages.DesiredStateRunning, entry.DesiredState, "an explicit start must record running intent")
	assert.Equal(t, "running", entry.Status)
	require.NotNil(t, entry.Runtime.Port)
	assert.Equal(t, port, *entry.Runtime.Port)
}
