package server

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagejobs"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagemaint"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

type activityStorage struct {
	*stubStorage
	records  map[string][]*types.Execution
	err      error
	calls    []types.ExecutionFilter
	packages map[string]*types.AgentPackage
}

func (s *activityStorage) QueryExecutionRecords(_ context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	s.calls = append(s.calls, filter)
	if s.err != nil {
		return nil, s.err
	}
	if filter.Status == nil {
		return nil, nil
	}
	records := s.records[*filter.Status]
	result := make([]*types.Execution, 0, len(records))
	for _, record := range records {
		if filter.AgentNodeID != nil && (record == nil || record.AgentNodeID != *filter.AgentNodeID) {
			continue
		}
		result = append(result, record)
		if filter.Limit > 0 && len(result) == filter.Limit {
			break
		}
	}
	return result, nil
}

func (s *activityStorage) GetAgentPackage(_ context.Context, id string) (*types.AgentPackage, error) {
	if pkg := s.packages[id]; pkg != nil {
		return pkg, nil
	}
	return nil, errors.New("not found")
}
func (s *activityStorage) StoreAgentPackage(_ context.Context, pkg *types.AgentPackage) error {
	if s.packages == nil {
		s.packages = make(map[string]*types.AgentPackage)
	}
	s.packages[pkg.ID] = pkg
	return nil
}
func (s *activityStorage) UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	return s.StoreAgentPackage(ctx, pkg)
}
func (s *activityStorage) QueryAgentPackages(_ context.Context, _ types.PackageFilters) ([]*types.AgentPackage, error) {
	result := make([]*types.AgentPackage, 0, len(s.packages))
	for _, pkg := range s.packages {
		result = append(result, pkg)
	}
	return result, nil
}

func writeActivityPackage(t *testing.T, home, packageName, nodeID string) {
	t.Helper()
	pkgDir := filepath.Join(home, "packages", packageName)
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := "name: " + packageName + "\nversion: 1.0.0\nagent_node:\n  node_id: " + nodeID + "\n"
	if err := os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	registry := "installed:\n  " + packageName + ":\n    name: " + packageName + "\n    path: " + pkgDir + "\n"
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte(registry), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestPackageExecutionActivityMatchesManifestAndEquivalentNodeIDs(t *testing.T) {
	home := t.TempDir()
	writeActivityPackage(t, home, "install-name", "manifest_node")
	store := &activityStorage{
		stubStorage: newStubStorage(),
		records: map[string][]*types.Execution{
			types.ExecutionStatusRunning: {{AgentNodeID: "unrelated-node"}, {AgentNodeID: "MANIFEST-NODE"}},
		},
	}
	busy, err := (packageExecutionActivity{storage: store, home: home}).HasActiveExecutions(context.Background(), "install-name")
	if err != nil || !busy {
		t.Fatalf("busy=%v err=%v calls=%+v", busy, err, store.calls)
	}
	if len(store.calls) != 3 || store.calls[0].AgentNodeID == nil || store.calls[1].AgentNodeID == nil || store.calls[2].AgentNodeID != nil {
		t.Fatalf("activity query must try exact candidates before equivalence fallback: %+v", store.calls)
	}
	for index, call := range store.calls {
		wantLimit := 1
		if index == len(store.calls)-1 {
			wantLimit = packageExecutionFallbackLimit
		}
		if call.Limit != wantLimit || !call.ExcludePayloads {
			t.Fatalf("activity query limit/payload mismatch: %+v", call)
		}
	}
}

func TestPackageExecutionActivityFailsClosedWhenManifestCannotResolve(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed:\n  demo:\n    name: demo\n    path: /missing/package\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := &activityStorage{stubStorage: newStubStorage()}
	busy, err := (packageExecutionActivity{storage: store, home: home}).HasActiveExecutions(context.Background(), "demo")
	if err == nil || busy || len(store.calls) != 0 {
		t.Fatalf("busy=%v err=%v calls=%+v", busy, err, store.calls)
	}
}

func TestPackageExecutionActivityPropagatesStorageFailure(t *testing.T) {
	home := t.TempDir()
	writeActivityPackage(t, home, "demo", "demo-node")
	store := &activityStorage{stubStorage: newStubStorage(), err: errors.New("storage unavailable")}
	busy, err := (packageExecutionActivity{storage: store, home: home}).HasActiveExecutions(context.Background(), "demo")
	if err == nil || busy {
		t.Fatalf("busy=%v err=%v", busy, err)
	}
}

func TestPackageExecutionActivityReportsIdleAcrossAllActiveStatuses(t *testing.T) {
	home := t.TempDir()
	writeActivityPackage(t, home, "demo", "demo-node")
	store := &activityStorage{stubStorage: newStubStorage(), records: map[string][]*types.Execution{
		types.ExecutionStatusPending: {nil, {AgentNodeID: "another-node"}},
	}}
	busy, err := (packageExecutionActivity{storage: store, home: home}).HasActiveExecutions(context.Background(), "demo")
	if err != nil || busy || len(store.calls) != 12 {
		t.Fatalf("busy=%v err=%v calls=%+v", busy, err, store.calls)
	}
	for _, call := range store.calls {
		if call.AgentNodeID != nil && call.Limit != 1 {
			t.Fatalf("exact activity query was not bounded: %+v", call)
		}
		if call.AgentNodeID == nil && call.Limit != packageExecutionFallbackLimit {
			t.Fatalf("equivalence fallback was incorrectly limited: %+v", call)
		}
	}
}

func TestPackageExecutionActivitySkipsFallbackForExactCandidate(t *testing.T) {
	home := t.TempDir()
	writeActivityPackage(t, home, "demo", "demo")
	store := &activityStorage{stubStorage: newStubStorage()}
	busy, err := (packageExecutionActivity{storage: store, home: home}).HasActiveExecutions(context.Background(), "demo")
	if err != nil || busy || len(store.calls) != 4 {
		t.Fatalf("busy=%v err=%v calls=%+v", busy, err, store.calls)
	}
	for _, call := range store.calls {
		if call.AgentNodeID == nil || call.Limit != 1 {
			t.Fatalf("exact candidate unexpectedly used fallback scan: %+v", call)
		}
	}
}

func TestPackageExecutionActivityCapsNonExactFallbackAtFiveHundred(t *testing.T) {
	home := t.TempDir()
	writeActivityPackage(t, home, "install-name", "manifest_node")
	records := make([]*types.Execution, 0, packageExecutionFallbackLimit+1)
	for index := 0; index < packageExecutionFallbackLimit; index++ {
		records = append(records, &types.Execution{AgentNodeID: "unrelated-node"})
	}
	records = append(records, &types.Execution{AgentNodeID: "MANIFEST-NODE"})
	store := &activityStorage{
		stubStorage: newStubStorage(),
		records: map[string][]*types.Execution{
			types.ExecutionStatusRunning: records,
		},
	}
	busy, err := (packageExecutionActivity{storage: store, home: home}).HasActiveExecutions(context.Background(), "install-name")
	if err != nil || busy {
		t.Fatalf("bounded fallback reached a record after the cap: busy=%v err=%v", busy, err)
	}
	if len(store.calls) < 3 || store.calls[2].Limit != packageExecutionFallbackLimit {
		t.Fatalf("fallback query was not capped: %+v", store.calls)
	}
}

func TestPackageAgentNodeIDsIncludeRegistryAndManifestCandidates(t *testing.T) {
	home := t.TempDir()
	writeActivityPackage(t, home, "install-name", "runtime-node")
	candidates, err := packageAgentNodeIDs(home, "install-name")
	if err != nil || len(candidates) != 2 || candidates[0] != "install-name" || candidates[1] != "runtime-node" {
		t.Fatalf("candidates=%v err=%v", candidates, err)
	}
}

func TestPackageAgentNodeIDResolutionErrorsAreFailClosed(t *testing.T) {
	for _, test := range []struct {
		name    string
		content string
		write   bool
	}{
		{name: "missing registry"},
		{name: "invalid registry", content: "installed: [", write: true},
		{name: "missing package", content: "installed: {}\n", write: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			home := t.TempDir()
			if test.write {
				if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte(test.content), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if candidates, err := packageAgentNodeIDs(home, "demo"); err == nil || candidates != nil {
				t.Fatalf("candidates=%v err=%v", candidates, err)
			}
			if fallback := packageAgentNodeID(home, "demo"); fallback != "demo" {
				t.Fatalf("fallback=%q", fallback)
			}
		})
	}
}

func TestPackageUpdateGraceHookTracksManifestAndRegistryNames(t *testing.T) {
	home := t.TempDir()
	writeActivityPackage(t, home, "install-name", "runtime-node")
	hook := &packageUpdateGraceHook{home: home, nodeIDs: make(map[string]string)}
	hook.set("install-name", true)
	t.Cleanup(func() { hook.set("install-name", false) })
	if hook.nodeIDs["install-name"] != "runtime-node" {
		t.Fatalf("tracked node IDs=%v", hook.nodeIDs)
	}
	hook.set("install-name", false)
	if len(hook.nodeIDs) != 0 {
		t.Fatalf("grace hook did not clear state: %v", hook.nodeIDs)
	}
}

func TestPackageMaintenanceServerAdapterWiresRegistryChangesAndUnattendedJobs(t *testing.T) {
	home := t.TempDir()
	writeActivityPackage(t, home, "demo", "demo-node")
	store := &activityStorage{stubStorage: newStubStorage()}
	jobs := packagejobs.NewManager(nil, home, nil)
	adapter := unattendedPackageJobs{manager: jobs}
	if _, err := adapter.StartUpdate("missing", ""); !errors.Is(err, packagejobs.ErrNotFound) {
		t.Fatalf("unattended update err=%v", err)
	}
	if _, ok := adapter.GetJob("missing"); ok {
		t.Fatal("unknown job unexpectedly exists")
	}
	maintenance := newPackageMaintenance(home, store, nil, jobs)
	if _, err := maintenance.SetAutoUpdate("demo", false); err != nil {
		t.Fatal(err)
	}
	if _, ok := store.packages["demo"]; !ok {
		t.Fatal("registry-change hook did not synchronize the package mirror")
	}
}

func TestServerStopCancelsPackageMaintenanceLifecycle(t *testing.T) {
	lifecycle, cancel := context.WithCancel(context.Background())
	service := packagemaint.New(packagemaint.Config{AgentFieldHome: t.TempDir()})
	service.SetLifecycleContext(lifecycle)
	server := &AgentFieldServer{packageMaintenance: service, packageMaintenanceCancel: cancel}
	if err := server.Stop(); err != nil {
		t.Fatal(err)
	}
	if lifecycle.Err() == nil || server.packageMaintenanceCancel != nil {
		t.Fatalf("lifecycle err=%v cancel=%v", lifecycle.Err(), server.packageMaintenanceCancel)
	}
}
