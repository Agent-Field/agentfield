package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages/updatecheck"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagemaint"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

type endpointRunner struct{ sha string }

func (r endpointRunner) Run(context.Context, ...string) ([]byte, error) {
	return []byte(r.sha + "\tHEAD\n"), nil
}

type requestContextRunner struct{ canceled bool }

func (r *requestContextRunner) Run(ctx context.Context, _ ...string) ([]byte, error) {
	r.canceled = ctx.Err() != nil
	return []byte("new\tHEAD\n"), nil
}

func TestC18PackagesListExposesFailedUpdateMessage(t *testing.T) {
	info := packageInfoFromRegistry("demo", packages.InstalledPackage{Name: "demo"}, updatecheck.Update{
		Status: updatecheck.StatusFailed, LatestCommit: "deadbeef", Message: "rename requires a manual update",
	})
	if info.Update.Status != updatecheck.StatusFailed || info.Update.Message != "rename requires a manual update" {
		t.Fatalf("update=%+v", info.Update)
	}
}

type endpointMaintenance struct {
	entries       []updatecheck.Entry
	entriesErr    error
	checker       *updatecheck.Checker
	entry         packages.InstalledPackage
	setErr        error
	status        packagemaint.Status
	started       bool
	startErr      error
	registry      map[string]packages.InstalledPackage
	registryReads int
}

func (m *endpointMaintenance) Entries() ([]updatecheck.Entry, error) { return m.entries, m.entriesErr }
func (m *endpointMaintenance) Checker() *updatecheck.Checker         { return m.checker }
func (m *endpointMaintenance) RegistryEntry(_ string) (packages.InstalledPackage, bool, error) {
	return m.entry, m.entry.Name != "", nil
}
func (m *endpointMaintenance) RegistryEntries() (map[string]packages.InstalledPackage, error) {
	m.registryReads++
	if m.registry != nil {
		return m.registry, nil
	}
	if m.entry.Name != "" {
		return map[string]packages.InstalledPackage{"demo": m.entry}, nil
	}
	return map[string]packages.InstalledPackage{}, nil
}
func (m *endpointMaintenance) SetAutoUpdate(_ string, enabled bool) (packages.InstalledPackage, error) {
	if m.setErr != nil {
		return packages.InstalledPackage{}, m.setErr
	}
	if m.entry.Name == "" {
		return packages.InstalledPackage{}, os.ErrNotExist
	}
	m.entry.AutoUpdate = &enabled
	return m.entry, nil
}
func (m *endpointMaintenance) Status() packagemaint.Status { return m.status }
func (m *endpointMaintenance) StartPass() error {
	if m.startErr != nil {
		return m.startErr
	}
	if !m.started {
		return packagemaint.ErrPassAlreadyRunning
	}
	return nil
}

func TestCheckPackageUpdatesHandlerContract(t *testing.T) {
	checker := updatecheck.NewChecker(endpointRunner{sha: "new"})
	maintenance := &endpointMaintenance{
		checker: checker,
		entries: []updatecheck.Entry{{ID: "demo", Name: "Demo", Source: "https://github.com/acme/demo", InstalledCommit: "old"}},
	}
	ctx, response := testContext(http.MethodPost, "/", nil)
	NewPackageUpdateHandler(maintenance).CheckUpdatesHandler(ctx)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var body struct {
		CheckedAt string `json:"checked_at"`
		Packages  []struct {
			ID              string             `json:"id"`
			Name            string             `json:"name"`
			InstalledCommit string             `json:"installed_commit"`
			Update          updatecheck.Update `json:"update"`
		} `json:"packages"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.CheckedAt == "" || len(body.Packages) != 1 || body.Packages[0].ID != "demo" || body.Packages[0].InstalledCommit != "old" || body.Packages[0].Update.Status != updatecheck.StatusAvailable {
		t.Fatalf("response=%+v", body)
	}
}

func TestCheckPackageUpdatesOutlivesCanceledRequestContext(t *testing.T) {
	runner := &requestContextRunner{}
	maintenance := &endpointMaintenance{
		checker: updatecheck.NewChecker(runner),
		entries: []updatecheck.Entry{{ID: "demo", Source: "https://github.com/acme/demo", InstalledCommit: "old"}},
	}
	ctx, response := testContext(http.MethodPost, "/", nil)
	requestCtx, cancel := context.WithCancel(ctx.Request.Context())
	cancel()
	ctx.Request = ctx.Request.WithContext(requestCtx)
	NewPackageUpdateHandler(maintenance).CheckUpdatesHandler(ctx)
	if response.Code != http.StatusOK || runner.canceled {
		t.Fatalf("status=%d runner saw canceled context=%v body=%s", response.Code, runner.canceled, response.Body.String())
	}
}

func TestCheckPackageUpdatesHandlesRegistryErrorsAndBudgetBounds(t *testing.T) {
	if got := packageUpdateCheckBudget(0); got != 10*time.Second {
		t.Fatalf("empty check budget = %s, want 10s floor", got)
	}
	if got := packageUpdateCheckBudget(13); got != 120*time.Second {
		t.Fatalf("13-entry check budget = %s, want 120s cap", got)
	}
	maintenance := &endpointMaintenance{checker: updatecheck.NewChecker(endpointRunner{sha: "new"}), entriesErr: errors.New("registry unreadable")}
	ctx, response := testContext(http.MethodPost, "/", nil)
	NewPackageUpdateHandler(maintenance).CheckUpdatesHandler(ctx)
	if response.Code != http.StatusInternalServerError {
		t.Fatalf("registry error status=%d body=%s", response.Code, response.Body.String())
	}

	maintenance.entriesErr = nil
	maintenance.entries = nil
	ctx, response = testContext(http.MethodPost, "/", nil)
	NewPackageUpdateHandler(maintenance).CheckUpdatesHandler(ctx)
	if response.Code != http.StatusOK {
		t.Fatalf("empty check status=%d", response.Code)
	}

	maintenance.entries = make([]updatecheck.Entry, 13)
	for index := range maintenance.entries {
		maintenance.entries[index] = updatecheck.Entry{ID: fmt.Sprintf("demo-%d", index), Source: "https://github.com/acme/demo", InstalledCommit: "old"}
	}
	ctx, response = testContext(http.MethodPost, "/", nil)
	NewPackageUpdateHandler(maintenance).CheckUpdatesHandler(ctx)
	if response.Code != http.StatusOK {
		t.Fatalf("bounded large check status=%d", response.Code)
	}
}

func TestSetPackageAutoUpdateHandlerReturnsPackageInfo(t *testing.T) {
	maintenance := &endpointMaintenance{
		checker: updatecheck.NewChecker(endpointRunner{sha: "new"}),
		entry: packages.InstalledPackage{
			Name: "demo", Version: "1.0.0", Status: "running", SourcePath: "https://github.com/acme/demo",
			Commit: "installed", Ref: "", Description: "demo package",
		},
	}
	ctx, response := testContext(http.MethodPut, "/", []byte(`{"enabled":false}`), ginParam("packageId", "demo"))
	NewPackageUpdateHandler(maintenance).SetAutoUpdateHandler(ctx)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var body PackageInfo
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.ID != "demo" || body.InstalledCommit != "installed" || body.AutoUpdate || body.Update.Status != updatecheck.StatusUnknown {
		t.Fatalf("response=%+v", body)
	}
}

func TestSetPackageAutoUpdateValidationAndErrors(t *testing.T) {
	maintenance := &endpointMaintenance{checker: updatecheck.NewChecker(endpointRunner{})}
	handler := NewPackageUpdateHandler(maintenance)
	for _, test := range []struct {
		name   string
		body   string
		setErr error
		status int
	}{
		{name: "missing enabled", body: `{}`, status: http.StatusBadRequest},
		{name: "not found", body: `{"enabled":true}`, setErr: os.ErrNotExist, status: http.StatusNotFound},
		{name: "registry failure", body: `{"enabled":true}`, setErr: errors.New("write failed"), status: http.StatusInternalServerError},
	} {
		t.Run(test.name, func(t *testing.T) {
			maintenance.setErr = test.setErr
			ctx, response := testContext(http.MethodPut, "/", []byte(test.body), ginParam("packageId", "demo"))
			handler.SetAutoUpdateHandler(ctx)
			if response.Code != test.status {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func TestPackageInfoFromRegistryUsesIDFallbackAndRuntime(t *testing.T) {
	port, pid, started := 8123, 42, "2026-08-24T19:40:00Z"
	info := packageInfoFromRegistry("demo", packages.InstalledPackage{
		Status: "running", Runtime: packages.RuntimeInfo{Port: &port, PID: &pid, StartedAt: &started},
	}, updatecheck.Update{})
	if info.Name != "demo" || info.Port != port || info.ProcessID != pid || info.LastStarted != started {
		t.Fatalf("info=%+v", info)
	}
}

func TestStoppedPackageInfoDoesNotAdvertiseRememberedRuntime(t *testing.T) {
	port, pid, started := 8123, 42, "2026-08-24T19:40:00Z"
	entry := packages.InstalledPackage{
		Name: "demo", Status: "stopped",
		Runtime: packages.RuntimeInfo{Port: &port, PID: &pid, StartedAt: &started},
	}
	info := packageInfoFromRegistry("demo", entry, updatecheck.Update{})
	if info.Port != 0 || info.ProcessID != 0 || info.LastStarted != started {
		t.Fatalf("stopped registry projection=%+v", info)
	}

	description := "demo"
	dbPackage := &types.AgentPackage{ID: "demo", Name: "demo", Description: &description, Status: types.PackageStatusStopped}
	store := &overrideStorage{queryAgentPackagesFn: func(context.Context, types.PackageFilters) ([]*types.AgentPackage, error) {
		return []*types.AgentPackage{dbPackage}, nil
	}}
	maintenance := &endpointMaintenance{
		checker:  updatecheck.NewChecker(endpointRunner{}),
		registry: map[string]packages.InstalledPackage{"demo": entry},
	}
	handler := NewPackageHandler(store)
	handler.ConfigurePackageUpdates(maintenance)
	ctx, response := testContext(http.MethodGet, "/", nil)
	handler.ListPackagesHandler(ctx)
	var body PackageListResponse
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusOK || len(body.Packages) != 1 || body.Packages[0].Port != 0 || body.Packages[0].ProcessID != 0 {
		t.Fatalf("status=%d body=%+v", response.Code, body)
	}
}

func TestPackageListAppliesSearchAndStatusBeforeBuildingPackageInfo(t *testing.T) {
	schema := json.RawMessage(`{"required":{"token":{"type":"secret"}}}`)
	rows := []*types.AgentPackage{
		{ID: "match", Name: "Matching package", ConfigurationSchema: schema},
		{ID: "skip", Name: "Unrelated package", ConfigurationSchema: schema},
	}
	configurationReads := map[string]int{}
	store := &overrideStorage{
		queryAgentPackagesFn: func(context.Context, types.PackageFilters) ([]*types.AgentPackage, error) { return rows, nil },
		getAgentConfigurationFn: func(_ context.Context, agentID, _ string) (*types.AgentConfiguration, error) {
			configurationReads[agentID]++
			return nil, errors.New("not configured")
		},
	}
	handler := NewPackageHandler(store)

	searchCtx, response := testContext(http.MethodGet, "/?search=matching", nil)
	handler.ListPackagesHandler(searchCtx)
	if response.Code != http.StatusOK || configurationReads["skip"] != 0 || configurationReads["match"] != 2 {
		t.Fatalf("search status=%d reads=%v body=%s", response.Code, configurationReads, response.Body.String())
	}

	configurationReads = map[string]int{}
	statusCtx, response := testContext(http.MethodGet, "/?status=configured", nil)
	handler.ListPackagesHandler(statusCtx)
	if response.Code != http.StatusOK || len(configurationReads) != 2 || configurationReads["match"] != 1 || configurationReads["skip"] != 1 {
		t.Fatalf("status filter status=%d reads=%v body=%s", response.Code, configurationReads, response.Body.String())
	}
}

func TestSetPackageAutoUpdateReturnsTheSameFullShapeAsList(t *testing.T) {
	description, author, repository := "database description", "Package Author", "database source"
	pkg := &types.AgentPackage{
		ID: "demo", Name: "Demo", Version: "1.0.0", Description: &description, Author: &author,
		Repository: &repository, InstallPath: "/db/path", Status: types.PackageStatusRunning,
		InstalledAt: time.Date(2026, 8, 24, 19, 40, 0, 0, time.UTC),
	}
	store := &overrideStorage{getAgentPackageFn: func(context.Context, string) (*types.AgentPackage, error) {
		return pkg, nil
	}}
	maintenance := &endpointMaintenance{
		checker: updatecheck.NewChecker(endpointRunner{sha: "new"}),
		entry: packages.InstalledPackage{
			Name: "Demo", Version: "1.0.0", Status: "running", SourcePath: "https://github.com/acme/demo",
			Path: "/registry/path", Commit: "installed", Description: "registry description",
		},
	}
	packageHandler := NewPackageHandler(store)
	packageHandler.ConfigurePackageUpdates(maintenance)
	handler := NewPackageUpdateHandler(maintenance, packageHandler)
	ginCtx, response := testContext(http.MethodPut, "/", []byte(`{"enabled":false}`), ginParam("packageId", "demo"))
	handler.SetAutoUpdateHandler(ginCtx)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var body PackageInfo
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Status != "configured" || !body.ConfigurationComplete || body.Author != author || body.InstallPath != "/registry/path" || body.AutoUpdate {
		t.Fatalf("response=%+v", body)
	}
	if body.InstalledAt != "2026-08-24T19:40:00Z" {
		t.Fatalf("empty registry installed_at hid DB value: %+v", body)
	}
}

func TestPackageListUsesOneRegistrySnapshotAndNonEmptyOverlay(t *testing.T) {
	description, source := "database description", "database source"
	packagesInDB := []*types.AgentPackage{
		{ID: "one", Name: "One", Status: types.PackageStatusRunning, InstallPath: "/db/one", Repository: &source, Description: &description},
		{ID: "two", Name: "Two", Status: types.PackageStatusStopped, InstallPath: "/db/two", Repository: &source, Description: &description},
	}
	store := &overrideStorage{queryAgentPackagesFn: func(context.Context, types.PackageFilters) ([]*types.AgentPackage, error) {
		return packagesInDB, nil
	}}
	maintenance := &endpointMaintenance{
		checker: updatecheck.NewChecker(endpointRunner{}),
		registry: map[string]packages.InstalledPackage{
			"one": {Name: "One", InstalledAt: "2026-08-24T15:40:00-04:00", Commit: "abc"},
			"two": {Name: "Two", Status: "stopped", Path: "/registry/two"},
		},
	}
	handler := NewPackageHandler(store)
	handler.ConfigurePackageUpdates(maintenance)
	ctx, response := testContext(http.MethodGet, "/", nil)
	handler.ListPackagesHandler(ctx)
	var body PackageListResponse
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusOK || len(body.Packages) != 2 || maintenance.registryReads != 1 {
		t.Fatalf("status=%d reads=%d body=%+v", response.Code, maintenance.registryReads, body)
	}
	if body.Packages[0].InstallStatus != string(types.PackageStatusRunning) || body.Packages[0].InstallPath != "/db/one" || body.Packages[0].Source != source {
		t.Fatalf("empty registry fields overwrote DB projection: %+v", body.Packages[0])
	}
	if body.Packages[0].InstalledAt != "2026-08-24T19:40:00Z" {
		t.Fatalf("installed_at was not normalized to UTC: %+v", body.Packages[0])
	}
}

func TestPackageListAndDetailsOverlayRegistryUpdateState(t *testing.T) {
	description, repository := "demo description", "stale source"
	pkg := &types.AgentPackage{
		ID: "demo", Name: "Demo", Version: "1.0.0", Description: &description,
		Repository: &repository, InstallPath: "/db/path", Status: types.PackageStatusStopped,
	}
	store := &overrideStorage{
		getAgentPackageFn: func(context.Context, string) (*types.AgentPackage, error) { return pkg, nil },
		queryAgentPackagesFn: func(context.Context, types.PackageFilters) ([]*types.AgentPackage, error) {
			return []*types.AgentPackage{pkg}, nil
		},
	}
	checker := updatecheck.NewChecker(endpointRunner{sha: "new"})
	checker.Set("demo", updatecheck.Update{Status: updatecheck.StatusAvailable, LatestCommit: "new", CheckedAt: time.Now()})
	maintenance := &endpointMaintenance{checker: checker, entry: packages.InstalledPackage{
		Name: "Demo", Version: "1.0.0", Status: "stopped", Path: "/registry/path",
		SourcePath: "https://github.com/acme/demo", Commit: "old", Ref: "", AutoUpdate: func() *bool { value := false; return &value }(),
	}}
	handler := NewPackageHandler(store)
	handler.ConfigurePackageUpdates(maintenance)

	listCtx, listResponse := testContext(http.MethodGet, "/", nil)
	handler.ListPackagesHandler(listCtx)
	var list PackageListResponse
	if err := json.Unmarshal(listResponse.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if listResponse.Code != http.StatusOK || len(list.Packages) != 1 || list.Packages[0].InstalledCommit != "old" || list.Packages[0].Update.Status != updatecheck.StatusAvailable || list.Packages[0].AutoUpdate {
		t.Fatalf("list status=%d response=%+v", listResponse.Code, list)
	}

	detailCtx, detailResponse := testContext(http.MethodGet, "/", nil, ginParam("packageId", "demo"))
	handler.GetPackageDetailsHandler(detailCtx)
	var details PackageDetailsResponse
	if err := json.Unmarshal(detailResponse.Body.Bytes(), &details); err != nil {
		t.Fatal(err)
	}
	if detailResponse.Code != http.StatusOK || details.InstalledCommit != "old" || details.Source != "https://github.com/acme/demo" || details.Update.Status != updatecheck.StatusAvailable || details.AutoUpdate {
		t.Fatalf("details status=%d response=%+v", detailResponse.Code, details)
	}
}

func TestMaintenanceHandlersExposeStatusAndConflict(t *testing.T) {
	maintenance := &endpointMaintenance{
		checker: updatecheck.NewChecker(endpointRunner{}),
		status:  packagemaint.Status{Enabled: false, Reason: "git is not available on PATH", Interval: "6h0m0s"},
		started: false,
	}
	handler := NewPackageUpdateHandler(maintenance)

	ctx, response := testContext(http.MethodGet, "/", nil)
	handler.MaintenanceStatusHandler(ctx)
	if response.Code != http.StatusOK || !containsJSON(response.Body.Bytes(), `"enabled":false`, `"interval":"6h0m0s"`) {
		t.Fatalf("status response=%d %s", response.Code, response.Body.String())
	}

	ctx, response = testContext(http.MethodPost, "/", nil)
	handler.RunMaintenanceHandler(ctx)
	if response.Code != http.StatusConflict {
		t.Fatalf("run status=%d body=%s", response.Code, response.Body.String())
	}

	maintenance.started = true
	ctx, response = testContext(http.MethodPost, "/", nil)
	handler.RunMaintenanceHandler(ctx)
	if response.Code != http.StatusAccepted || !containsJSON(response.Body.Bytes(), `"started":true`) {
		t.Fatalf("run status=%d body=%s", response.Code, response.Body.String())
	}

	maintenance.startErr = packagemaint.ErrShuttingDown
	ctx, response = testContext(http.MethodPost, "/", nil)
	handler.RunMaintenanceHandler(ctx)
	if response.Code != http.StatusServiceUnavailable || !strings.Contains(response.Body.String(), "shutting down") {
		t.Fatalf("shutdown status=%d body=%s", response.Code, response.Body.String())
	}
}

func ginParam(key, value string) gin.Param { return gin.Param{Key: key, Value: value} }

func containsJSON(body []byte, values ...string) bool {
	text := string(body)
	for _, value := range values {
		if !strings.Contains(text, value) {
			return false
		}
	}
	return true
}
