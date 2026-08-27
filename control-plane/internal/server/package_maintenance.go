package server

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagejobs"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagemaint"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"gopkg.in/yaml.v3"
)

const packageExecutionFallbackLimit = 500

type unattendedPackageJobs struct{ manager *packagejobs.Manager }

func (j unattendedPackageJobs) StartUpdate(name, source string) (*packagejobs.Job, error) {
	return j.manager.StartMaintenanceUpdate(name, source)
}

func (j unattendedPackageJobs) GetJob(id string) (*packagejobs.Job, bool) {
	return j.manager.GetJob(id)
}

func (j unattendedPackageJobs) ActiveFor(name string) bool {
	return j.manager.ActiveFor(name)
}

type packageExecutionActivity struct {
	storage storage.StorageProvider
	home    string
}

func (a packageExecutionActivity) HasActiveExecutions(ctx context.Context, packageName string) (bool, error) {
	candidates, err := packageAgentNodeIDs(a.home, packageName)
	if err != nil {
		return false, err
	}
	for _, status := range []string{
		types.ExecutionStatusRunning,
		types.ExecutionStatusPending,
		types.ExecutionStatusQueued,
		types.ExecutionStatusWaiting,
	} {
		for _, candidate := range candidates {
			candidate := candidate
			executions, err := a.storage.QueryExecutionRecords(ctx, types.ExecutionFilter{
				AgentNodeID: &candidate, Status: &status, Limit: 1, ExcludePayloads: true,
			})
			if err != nil {
				return false, err
			}
			if len(executions) > 0 {
				return true, nil
			}
		}
		if len(candidates) == 1 {
			continue
		}
		executions, err := a.storage.QueryExecutionRecords(ctx, types.ExecutionFilter{
			Status: &status, Limit: packageExecutionFallbackLimit, ExcludePayloads: true,
		})
		if err != nil {
			return false, err
		}
		for _, execution := range executions {
			if execution == nil {
				continue
			}
			for _, candidate := range candidates {
				if packages.NodeIDsEquivalent(execution.AgentNodeID, candidate) {
					return true, nil
				}
			}
		}
	}
	return false, nil
}

func (a packageExecutionActivity) ActiveExecutions(ctx context.Context, packageName string) (int, error) {
	candidates, err := packageAgentNodeIDs(a.home, packageName)
	if err != nil {
		return 0, err
	}
	active := make(map[string]struct{})
	for _, status := range []string{
		types.ExecutionStatusRunning,
		types.ExecutionStatusPending,
		types.ExecutionStatusQueued,
		types.ExecutionStatusWaiting,
	} {
		for _, candidate := range candidates {
			candidate := candidate
			executions, err := a.storage.QueryExecutionRecords(ctx, types.ExecutionFilter{
				AgentNodeID:     &candidate,
				Status:          &status,
				Limit:           packageExecutionFallbackLimit,
				ExcludePayloads: true,
			})
			if err != nil {
				return 0, err
			}
			for _, execution := range executions {
				if execution != nil {
					active[execution.ExecutionID] = struct{}{}
				}
			}
		}

		if len(candidates) == 1 {
			continue
		}
		// More than one resolved candidate means the registry/manifest names do
		// not identify one exact execution key. Scan a bounded compatibility
		// window for older SDK punctuation/case variants only in that case.
		executions, err := a.storage.QueryExecutionRecords(ctx, types.ExecutionFilter{
			Status:          &status,
			Limit:           packageExecutionFallbackLimit,
			ExcludePayloads: true,
		})
		if err != nil {
			return 0, err
		}
		for _, execution := range executions {
			if execution == nil {
				continue
			}
			for _, candidate := range candidates {
				if packages.NodeIDsEquivalent(execution.AgentNodeID, candidate) {
					active[execution.ExecutionID] = struct{}{}
					break
				}
			}
		}
	}
	return len(active), nil
}

func packageAgentNodeIDs(home, packageName string) ([]string, error) {
	data, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
	if err != nil {
		return nil, fmt.Errorf("resolve node id for %s: %w", packageName, err)
	}
	var registry packages.InstallationRegistry
	if err := yaml.Unmarshal(data, &registry); err != nil {
		return nil, fmt.Errorf("resolve node id for %s: %w", packageName, err)
	}
	entry, ok := registry.Installed[packageName]
	if !ok {
		return nil, fmt.Errorf("resolve node id for %s: package is not in installed.yaml", packageName)
	}
	metadata, err := packages.ParsePackageMetadata(entry.Path)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("package", packageName).Msg("could not read package manifest for execution activity; using the registry name")
		return []string{packageName}, nil
	}
	candidates := []string{packageName}
	if entry.Name != "" && entry.Name != packageName {
		candidates = append(candidates, entry.Name)
	}
	if metadata.AgentNode.NodeID != "" {
		duplicate := false
		for _, candidate := range candidates {
			duplicate = duplicate || candidate == metadata.AgentNode.NodeID
		}
		if !duplicate {
			candidates = append(candidates, metadata.AgentNode.NodeID)
		}
	}
	return candidates, nil
}

func packageAgentNodeID(home, packageName string) string {
	candidates, err := packageAgentNodeIDs(home, packageName)
	if err != nil || len(candidates) == 0 {
		return packageName
	}
	return candidates[len(candidates)-1]
}

type packageUpdateGraceHook struct {
	home    string
	mu      sync.Mutex
	entries map[string]packageGraceEntry
}

type packageGraceEntry struct {
	nodeID string
	count  int
}

func (h *packageUpdateGraceHook) set(packageName string, active bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if active {
		entry := h.entries[packageName]
		if entry.count == 0 {
			entry.nodeID = packageAgentNodeID(h.home, packageName)
		}
		entry.count++
		h.entries[packageName] = entry
		handlers.SetAgentUpdateInProgress(packageName, true)
		handlers.SetAgentUpdateInProgress(entry.nodeID, true)
		return
	}
	entry, ok := h.entries[packageName]
	if !ok {
		return
	}
	entry.count--
	if entry.count == 0 {
		delete(h.entries, packageName)
	} else {
		h.entries[packageName] = entry
	}
	handlers.SetAgentUpdateInProgress(packageName, false)
	if entry.nodeID != "" {
		handlers.SetAgentUpdateInProgress(entry.nodeID, false)
	}
}

func newPackageMaintenance(home string, store storage.StorageProvider, agent interfaces.AgentService, jobs *packagejobs.Manager, ready <-chan struct{}) *packagemaint.Service {
	grace := &packageUpdateGraceHook{home: home, entries: make(map[string]packageGraceEntry)}
	activity := packageExecutionActivity{storage: store, home: home}
	maintenance := packagemaint.New(packagemaint.Config{
		AgentFieldHome:   home,
		Jobs:             unattendedPackageJobs{manager: jobs},
		Agent:            agent,
		Executions:       activity,
		Ready:            ready,
		OnRestoreState:   grace.set,
		OnRegistryChange: func() { _ = SyncPackagesFromRegistry(home, store) },
	})
	jobs.SetUpdateCacheClearer(maintenance.Checker().Clear)
	jobs.SetUpdateStateHook(grace.set)
	jobs.SetExecutionActivity(func(packageName string) (int, error) {
		return activity.ActiveExecutions(context.Background(), packageName)
	})
	return maintenance
}
