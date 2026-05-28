package services

import (
	"context"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

type PresenceManagerConfig struct {
	HeartbeatTTL  time.Duration
	SweepInterval time.Duration
	HardEvictTTL  time.Duration
}

type presenceLease struct {
	LastSeen      time.Time
	LastExpired   time.Time
	MarkedOffline bool
	Version       string
}

type PresenceManager struct {
	statusManager *StatusManager
	config        PresenceManagerConfig

	leases   map[string]*presenceLease
	mu       sync.RWMutex
	stopCh   chan struct{}
	stopOnce sync.Once

	expireCallback func(string)
}

func NewPresenceManager(statusManager *StatusManager, config PresenceManagerConfig) *PresenceManager {
	if config.HeartbeatTTL == 0 {
		config.HeartbeatTTL = 15 * time.Second
	}
	if config.SweepInterval == 0 {
		config.SweepInterval = config.HeartbeatTTL / 3
		if config.SweepInterval < time.Second {
			config.SweepInterval = time.Second
		}
	}
	if config.HardEvictTTL == 0 {
		config.HardEvictTTL = 5 * time.Minute
	}

	return &PresenceManager{
		statusManager: statusManager,
		config:        config,
		leases:        make(map[string]*presenceLease),
		stopCh:        make(chan struct{}),
	}
}

func (pm *PresenceManager) Start() {
	go pm.loop()
}

func (pm *PresenceManager) Stop() {
	pm.stopOnce.Do(func() {
		close(pm.stopCh)
	})
}

func (pm *PresenceManager) Touch(nodeID string, version string, seenAt time.Time) {
	pm.mu.Lock()
	lease, exists := pm.leases[nodeID]
	if !exists {
		lease = &presenceLease{}
		pm.leases[nodeID] = lease
	}
	lease.LastSeen = seenAt
	lease.MarkedOffline = false
	if version != "" {
		lease.Version = version
	}
	pm.mu.Unlock()
}

func (pm *PresenceManager) Forget(nodeID string) {
	pm.mu.Lock()
	delete(pm.leases, nodeID)
	pm.mu.Unlock()
}

func (pm *PresenceManager) HasLease(nodeID string) bool {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	_, exists := pm.leases[nodeID]
	return exists
}

// HasFreshLease returns true if the agent has a lease with a heartbeat
// received within the HeartbeatTTL. This is used by the health monitor
// to avoid marking agents inactive when heartbeats are still flowing.
func (pm *PresenceManager) HasFreshLease(nodeID string) bool {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	lease, exists := pm.leases[nodeID]
	if !exists {
		return false
	}
	return !lease.MarkedOffline && time.Since(lease.LastSeen) < pm.config.HeartbeatTTL
}

func (pm *PresenceManager) SetExpireCallback(fn func(string)) {
	pm.mu.Lock()
	pm.expireCallback = fn
	pm.mu.Unlock()
}

// RecoverFromDatabase loads previously registered nodes from the database
// and initializes presence leases based on their LastHeartbeat timestamps.
// This should be called on startup to recover state after a control plane restart.
func (pm *PresenceManager) RecoverFromDatabase(ctx context.Context, storageProvider storage.StorageProvider) error {
	nodes, err := storageProvider.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return err
	}

	if len(nodes) == 0 {
		logger.Logger.Debug().Msg("📍 No nodes to recover for presence manager")
		return nil
	}

	logger.Logger.Info().Int("count", len(nodes)).Msg("📍 Recovering presence leases from database")

	pm.mu.Lock()
	defer pm.mu.Unlock()

	for _, node := range nodes {
		if node == nil {
			continue
		}

		// Initialize lease based on LastHeartbeat from database
		pm.leases[node.ID] = &presenceLease{
			LastSeen:      node.LastHeartbeat,
			MarkedOffline: time.Since(node.LastHeartbeat) > pm.config.HeartbeatTTL,
			Version:       node.Version,
		}
	}

	logger.Logger.Info().Msg("📍 Presence lease recovery complete")
	return nil
}

func (pm *PresenceManager) loop() {
	ticker := time.NewTicker(pm.config.SweepInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pm.checkExpirations()
		case <-pm.stopCh:
			return
		}
	}
}

func (pm *PresenceManager) checkExpirations() {
	now := time.Now()
	var expired []string

	pm.mu.Lock()
	for nodeID, lease := range pm.leases {
		if now.Sub(lease.LastSeen) >= pm.config.HeartbeatTTL {
			if !lease.MarkedOffline {
				lease.MarkedOffline = true
				lease.LastExpired = now
				expired = append(expired, nodeID)
			} else if pm.config.HardEvictTTL > 0 && now.Sub(lease.LastSeen) >= pm.config.HardEvictTTL {
				delete(pm.leases, nodeID)
			}
		}
	}
	pm.mu.Unlock()

	for _, nodeID := range expired {
		pm.markInactive(nodeID)
	}
}

func (pm *PresenceManager) markInactive(nodeID string) {
	if pm.statusManager == nil {
		return
	}

	// Re-check lease freshness under lock. Between collecting expired nodes in
	// checkExpirations() and calling this callback, a Touch() may have refreshed
	// the lease (e.g., agent re-registered). If so, skip the inactive transition.
	pm.mu.RLock()
	lease, ok := pm.leases[nodeID]
	if !ok {
		pm.mu.RUnlock()
		return
	}
	if !lease.MarkedOffline {
		// Lease was refreshed by Touch() after we collected it as expired
		pm.mu.RUnlock()
		return
	}
	version := lease.Version
	pm.mu.RUnlock()

	ctx := context.Background()
	inactive := types.AgentStateInactive
	zero := 0
	update := &types.AgentStatusUpdate{
		State:       &inactive,
		HealthScore: &zero,
		Source:      types.StatusSourcePresence,
		Reason:      "presence lease expired",
		Version:     version,
	}

	if err := pm.statusManager.UpdateAgentStatus(ctx, nodeID, update); err != nil {
		logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("❌ Failed to mark node inactive from presence manager")
		return
	}

	logger.Logger.Debug().Str("node_id", nodeID).Msg("📉 Presence lease expired; node marked inactive")

	var callback func(string)
	pm.mu.RLock()
	callback = pm.expireCallback
	pm.mu.RUnlock()

	if callback != nil {
		go callback(nodeID)
	}
}
