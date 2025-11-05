package services

import (
	"context"
	"sync"
	"time"

	"github.com/your-org/haxen/control-plane/internal/logger"
	"github.com/your-org/haxen/control-plane/pkg/types"
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

func (pm *PresenceManager) Touch(nodeID string, seenAt time.Time) {
	pm.mu.Lock()
	lease, exists := pm.leases[nodeID]
	if !exists {
		lease = &presenceLease{}
		pm.leases[nodeID] = lease
	}
	lease.LastSeen = seenAt
	lease.MarkedOffline = false
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

func (pm *PresenceManager) SetExpireCallback(fn func(string)) {
	pm.mu.Lock()
	pm.expireCallback = fn
	pm.mu.Unlock()
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

	ctx := context.Background()
	inactive := types.AgentStateInactive
	zero := 0
	update := &types.AgentStatusUpdate{
		State:       &inactive,
		HealthScore: &zero,
		Source:      types.StatusSourcePresence,
		Reason:      "presence lease expired",
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
