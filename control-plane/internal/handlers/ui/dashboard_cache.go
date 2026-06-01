package ui

import (
	"fmt"
	"sync"
	"time"
)

// DashboardCache provides 30-second caching for dashboard data
type DashboardCache struct {
	data      *DashboardSummaryResponse
	timestamp time.Time
	mutex     sync.RWMutex
	ttl       time.Duration
}

// NewDashboardCache creates a new dashboard cache with 30-second TTL
func NewDashboardCache() *DashboardCache {
	return &DashboardCache{
		ttl: 30 * time.Second,
	}
}

// Get retrieves cached data if still valid
func (c *DashboardCache) Get() (*DashboardSummaryResponse, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if c.data != nil && time.Since(c.timestamp) < c.ttl {
		return c.data, true
	}
	return nil, false
}

// Set stores data in cache with current timestamp
func (c *DashboardCache) Set(data *DashboardSummaryResponse) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.data = data
	c.timestamp = time.Now()
}

// TimeRangePreset represents standard time range options
type TimeRangePreset string

const (
	TimeRangePreset1h     TimeRangePreset = "1h"
	TimeRangePreset24h    TimeRangePreset = "24h"
	TimeRangePreset7d     TimeRangePreset = "7d"
	TimeRangePreset30d    TimeRangePreset = "30d"
	TimeRangePresetCustom TimeRangePreset = "custom"
)

// enhancedCacheEntry holds cached data with timestamp
type enhancedCacheEntry struct {
	data      *EnhancedDashboardResponse
	timestamp time.Time
}

// EnhancedDashboardCache provides time-range-aware caching for the enhanced dashboard response
type EnhancedDashboardCache struct {
	entries map[string]*enhancedCacheEntry
	mutex   sync.RWMutex
	maxSize int
}

// NewEnhancedDashboardCache creates a new cache instance for enhanced dashboard data
func NewEnhancedDashboardCache() *EnhancedDashboardCache {
	return &EnhancedDashboardCache{
		entries: make(map[string]*enhancedCacheEntry),
		maxSize: 10, // LRU limit
	}
}

// getTTLForPreset returns the appropriate cache TTL based on time range
func getTTLForPreset(preset TimeRangePreset) time.Duration {
	switch preset {
	case TimeRangePreset1h:
		return 30 * time.Second
	case TimeRangePreset24h:
		return 60 * time.Second
	case TimeRangePreset7d:
		return 2 * time.Minute
	case TimeRangePreset30d:
		return 5 * time.Minute
	default:
		return 60 * time.Second
	}
}

// generateCacheKey creates a cache key from time range parameters
func generateCacheKey(startTime, endTime time.Time, compare bool) string {
	// Round to hour for better cache reuse
	startHour := startTime.Truncate(time.Hour).Unix()
	endHour := endTime.Truncate(time.Hour).Unix()
	compareStr := "0"
	if compare {
		compareStr = "1"
	}
	return fmt.Sprintf("%d-%d-%s", startHour, endHour, compareStr)
}

// Get retrieves cached enhanced dashboard data if still valid
func (c *EnhancedDashboardCache) Get(key string, preset TimeRangePreset) (*EnhancedDashboardResponse, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	entry, exists := c.entries[key]
	if !exists {
		return nil, false
	}

	ttl := getTTLForPreset(preset)
	if time.Since(entry.timestamp) >= ttl {
		return nil, false
	}

	return entry.data, true
}

// Set stores enhanced dashboard data in the cache with LRU eviction
func (c *EnhancedDashboardCache) Set(key string, data *EnhancedDashboardResponse) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Simple LRU: if at capacity, remove oldest entry
	if len(c.entries) >= c.maxSize {
		var oldestKey string
		var oldestTime time.Time
		for k, entry := range c.entries {
			if oldestKey == "" || entry.timestamp.Before(oldestTime) {
				oldestKey = k
				oldestTime = entry.timestamp
			}
		}
		if oldestKey != "" {
			delete(c.entries, oldestKey)
		}
	}

	c.entries[key] = &enhancedCacheEntry{
		data:      data,
		timestamp: time.Now(),
	}
}
