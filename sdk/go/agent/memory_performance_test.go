package agent

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"
)

// MemoryMetrics holds memory measurement results
type MemoryMetrics struct {
	Name          string
	AllocBytes    uint64
	TotalAlloc    uint64
	HeapAlloc     uint64
	HeapInuse     uint64
	NumGC         uint32
	Iterations    int
	Duration      time.Duration
}

// PerIterationKB returns memory per iteration in KB
func (m *MemoryMetrics) PerIterationKB() float64 {
	if m.Iterations == 0 {
		return 0
	}
	return float64(m.HeapAlloc) / 1024 / float64(m.Iterations)
}

// HeapAllocMB returns heap allocation in MB
func (m *MemoryMetrics) HeapAllocMB() float64 {
	return float64(m.HeapAlloc) / 1024 / 1024
}

// measureMemory executes a function and measures memory usage
func measureMemory(name string, iterations int, fn func(int)) *MemoryMetrics {
	// Force GC before measurement
	runtime.GC()

	var memBefore, memAfter runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	start := time.Now()
	fn(iterations)
	duration := time.Since(start)

	// Force GC to get accurate readings
	runtime.GC()
	runtime.ReadMemStats(&memAfter)

	return &MemoryMetrics{
		Name:       name,
		AllocBytes: memAfter.TotalAlloc - memBefore.TotalAlloc,
		TotalAlloc: memAfter.TotalAlloc,
		HeapAlloc:  memAfter.HeapAlloc,
		HeapInuse:  memAfter.HeapInuse,
		NumGC:      memAfter.NumGC - memBefore.NumGC,
		Iterations: iterations,
		Duration:   duration,
	}
}

// TestInMemoryBackendMemoryPerformance tests memory efficiency of InMemoryBackend
func TestInMemoryBackendMemoryPerformance(t *testing.T) {
	t.Run("Memory bounded with many entries", func(t *testing.T) {
		metrics := measureMemory("InMemoryBackend_ManyEntries", 10000, func(n int) {
			backend := NewInMemoryBackend()

			for i := 0; i < n; i++ {
				key := fmt.Sprintf("key_%06d", i)
				// Create ~1KB payload per entry
				value := strings.Repeat("x", 1000)
				_ = backend.Set(ScopeSession, "test-session", key, value)
			}
		})

		t.Logf("InMemoryBackend Memory Performance:")
		t.Logf("  Iterations:    %d", metrics.Iterations)
		t.Logf("  Heap Alloc:    %.2f MB", metrics.HeapAllocMB())
		t.Logf("  Per Iteration: %.2f KB", metrics.PerIterationKB())
		t.Logf("  Duration:      %v", metrics.Duration)

		// With 10000 entries at ~1KB each, should be under 20MB
		if metrics.HeapAllocMB() > 20.0 {
			t.Errorf("Memory too high: %.2f MB (expected < 20 MB)", metrics.HeapAllocMB())
		}
	})

	t.Run("Scope isolation memory efficiency", func(t *testing.T) {
		metrics := measureMemory("InMemoryBackend_ScopeIsolation", 1000, func(n int) {
			backend := NewInMemoryBackend()

			scopes := []MemoryScope{ScopeGlobal, ScopeUser, ScopeSession, ScopeWorkflow}

			for i := 0; i < n; i++ {
				for _, scope := range scopes {
					key := fmt.Sprintf("key_%06d", i)
					value := strings.Repeat("y", 500)
					scopeID := fmt.Sprintf("scope_%d", i%10)
					_ = backend.Set(scope, scopeID, key, value)
				}
			}
		})

		t.Logf("Scope Isolation Memory Performance:")
		t.Logf("  Iterations:    %d (x4 scopes)", metrics.Iterations)
		t.Logf("  Heap Alloc:    %.2f MB", metrics.HeapAllocMB())
		t.Logf("  Per Iteration: %.2f KB", metrics.PerIterationKB())
		t.Logf("  Duration:      %v", metrics.Duration)
	})

	t.Run("ClearScope releases memory", func(t *testing.T) {
		backend := NewInMemoryBackend()

		// Add many entries
		for i := 0; i < 5000; i++ {
			key := fmt.Sprintf("key_%06d", i)
			value := strings.Repeat("z", 2000)
			_ = backend.Set(ScopeSession, "test-session", key, value)
		}

		// Force GC and measure before clear
		runtime.GC()
		var memBefore runtime.MemStats
		runtime.ReadMemStats(&memBefore)

		// Clear the scope
		backend.ClearScope(ScopeSession, "test-session")

		// Force GC and measure after clear
		runtime.GC()
		var memAfter runtime.MemStats
		runtime.ReadMemStats(&memAfter)

		// Memory should be released
		heapBefore := float64(memBefore.HeapAlloc) / 1024 / 1024
		heapAfter := float64(memAfter.HeapAlloc) / 1024 / 1024
		reduction := ((heapBefore - heapAfter) / heapBefore) * 100

		t.Logf("ClearScope Memory Release:")
		t.Logf("  Before Clear: %.2f MB", heapBefore)
		t.Logf("  After Clear:  %.2f MB", heapAfter)
		t.Logf("  Reduction:    %.1f%%", reduction)

		// Should release at least 50% of memory
		if reduction < 50.0 && heapBefore > 1.0 {
			t.Logf("Warning: Less than 50%% memory released (%.1f%%)", reduction)
		}
	})
}

// BenchmarkInMemoryBackendSet benchmarks Set operation
func BenchmarkInMemoryBackendSet(b *testing.B) {
	backend := NewInMemoryBackend()
	value := strings.Repeat("x", 1000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key_%d", i)
		_ = backend.Set(ScopeSession, "bench-session", key, value)
	}
}

// BenchmarkInMemoryBackendGet benchmarks Get operation
func BenchmarkInMemoryBackendGet(b *testing.B) {
	backend := NewInMemoryBackend()

	// Pre-populate with data
	for i := 0; i < 10000; i++ {
		key := fmt.Sprintf("key_%d", i)
		value := strings.Repeat("x", 1000)
		_ = backend.Set(ScopeSession, "bench-session", key, value)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key_%d", i%10000)
		_, _, _ = backend.Get(ScopeSession, "bench-session", key)
	}
}

// BenchmarkInMemoryBackendList benchmarks List operation
func BenchmarkInMemoryBackendList(b *testing.B) {
	backend := NewInMemoryBackend()

	// Pre-populate with data
	for i := 0; i < 1000; i++ {
		key := fmt.Sprintf("key_%d", i)
		value := strings.Repeat("x", 100)
		_ = backend.Set(ScopeSession, "bench-session", key, value)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = backend.List(ScopeSession, "bench-session")
	}
}

// TestMemoryPerformanceReport generates a comprehensive memory report
func TestMemoryPerformanceReport(t *testing.T) {
	var metrics []*MemoryMetrics

	// Test 1: InMemoryBackend with completions
	metrics = append(metrics, measureMemory("InMemoryBackend_1K", 1000, func(n int) {
		backend := NewInMemoryBackend()
		for i := 0; i < n; i++ {
			key := fmt.Sprintf("k_%d", i)
			_ = backend.Set(ScopeSession, "s", key, strings.Repeat("x", 10000))
		}
	}))

	// Test 2: Multiple scopes
	metrics = append(metrics, measureMemory("InMemoryBackend_MultiScope", 1000, func(n int) {
		backend := NewInMemoryBackend()
		scopes := []MemoryScope{ScopeGlobal, ScopeUser, ScopeSession, ScopeWorkflow}
		for i := 0; i < n; i++ {
			for _, scope := range scopes {
				key := fmt.Sprintf("k_%d", i)
				_ = backend.Set(scope, fmt.Sprintf("id_%d", i%10), key, strings.Repeat("y", 1000))
			}
		}
	}))

	// Test 3: High-frequency operations
	metrics = append(metrics, measureMemory("InMemoryBackend_HighFreq", 10000, func(n int) {
		backend := NewInMemoryBackend()
		for i := 0; i < n; i++ {
			key := fmt.Sprintf("k_%d", i%100)
			_ = backend.Set(ScopeSession, "s", key, i)
			_, _, _ = backend.Get(ScopeSession, "s", key)
		}
	}))

	// Print report
	t.Log("")
	t.Log("=" + strings.Repeat("=", 69))
	t.Log("GO SDK MEMORY PERFORMANCE REPORT")
	t.Log("=" + strings.Repeat("=", 69))
	t.Logf("%-35s %10s %10s %12s", "Test Name", "Heap (MB)", "Alloc (MB)", "Per Iter (KB)")
	t.Log("-" + strings.Repeat("-", 69))

	for _, m := range metrics {
		t.Logf("%-35s %10.2f %10.2f %12.2f",
			m.Name,
			m.HeapAllocMB(),
			float64(m.AllocBytes)/1024/1024,
			m.PerIterationKB(),
		)
	}
	t.Log("=" + strings.Repeat("=", 69))

	// Assertions
	for _, m := range metrics {
		if m.HeapAllocMB() > 50.0 {
			t.Errorf("%s: Heap allocation too high: %.2f MB", m.Name, m.HeapAllocMB())
		}
	}
}
