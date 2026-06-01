package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	agent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

// BenchmarkResult holds results from a single benchmark run
type BenchmarkResult struct {
	Metric     string  `json:"metric"`
	Value      float64 `json:"value"`
	Unit       string  `json:"unit"`
	Iterations int     `json:"iterations,omitempty"`
}

// BenchmarkSuite holds all results
type BenchmarkSuite struct {
	Framework   string             `json:"framework"`
	Language    string             `json:"language"`
	GoVersion   string             `json:"go_version"`
	Timestamp   string             `json:"timestamp"`
	System      SystemInfo         `json:"system"`
	Results     []BenchmarkResult  `json:"results"`
	RawData     map[string][]float64 `json:"raw_data"`
}

type SystemInfo struct {
	OS       string `json:"os"`
	Arch     string `json:"arch"`
	NumCPU   int    `json:"num_cpu"`
	MaxProcs int    `json:"max_procs"`
}

// Stats calculates statistical measures
type Stats struct {
	Mean   float64
	StdDev float64
	Min    float64
	Max    float64
	P50    float64
	P95    float64
	P99    float64
}

func calculateStats(data []float64) Stats {
	if len(data) == 0 {
		return Stats{}
	}

	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	var sum float64
	for _, v := range data {
		sum += v
	}
	mean := sum / float64(len(data))

	var variance float64
	for _, v := range data {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(data))
	stdDev := math.Sqrt(variance)

	return Stats{
		Mean:   mean,
		StdDev: stdDev,
		Min:    sorted[0],
		Max:    sorted[len(sorted)-1],
		P50:    percentile(sorted, 0.50),
		P95:    percentile(sorted, 0.95),
		P99:    percentile(sorted, 0.99),
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}

func main() {
	var (
		numHandlers = flag.Int("handlers", 100000, "Number of handlers to register")
		iterations  = flag.Int("iterations", 10, "Number of benchmark iterations")
		warmup      = flag.Int("warmup", 2, "Number of warmup iterations to discard")
		serverMode  = flag.Bool("server", false, "Run in server mode for throughput testing")
		jsonOutput  = flag.Bool("json", false, "Output results as JSON")
	)
	flag.Parse()

	suite := BenchmarkSuite{
		Framework: "AgentField",
		Language:  "Go",
		GoVersion: runtime.Version(),
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		System: SystemInfo{
			OS:       runtime.GOOS,
			Arch:     runtime.GOARCH,
			NumCPU:   runtime.NumCPU(),
			MaxProcs: runtime.GOMAXPROCS(0),
		},
		RawData: make(map[string][]float64),
	}

	if !*jsonOutput {
		fmt.Printf("AgentField Go SDK Benchmark\n")
		fmt.Printf("===========================\n")
		fmt.Printf("Handlers: %d | Iterations: %d | Warmup: %d\n\n", *numHandlers, *iterations, *warmup)
	}

	// Benchmark 1: Handler Registration Time
	regTimes := benchmarkRegistration(*numHandlers, *iterations, *warmup, !*jsonOutput)
	regStats := calculateStats(regTimes)
	suite.RawData["registration_time_ms"] = regTimes
	suite.Results = append(suite.Results,
		BenchmarkResult{Metric: "registration_time_mean_ms", Value: regStats.Mean, Unit: "ms", Iterations: len(regTimes)},
		BenchmarkResult{Metric: "registration_time_stddev_ms", Value: regStats.StdDev, Unit: "ms"},
		BenchmarkResult{Metric: "registration_time_p50_ms", Value: regStats.P50, Unit: "ms"},
		BenchmarkResult{Metric: "registration_time_p99_ms", Value: regStats.P99, Unit: "ms"},
	)

	// Benchmark 2: Memory Footprint
	memData := benchmarkMemory(*numHandlers, *iterations, *warmup, !*jsonOutput)
	memStats := calculateStats(memData)
	suite.RawData["memory_mb"] = memData
	suite.Results = append(suite.Results,
		BenchmarkResult{Metric: "memory_mean_mb", Value: memStats.Mean, Unit: "MB", Iterations: len(memData)},
		BenchmarkResult{Metric: "memory_stddev_mb", Value: memStats.StdDev, Unit: "MB"},
		BenchmarkResult{Metric: "memory_per_handler_bytes", Value: (memStats.Mean * 1024 * 1024) / float64(*numHandlers), Unit: "bytes"},
	)

	// Benchmark 3: Cold Start Time (time to create agent)
	coldTimes := benchmarkColdStart(*iterations, *warmup, !*jsonOutput)
	coldStats := calculateStats(coldTimes)
	suite.RawData["cold_start_ms"] = coldTimes
	suite.Results = append(suite.Results,
		BenchmarkResult{Metric: "cold_start_mean_ms", Value: coldStats.Mean, Unit: "ms", Iterations: len(coldTimes)},
		BenchmarkResult{Metric: "cold_start_p99_ms", Value: coldStats.P99, Unit: "ms"},
	)

	// Benchmark 4: Request Processing (internal, no HTTP)
	reqTimes := benchmarkRequestProcessing(*numHandlers, 10000, !*jsonOutput)
	reqStats := calculateStats(reqTimes)
	suite.RawData["request_latency_us"] = reqTimes
	suite.Results = append(suite.Results,
		BenchmarkResult{Metric: "request_latency_mean_us", Value: reqStats.Mean, Unit: "us"},
		BenchmarkResult{Metric: "request_latency_p50_us", Value: reqStats.P50, Unit: "us"},
		BenchmarkResult{Metric: "request_latency_p95_us", Value: reqStats.P95, Unit: "us"},
		BenchmarkResult{Metric: "request_latency_p99_us", Value: reqStats.P99, Unit: "us"},
	)

	// Calculate throughput from latency
	if reqStats.Mean > 0 {
		theoreticalRPS := 1_000_000 / reqStats.Mean // us to seconds
		suite.Results = append(suite.Results,
			BenchmarkResult{Metric: "theoretical_single_thread_rps", Value: theoreticalRPS, Unit: "req/s"},
		)
	}

	if *jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(suite)
	} else {
		fmt.Printf("\n=== Summary ===\n")
		fmt.Printf("Registration: %.2f ms (±%.2f)\n", regStats.Mean, regStats.StdDev)
		fmt.Printf("Memory: %.2f MB (%.0f bytes/handler)\n", memStats.Mean, (memStats.Mean*1024*1024)/float64(*numHandlers))
		fmt.Printf("Cold Start: %.2f ms\n", coldStats.Mean)
		fmt.Printf("Request Latency p99: %.2f µs\n", reqStats.P99)
	}

	if *serverMode {
		runServer(*numHandlers)
	}
}

func benchmarkRegistration(numHandlers, iterations, warmup int, verbose bool) []float64 {
	if verbose {
		fmt.Printf("Benchmark: Handler Registration (%d handlers)\n", numHandlers)
	}

	var results []float64

	for i := 0; i < iterations+warmup; i++ {
		runtime.GC()
		time.Sleep(10 * time.Millisecond) // Let GC settle

		start := time.Now()

		a, err := agent.New(agent.Config{
			NodeID:           fmt.Sprintf("bench-%d", i),
			Version:          "1.0.0",
			ListenAddress:    ":0", // Random port
			DisableLeaseLoop: true,
		})
		if err != nil {
			panic(err)
		}

		for j := 0; j < numHandlers; j++ {
			idx := j
			a.RegisterReasoner(
				fmt.Sprintf("handler-%d", j),
				func(ctx context.Context, input map[string]any) (any, error) {
					return map[string]any{"id": idx, "processed": true}, nil
				},
			)
		}

		elapsed := time.Since(start)
		elapsedMs := float64(elapsed.Microseconds()) / 1000.0

		if i >= warmup {
			results = append(results, elapsedMs)
			if verbose {
				fmt.Printf("  Run %d: %.2f ms\n", i-warmup+1, elapsedMs)
			}
		}
	}

	return results
}

func benchmarkMemory(numHandlers, iterations, warmup int, verbose bool) []float64 {
	if verbose {
		fmt.Printf("\nBenchmark: Memory Footprint (%d handlers)\n", numHandlers)
	}

	var results []float64

	for i := 0; i < iterations+warmup; i++ {
		// Force GC to get clean baseline
		runtime.GC()
		runtime.GC() // Double GC for thorough cleanup
		time.Sleep(50 * time.Millisecond)
		var mBefore runtime.MemStats
		runtime.ReadMemStats(&mBefore)
		baseHeap := mBefore.HeapAlloc

		a, _ := agent.New(agent.Config{
			NodeID:           fmt.Sprintf("mem-bench-%d", i),
			Version:          "1.0.0",
			ListenAddress:    ":0",
			DisableLeaseLoop: true,
		})

		for j := 0; j < numHandlers; j++ {
			idx := j
			a.RegisterReasoner(
				fmt.Sprintf("handler-%d", j),
				func(ctx context.Context, input map[string]any) (any, error) {
					return map[string]any{"id": idx}, nil
				},
			)
		}

		// Measure heap WITHOUT GC to capture actual allocations
		var mAfter runtime.MemStats
		runtime.ReadMemStats(&mAfter)
		currentHeap := mAfter.HeapAlloc

		// Use HeapAlloc delta (more stable than Alloc)
		var memUsedMB float64
		if currentHeap > baseHeap {
			memUsedMB = float64(currentHeap-baseHeap) / 1024 / 1024
		} else {
			// Fallback: use absolute HeapInuse as minimum estimate
			memUsedMB = float64(mAfter.HeapInuse) / 1024 / 1024 / 10 // Conservative
		}

		if i >= warmup {
			results = append(results, memUsedMB)
			if verbose {
				fmt.Printf("  Run %d: %.2f MB (HeapAlloc: %.2f MB)\n", i-warmup+1, memUsedMB, float64(currentHeap)/1024/1024)
			}
		}

		// Help GC by clearing reference
		a = nil
		runtime.GC()
	}

	return results
}

func benchmarkColdStart(iterations, warmup int, verbose bool) []float64 {
	if verbose {
		fmt.Printf("\nBenchmark: Cold Start Time\n")
	}

	var results []float64

	for i := 0; i < iterations+warmup; i++ {
		runtime.GC()

		start := time.Now()
		a, _ := agent.New(agent.Config{
			NodeID:           fmt.Sprintf("cold-%d", i),
			Version:          "1.0.0",
			ListenAddress:    ":0",
			DisableLeaseLoop: true,
		})
		// Register one handler to be "ready"
		a.RegisterReasoner("ping", func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{"pong": true}, nil
		})
		elapsed := time.Since(start)

		if i >= warmup {
			results = append(results, float64(elapsed.Microseconds())/1000.0)
			if verbose {
				fmt.Printf("  Run %d: %.3f ms\n", i-warmup+1, float64(elapsed.Microseconds())/1000.0)
			}
		}
	}

	return results
}

func benchmarkRequestProcessing(numHandlers, numRequests int, verbose bool) []float64 {
	if verbose {
		fmt.Printf("\nBenchmark: Request Processing Latency (%d requests)\n", numRequests)
	}

	// Create agent with handlers
	a, _ := agent.New(agent.Config{
		NodeID:           "latency-bench",
		Version:          "1.0.0",
		ListenAddress:    ":0",
		DisableLeaseLoop: true,
	})

	// Store handlers in a slice for direct invocation
	type handlerEntry struct {
		name    string
		handler agent.HandlerFunc
	}
	handlers := make([]handlerEntry, numHandlers)

	for i := 0; i < numHandlers; i++ {
		idx := i
		h := func(ctx context.Context, input map[string]any) (any, error) {
			// Simulate minimal work: read input, create output
			_ = input["query"]
			return map[string]any{
				"handler_id": idx,
				"processed":  true,
				"timestamp":  time.Now().UnixNano(),
			}, nil
		}
		handlers[i] = handlerEntry{name: fmt.Sprintf("handler-%d", i), handler: h}
		a.RegisterReasoner(handlers[i].name, h)
	}

	// Warm up
	ctx := context.Background()
	input := map[string]any{"query": "test", "value": 42}
	for i := 0; i < 1000; i++ {
		handlers[i%numHandlers].handler(ctx, input)
	}

	// Measure individual request latencies
	results := make([]float64, numRequests)
	for i := 0; i < numRequests; i++ {
		handlerIdx := i % numHandlers
		start := time.Now()
		_, _ = handlers[handlerIdx].handler(ctx, input)
		results[i] = float64(time.Since(start).Nanoseconds()) / 1000.0 // Convert to microseconds
	}

	if verbose {
		stats := calculateStats(results)
		fmt.Printf("  p50: %.2f µs, p95: %.2f µs, p99: %.2f µs\n", stats.P50, stats.P95, stats.P99)
	}

	return results
}

func runServer(numHandlers int) {
	fmt.Printf("\nStarting server with %d handlers on :8001...\n", numHandlers)
	fmt.Printf("Test with: wrk -t4 -c100 -d30s http://localhost:8001/handler-0\n")
	fmt.Printf("Or: hey -n 10000 -c 100 http://localhost:8001/handler-0\n")

	a, _ := agent.New(agent.Config{
		NodeID:           "server-bench",
		Version:          "1.0.0",
		ListenAddress:    ":8001",
		DisableLeaseLoop: true,
	})

	var requestCount uint64

	for i := 0; i < numHandlers; i++ {
		idx := i
		a.RegisterReasoner(
			fmt.Sprintf("handler-%d", i),
			func(ctx context.Context, input map[string]any) (any, error) {
				atomic.AddUint64(&requestCount, 1)
				return map[string]any{
					"handler_id": idx,
					"processed":  true,
				}, nil
			},
		)
	}

	// Print stats every 5 seconds
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		var lastCount uint64
		for range ticker.C {
			current := atomic.LoadUint64(&requestCount)
			rps := float64(current-lastCount) / 5.0
			fmt.Printf("Requests: %d (%.0f req/s)\n", current, rps)
			lastCount = current
		}
	}()

	a.Run(context.Background())
}

// Additional benchmark: Concurrent handler invocation
func benchmarkConcurrentRequests(numHandlers int, concurrency int, duration time.Duration, verbose bool) (float64, float64) {
	a, _ := agent.New(agent.Config{
		NodeID:           "concurrent-bench",
		Version:          "1.0.0",
		ListenAddress:    ":0",
		DisableLeaseLoop: true,
	})

	handlers := make([]agent.HandlerFunc, numHandlers)
	for i := 0; i < numHandlers; i++ {
		idx := i
		handlers[i] = func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{"id": idx}, nil
		}
		a.RegisterReasoner(fmt.Sprintf("h%d", i), handlers[i])
	}

	var totalRequests uint64
	var totalLatencyNs uint64
	var wg sync.WaitGroup
	ctx := context.Background()
	input := map[string]any{"v": 1}

	start := time.Now()
	deadline := start.Add(duration)

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for time.Now().Before(deadline) {
				handlerIdx := workerID % numHandlers
				reqStart := time.Now()
				handlers[handlerIdx](ctx, input)
				atomic.AddUint64(&totalLatencyNs, uint64(time.Since(reqStart).Nanoseconds()))
				atomic.AddUint64(&totalRequests, 1)
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	rps := float64(totalRequests) / elapsed.Seconds()
	avgLatencyUs := float64(totalLatencyNs) / float64(totalRequests) / 1000.0

	if verbose {
		fmt.Printf("\nConcurrent Benchmark (c=%d, t=%v)\n", concurrency, duration)
		fmt.Printf("  Total Requests: %d\n", totalRequests)
		fmt.Printf("  Throughput: %.0f req/s\n", rps)
		fmt.Printf("  Avg Latency: %.2f µs\n", avgLatencyUs)
	}

	return rps, avgLatencyUs
}
