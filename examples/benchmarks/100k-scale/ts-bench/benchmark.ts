/**
 * AgentField TypeScript SDK Benchmark
 *
 * Measures: registration time, memory footprint, cold start, request latency
 */

import { Agent } from '@agent-field/sdk';

interface BenchmarkResult {
  metric: string;
  value: number;
  unit: string;
  iterations?: number;
}

interface BenchmarkSuite {
  framework: string;
  language: string;
  nodeVersion: string;
  timestamp: string;
  system: {
    platform: string;
    arch: string;
  };
  results: BenchmarkResult[];
  rawData: Record<string, number[]>;
}

interface Stats {
  mean: number;
  stdDev: number;
  min: number;
  max: number;
  p50: number;
  p95: number;
  p99: number;
}

function calculateStats(data: number[]): Stats {
  if (data.length === 0) {
    return { mean: 0, stdDev: 0, min: 0, max: 0, p50: 0, p95: 0, p99: 0 };
  }

  const sorted = [...data].sort((a, b) => a - b);
  const sum = data.reduce((a, b) => a + b, 0);
  const mean = sum / data.length;

  const variance = data.reduce((acc, val) => acc + (val - mean) ** 2, 0) / data.length;
  const stdDev = Math.sqrt(variance);

  const percentile = (p: number) => sorted[Math.floor((sorted.length - 1) * p)];

  return {
    mean,
    stdDev,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99),
  };
}

function getMemoryUsageMB(): number {
  const used = process.memoryUsage();
  return used.heapUsed / 1024 / 1024;
}

async function benchmarkRegistration(
  numHandlers: number,
  iterations: number,
  warmup: number,
  verbose: boolean
): Promise<number[]> {
  if (verbose) {
    console.log(`Benchmark: Handler Registration (${numHandlers} handlers)`);
  }

  const results: number[] = [];

  for (let i = 0; i < iterations + warmup; i++) {
    // Force GC if available
    if (global.gc) global.gc();

    const start = performance.now();

    const agent = new Agent({
      nodeId: `bench-${i}`,
      version: '1.0.0',
      port: 0, // Random port
      agentFieldUrl: 'http://localhost:8080',
    });

    for (let j = 0; j < numHandlers; j++) {
      const idx = j;
      agent.reasoner(`handler-${j}`, async (ctx) => {
        return { id: idx, processed: true };
      });
    }

    const elapsed = performance.now() - start;

    if (i >= warmup) {
      results.push(elapsed);
      if (verbose) {
        console.log(`  Run ${i - warmup + 1}: ${elapsed.toFixed(2)} ms`);
      }
    }
  }

  return results;
}

async function benchmarkMemory(
  numHandlers: number,
  iterations: number,
  warmup: number,
  verbose: boolean
): Promise<number[]> {
  if (verbose) {
    console.log(`\nBenchmark: Memory Footprint (${numHandlers} handlers)`);
  }

  const results: number[] = [];

  for (let i = 0; i < iterations + warmup; i++) {
    if (global.gc) global.gc();
    await new Promise((r) => setTimeout(r, 50));

    const memBefore = getMemoryUsageMB();

    const agent = new Agent({
      nodeId: `mem-bench-${i}`,
      version: '1.0.0',
      port: 0,
      agentFieldUrl: 'http://localhost:8080',
    });

    for (let j = 0; j < numHandlers; j++) {
      const idx = j;
      agent.reasoner(`handler-${j}`, async (ctx) => {
        return { id: idx };
      });
    }

    if (global.gc) global.gc();
    await new Promise((r) => setTimeout(r, 10));

    const memAfter = getMemoryUsageMB();
    const memUsed = memAfter - memBefore;

    if (i >= warmup) {
      results.push(Math.max(0, memUsed)); // Avoid negative due to GC timing
      if (verbose) {
        console.log(`  Run ${i - warmup + 1}: ${memUsed.toFixed(2)} MB`);
      }
    }
  }

  return results;
}

async function benchmarkColdStart(
  iterations: number,
  warmup: number,
  verbose: boolean
): Promise<number[]> {
  if (verbose) {
    console.log(`\nBenchmark: Cold Start Time`);
  }

  const results: number[] = [];

  for (let i = 0; i < iterations + warmup; i++) {
    if (global.gc) global.gc();

    const start = performance.now();

    const agent = new Agent({
      nodeId: `cold-${i}`,
      version: '1.0.0',
      port: 0,
      agentFieldUrl: 'http://localhost:8080',
    });

    // Register one handler to be "ready"
    agent.reasoner('ping', async () => ({ pong: true }));

    const elapsed = performance.now() - start;

    if (i >= warmup) {
      results.push(elapsed);
      if (verbose) {
        console.log(`  Run ${i - warmup + 1}: ${elapsed.toFixed(3)} ms`);
      }
    }
  }

  return results;
}

async function benchmarkRequestProcessing(
  numHandlers: number,
  numRequests: number,
  verbose: boolean
): Promise<number[]> {
  if (verbose) {
    console.log(`\nBenchmark: Request Processing Latency (${numRequests} requests)`);
  }

  const handlers: Array<(input: any) => Promise<any>> = [];

  for (let i = 0; i < numHandlers; i++) {
    const idx = i;
    handlers.push(async (input: any) => {
      return {
        handler_id: idx,
        processed: true,
        timestamp: Date.now(),
      };
    });
  }

  // Warm up
  const input = { query: 'test', value: 42 };
  for (let i = 0; i < 1000; i++) {
    await handlers[i % numHandlers](input);
  }

  // Measure
  const results: number[] = [];
  for (let i = 0; i < numRequests; i++) {
    const handlerIdx = i % numHandlers;
    const start = performance.now();
    await handlers[handlerIdx](input);
    const elapsed = (performance.now() - start) * 1000; // Convert to microseconds
    results.push(elapsed);
  }

  if (verbose) {
    const stats = calculateStats(results);
    console.log(`  p50: ${stats.p50.toFixed(2)} µs, p95: ${stats.p95.toFixed(2)} µs, p99: ${stats.p99.toFixed(2)} µs`);
  }

  return results;
}

async function main() {
  const args = process.argv.slice(2);
  const numHandlers = parseInt(args.find((a) => !a.startsWith('-'))?.replace('--handlers=', '') || '100000');
  const iterations = 10;
  const warmup = 2;
  const jsonOutput = args.includes('--json');
  const verbose = !jsonOutput;

  const suite: BenchmarkSuite = {
    framework: 'AgentField',
    language: 'TypeScript',
    nodeVersion: process.version,
    timestamp: new Date().toISOString(),
    system: {
      platform: process.platform,
      arch: process.arch,
    },
    results: [],
    rawData: {},
  };

  if (verbose) {
    console.log('AgentField TypeScript SDK Benchmark');
    console.log('====================================');
    console.log(`Handlers: ${numHandlers} | Iterations: ${iterations} | Warmup: ${warmup}\n`);
  }

  // Registration benchmark
  const regTimes = await benchmarkRegistration(numHandlers, iterations, warmup, verbose);
  const regStats = calculateStats(regTimes);
  suite.rawData['registration_time_ms'] = regTimes;
  suite.results.push(
    { metric: 'registration_time_mean_ms', value: regStats.mean, unit: 'ms', iterations: regTimes.length },
    { metric: 'registration_time_stddev_ms', value: regStats.stdDev, unit: 'ms' },
    { metric: 'registration_time_p50_ms', value: regStats.p50, unit: 'ms' },
    { metric: 'registration_time_p99_ms', value: regStats.p99, unit: 'ms' }
  );

  // Memory benchmark (smaller scale for TS due to overhead)
  const memHandlers = Math.min(numHandlers, 50000); // TS has more overhead
  const memData = await benchmarkMemory(memHandlers, iterations, warmup, verbose);
  const memStats = calculateStats(memData);
  suite.rawData['memory_mb'] = memData;
  suite.results.push(
    { metric: 'memory_mean_mb', value: memStats.mean, unit: 'MB', iterations: memData.length },
    { metric: 'memory_stddev_mb', value: memStats.stdDev, unit: 'MB' },
    { metric: 'memory_per_handler_bytes', value: (memStats.mean * 1024 * 1024) / memHandlers, unit: 'bytes' }
  );

  // Cold start benchmark
  const coldTimes = await benchmarkColdStart(iterations, warmup, verbose);
  const coldStats = calculateStats(coldTimes);
  suite.rawData['cold_start_ms'] = coldTimes;
  suite.results.push(
    { metric: 'cold_start_mean_ms', value: coldStats.mean, unit: 'ms', iterations: coldTimes.length },
    { metric: 'cold_start_p99_ms', value: coldStats.p99, unit: 'ms' }
  );

  // Request latency benchmark
  const reqTimes = await benchmarkRequestProcessing(Math.min(numHandlers, 10000), 10000, verbose);
  const reqStats = calculateStats(reqTimes);
  suite.rawData['request_latency_us'] = reqTimes;
  suite.results.push(
    { metric: 'request_latency_mean_us', value: reqStats.mean, unit: 'us' },
    { metric: 'request_latency_p50_us', value: reqStats.p50, unit: 'us' },
    { metric: 'request_latency_p95_us', value: reqStats.p95, unit: 'us' },
    { metric: 'request_latency_p99_us', value: reqStats.p99, unit: 'us' }
  );

  if (reqStats.mean > 0) {
    suite.results.push({
      metric: 'theoretical_single_thread_rps',
      value: 1_000_000 / reqStats.mean,
      unit: 'req/s',
    });
  }

  if (jsonOutput) {
    console.log(JSON.stringify(suite, null, 2));
  } else {
    console.log('\n=== Summary ===');
    console.log(`Registration: ${regStats.mean.toFixed(2)} ms (±${regStats.stdDev.toFixed(2)})`);
    console.log(`Memory: ${memStats.mean.toFixed(2)} MB (${((memStats.mean * 1024 * 1024) / memHandlers).toFixed(0)} bytes/handler)`);
    console.log(`Cold Start: ${coldStats.mean.toFixed(2)} ms`);
    console.log(`Request Latency p99: ${reqStats.p99.toFixed(2)} µs`);
  }
}

main().catch(console.error);
