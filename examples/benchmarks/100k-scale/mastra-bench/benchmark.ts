/**
 * Mastra AI Benchmark
 *
 * Measures: tool registration time, memory footprint, cold start, invocation latency
 */

import { createTool } from "@mastra/core/tools";
import { z } from "zod";

interface BenchmarkResult {
  metric: string;
  value: number;
  unit: string;
  iterations?: number;
  tool_count?: number;
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

  const variance =
    data.reduce((acc, val) => acc + (val - mean) ** 2, 0) / data.length;
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

async function benchmarkToolRegistration(
  numTools: number,
  iterations: number,
  warmup: number,
  verbose: boolean
): Promise<number[]> {
  if (verbose) {
    console.log(`Benchmark: Tool Registration (${numTools} tools)`);
  }

  const results: number[] = [];

  for (let i = 0; i < iterations + warmup; i++) {
    // Force GC if available
    if (global.gc) global.gc();
    await new Promise((r) => setTimeout(r, 10));

    const start = performance.now();

    const tools: any[] = [];
    for (let j = 0; j < numTools; j++) {
      const idx = j;
      const tool = createTool({
        id: `tool-${idx}`,
        description: `Tool number ${idx}`,
        inputSchema: z.object({
          query: z.string(),
        }),
        outputSchema: z.object({
          tool_id: z.number(),
          processed: z.boolean(),
        }),
        execute: async ({ context }) => {
          return { tool_id: idx, processed: true };
        },
      });
      tools.push(tool);
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
  numTools: number,
  iterations: number,
  warmup: number,
  verbose: boolean
): Promise<number[]> {
  if (verbose) {
    console.log(`\nBenchmark: Memory Footprint (${numTools} tools)`);
  }

  const results: number[] = [];

  for (let i = 0; i < iterations + warmup; i++) {
    if (global.gc) global.gc();
    await new Promise((r) => setTimeout(r, 50));

    const memBefore = getMemoryUsageMB();

    const tools: any[] = [];
    for (let j = 0; j < numTools; j++) {
      const idx = j;
      const tool = createTool({
        id: `tool-${idx}`,
        description: `Tool number ${idx}`,
        inputSchema: z.object({
          query: z.string(),
        }),
        outputSchema: z.object({
          tool_id: z.number(),
        }),
        execute: async ({ context }) => {
          return { tool_id: idx };
        },
      });
      tools.push(tool);
    }

    if (global.gc) global.gc();
    await new Promise((r) => setTimeout(r, 10));

    const memAfter = getMemoryUsageMB();
    const memUsed = memAfter - memBefore;

    if (i >= warmup) {
      results.push(Math.max(0, memUsed));
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

    const pingTool = createTool({
      id: "ping",
      description: "Ping tool",
      inputSchema: z.object({
        query: z.string(),
      }),
      outputSchema: z.object({
        pong: z.boolean(),
      }),
      execute: async () => {
        return { pong: true };
      },
    });

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

async function benchmarkToolInvocation(
  numTools: number,
  numInvocations: number,
  verbose: boolean
): Promise<number[]> {
  if (verbose) {
    console.log(
      `\nBenchmark: Tool Invocation Latency (${numInvocations} invocations)`
    );
  }

  // Create tools
  const tools: any[] = [];
  for (let i = 0; i < numTools; i++) {
    const idx = i;
    const tool = createTool({
      id: `tool-${idx}`,
      description: `Tool ${idx}`,
      inputSchema: z.object({
        query: z.string(),
      }),
      outputSchema: z.object({
        tool_id: z.number(),
        processed: z.boolean(),
        timestamp: z.number(),
      }),
      execute: async () => {
        return {
          tool_id: idx,
          processed: true,
          timestamp: Date.now(),
        };
      },
    });
    tools.push(tool);
  }

  // Warm up
  for (let i = 0; i < 1000; i++) {
    await tools[i % numTools].execute({ context: { query: "test" } });
  }

  // Measure
  const results: number[] = [];
  for (let i = 0; i < numInvocations; i++) {
    const toolIdx = i % numTools;
    const start = performance.now();
    await tools[toolIdx].execute({ context: { query: "test" } });
    const elapsed = (performance.now() - start) * 1000; // Convert to microseconds
    results.push(elapsed);
  }

  if (verbose) {
    const stats = calculateStats(results);
    console.log(
      `  p50: ${stats.p50.toFixed(2)} µs, p95: ${stats.p95.toFixed(2)} µs, p99: ${stats.p99.toFixed(2)} µs`
    );
  }

  return results;
}

async function main() {
  const args = process.argv.slice(2);
  const numTools = parseInt(
    args
      .find((a) => a.startsWith("--tools="))
      ?.replace("--tools=", "") || "1000"
  );
  const iterations = parseInt(
    args
      .find((a) => a.startsWith("--iterations="))
      ?.replace("--iterations=", "") || "10"
  );
  const warmup = parseInt(
    args.find((a) => a.startsWith("--warmup="))?.replace("--warmup=", "") || "2"
  );
  const jsonOutput = args.includes("--json");
  const verbose = !jsonOutput;

  const suite: BenchmarkSuite = {
    framework: "Mastra",
    language: "TypeScript",
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
    console.log("Mastra AI Benchmark");
    console.log("===================");
    console.log(
      `Tools: ${numTools} | Iterations: ${iterations} | Warmup: ${warmup}\n`
    );
  }

  // Registration benchmark
  const regTools = Math.min(numTools, 1000);
  const regTimes = await benchmarkToolRegistration(
    regTools,
    iterations,
    warmup,
    verbose
  );
  const regStats = calculateStats(regTimes);
  suite.rawData["registration_time_ms"] = regTimes;
  suite.results.push(
    {
      metric: "registration_time_mean_ms",
      value: regStats.mean,
      unit: "ms",
      iterations: regTimes.length,
      tool_count: regTools,
    },
    { metric: "registration_time_stddev_ms", value: regStats.stdDev, unit: "ms" },
    { metric: "registration_time_p50_ms", value: regStats.p50, unit: "ms" },
    { metric: "registration_time_p99_ms", value: regStats.p99, unit: "ms" }
  );

  // Memory benchmark
  const memTools = Math.min(numTools, 1000);
  const memData = await benchmarkMemory(memTools, iterations, warmup, verbose);
  const memStats = calculateStats(memData);
  suite.rawData["memory_mb"] = memData;
  suite.results.push(
    {
      metric: "memory_mean_mb",
      value: memStats.mean,
      unit: "MB",
      iterations: memData.length,
      tool_count: memTools,
    },
    { metric: "memory_stddev_mb", value: memStats.stdDev, unit: "MB" },
    {
      metric: "memory_per_tool_bytes",
      value: (memStats.mean * 1024 * 1024) / memTools,
      unit: "bytes",
    }
  );

  // Cold start benchmark
  const coldTimes = await benchmarkColdStart(iterations, warmup, verbose);
  const coldStats = calculateStats(coldTimes);
  suite.rawData["cold_start_ms"] = coldTimes;
  suite.results.push(
    {
      metric: "cold_start_mean_ms",
      value: coldStats.mean,
      unit: "ms",
      iterations: coldTimes.length,
    },
    { metric: "cold_start_p99_ms", value: coldStats.p99, unit: "ms" }
  );

  // Invocation latency benchmark
  const invTimes = await benchmarkToolInvocation(
    Math.min(numTools, 100),
    10000,
    verbose
  );
  const invStats = calculateStats(invTimes);
  suite.rawData["invocation_latency_us"] = invTimes;
  suite.results.push(
    { metric: "invocation_latency_mean_us", value: invStats.mean, unit: "us" },
    { metric: "invocation_latency_p50_us", value: invStats.p50, unit: "us" },
    { metric: "invocation_latency_p95_us", value: invStats.p95, unit: "us" },
    { metric: "invocation_latency_p99_us", value: invStats.p99, unit: "us" }
  );

  if (invStats.mean > 0) {
    suite.results.push({
      metric: "theoretical_single_thread_rps",
      value: 1_000_000 / invStats.mean,
      unit: "req/s",
    });
  }

  if (jsonOutput) {
    console.log(JSON.stringify(suite, null, 2));
  } else {
    console.log("\n=== Summary ===");
    console.log(
      `Registration (${regTools}): ${regStats.mean.toFixed(2)} ms (±${regStats.stdDev.toFixed(2)})`
    );
    console.log(
      `Memory (${memTools}): ${memStats.mean.toFixed(2)} MB (${((memStats.mean * 1024 * 1024) / memTools).toFixed(0)} bytes/tool)`
    );
    console.log(`Cold Start: ${coldStats.mean.toFixed(2)} ms`);
    console.log(`Invocation Latency p99: ${invStats.p99.toFixed(2)} µs`);
  }
}

main().catch(console.error);
