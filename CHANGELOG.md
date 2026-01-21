# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- changelog:entries -->

## [0.1.32] - 2026-01-21

## [0.1.32-rc.4] - 2026-01-21


### Added

- Feat(deploy): add Railway template for one-click deployment (#149)

* feat(deploy): add Railway template for one-click deployment

Add Railway configuration for easy deployment of the control plane with PostgreSQL:
- railway.toml and railway.json at repo root for Railway auto-detection
- Dockerfile reference to existing control-plane build
- Health check configuration (/api/v1/health)
- README with setup instructions and deploy button

Co-Authored-By: Claude <noreply@anthropic.com>

* fix: use correct CLI installation command

---------

Co-authored-by: Claude <noreply@anthropic.com> (7375d4f)



### Fixed

- Fix(ts-sdk): add HTTP timeout and always log heartbeat failures (#152)

- Add 30-second timeout to axios client to prevent requests from hanging
  indefinitely on network issues (matches Python SDK behavior)
- Always log heartbeat failures regardless of devMode setting to aid
  debugging when agents go offline

This fixes an issue where TypeScript agents would silently stop working
after ~5 minutes on Railway (and potentially other cloud platforms) due to
network requests hanging forever without any error logs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude <noreply@anthropic.com> (8aedc5c)

- Fix: add cache mount IDs for Railway compatibility (#150)

* feat(deploy): add Railway template for one-click deployment

Add Railway configuration for easy deployment of the control plane with PostgreSQL:
- railway.toml and railway.json at repo root for Railway auto-detection
- Dockerfile reference to existing control-plane build
- Health check configuration (/api/v1/health)
- README with setup instructions and deploy button

Co-Authored-By: Claude <noreply@anthropic.com>

* fix: use correct CLI installation command

* fix: add cache mount IDs for Railway compatibility

Railway's Docker builder requires explicit id parameters for cache mounts.
Added id=npm-cache, id=go-build-cache, and id=go-mod-cache to the
respective cache mount directives.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* fix: remove BuildKit cache mounts for Railway compatibility

Railway's builder has specific cache mount requirements that differ from
standard BuildKit. Removing cache mounts entirely - Railway has its own
layer caching, so builds still benefit from caching.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

---------

Co-authored-by: Claude <noreply@anthropic.com> (8ea9ecb)

## [0.1.32-rc.3] - 2026-01-20


### Other

- Add skill execution data to observability events (#147)

Include skill metadata in execution events when skills are invoked:
- skill_id: explicit skill identifier
- skill: skill schema (id, input_schema, tags)
- agent_skills: all skills on the agent node

This mirrors the existing pattern for reasoner data and enables
downstream systems to track skill usage and execution metrics.

Co-authored-by: Claude <noreply@anthropic.com> (584bf74)

- Include input payload in execution events and add output schemas to reasoner examples (#146)

- Include input payload in status update, completion, and failure events
- Add explicit output schemas to analyzeSentiment and processWithNotes reasoners
- Improves event data completeness for downstream consumers

Co-authored-by: Claude <noreply@anthropic.com> (aae99c2)

- Banner update (de723e3)

- Update banner image (dc0ce8f)

- Adds deep research banner to README

Adds a visual banner for the "Deep Research API" to the README file.

This enhances the visual appeal and branding of the project's main documentation page. (41b6ab7)

## [0.1.32-rc.2] - 2026-01-13


### Other

- Add fal-client dependency (#145)

* Fix: detect_multimodal_response now handles message.images

- Add _extract_image_from_data() helper for various image formats
- Add _find_images_recursive() for generalized fallback detection
- Extract images from message.images (OpenRouter/Gemini pattern)
- Handle data URLs with base64 extraction
- Add recursive fallback search for edge cases

* Add ai_generate_image and ai_generate_audio methods

- Add dedicated methods for image and audio generation
- Clearer naming than ai_with_vision/ai_with_audio
- Full documentation with examples
- Uses AIConfig defaults for model selection

* Add image_model computed property to AIConfig

- image_model is an alias for vision_model
- Provides clearer naming for image generation model config
- Backwards compatible - vision_model still works

* Add MediaProvider abstraction with Fal, LiteLLM, OpenRouter support

- MediaProvider abstract base class for unified media generation
- FalProvider: Fal.ai integration for flux-pro, f5-tts, etc.
- LiteLLMProvider: DALL-E, Azure, and LiteLLM-supported backends
- OpenRouterProvider: Gemini and other OpenRouter image models
- Provider registry with get_provider() and register_provider()
- Easy to add custom providers by subclassing MediaProvider

* Update FalProvider with correct fal-client API

- Use subscribe_async() for queue-based reliable execution
- Support fal image size presets (square_hd, landscape_16_9, etc.)
- Add video generation with generate_video() method
- Add audio transcription with transcribe_audio() method
- Support all major fal models: flux/dev, flux/schnell, flux-pro
- Add video models: minimax-video, luma-dream-machine, kling-video
- Improve documentation with examples
- Add seed, guidance_scale, num_inference_steps parameters

* Add unified multimodal UX with FalProvider integration

- Add fal_api_key and video_model to AIConfig
- Add _fal_provider lazy property to AgentAI
- Route fal-ai/ and fal/ prefixed models to FalProvider in:
  - ai_with_vision() for image generation
  - ai_with_audio() for TTS
- Add ai_generate_video() method for video generation
- Add ai_transcribe_audio() method for speech-to-text
- Update docstrings with Fal examples
- Add comprehensive tests for media providers

Unified UX pattern:
- app.ai_generate_image("...", model="fal-ai/flux/dev")  # Fal
- app.ai_generate_image("...", model="dall-e-3")        # LiteLLM
- app.ai_generate_video("...", model="fal-ai/minimax-video/...")
- app.ai_transcribe_audio(url, model="fal-ai/whisper")

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* Fix lint errors in multimodal UX implementation

- Add TYPE_CHECKING import for MultimodalResponse forward reference (F821)
- Remove unused width/height/content_type variables in FalProvider (F841)
- Remove unused sys/types imports in tests (F401)
- Remove unused result variable in test (F841)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* Fix remaining unused variable lint error

Remove unused result assignment in test_ai_generate_video_uses_default_model.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* Add fal-client dependency for media generation

Required for FalProvider to generate images, video, and transcribe audio
using Fal.ai models (Flux, MiniMax, Whisper, etc.)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (3bdb701)

## [0.1.32-rc.1] - 2026-01-12


### Other

- Santosh/multimodal (#144)

* Fix: detect_multimodal_response now handles message.images

- Add _extract_image_from_data() helper for various image formats
- Add _find_images_recursive() for generalized fallback detection
- Extract images from message.images (OpenRouter/Gemini pattern)
- Handle data URLs with base64 extraction
- Add recursive fallback search for edge cases

* Add ai_generate_image and ai_generate_audio methods

- Add dedicated methods for image and audio generation
- Clearer naming than ai_with_vision/ai_with_audio
- Full documentation with examples
- Uses AIConfig defaults for model selection

* Add image_model computed property to AIConfig

- image_model is an alias for vision_model
- Provides clearer naming for image generation model config
- Backwards compatible - vision_model still works

* Add MediaProvider abstraction with Fal, LiteLLM, OpenRouter support

- MediaProvider abstract base class for unified media generation
- FalProvider: Fal.ai integration for flux-pro, f5-tts, etc.
- LiteLLMProvider: DALL-E, Azure, and LiteLLM-supported backends
- OpenRouterProvider: Gemini and other OpenRouter image models
- Provider registry with get_provider() and register_provider()
- Easy to add custom providers by subclassing MediaProvider

* Update FalProvider with correct fal-client API

- Use subscribe_async() for queue-based reliable execution
- Support fal image size presets (square_hd, landscape_16_9, etc.)
- Add video generation with generate_video() method
- Add audio transcription with transcribe_audio() method
- Support all major fal models: flux/dev, flux/schnell, flux-pro
- Add video models: minimax-video, luma-dream-machine, kling-video
- Improve documentation with examples
- Add seed, guidance_scale, num_inference_steps parameters

* Add unified multimodal UX with FalProvider integration

- Add fal_api_key and video_model to AIConfig
- Add _fal_provider lazy property to AgentAI
- Route fal-ai/ and fal/ prefixed models to FalProvider in:
  - ai_with_vision() for image generation
  - ai_with_audio() for TTS
- Add ai_generate_video() method for video generation
- Add ai_transcribe_audio() method for speech-to-text
- Update docstrings with Fal examples
- Add comprehensive tests for media providers

Unified UX pattern:
- app.ai_generate_image("...", model="fal-ai/flux/dev")  # Fal
- app.ai_generate_image("...", model="dall-e-3")        # LiteLLM
- app.ai_generate_video("...", model="fal-ai/minimax-video/...")
- app.ai_transcribe_audio(url, model="fal-ai/whisper")

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* Fix lint errors in multimodal UX implementation

- Add TYPE_CHECKING import for MultimodalResponse forward reference (F821)
- Remove unused width/height/content_type variables in FalProvider (F841)
- Remove unused sys/types imports in tests (F401)
- Remove unused result variable in test (F841)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* Fix remaining unused variable lint error

Remove unused result assignment in test_ai_generate_video_uses_default_model.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (5f781b8)

## [0.1.31] - 2026-01-12

## [0.1.31-rc.2] - 2026-01-12


### Testing

- Test(server): add tests for public /health endpoint

Add tests to verify:
- /health bypasses API key authentication
- /health returns healthy status with proper JSON response
- /health returns CORS headers for cross-origin requests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com> (4d96445)

## [0.1.31-rc.1] - 2026-01-12


### Fixed

- Fix(server): add public /health endpoint for load balancer health checks

Add a root-level /health endpoint that bypasses API key authentication,
enabling load balancers and container orchestration platforms to perform
health checks without credentials.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com> (f90ef0b)

## [0.1.30] - 2026-01-11

## [0.1.30-rc.1] - 2026-01-11


### Performance

- Perf: Python SDK memory optimization + benchmark visualization improvements (#138)

* feat(benchmarks): add 100K scale benchmark suite

- Go SDK: 100K handlers in 16.4ms, 8.1M req/s throughput
- Python SDK benchmark with memory profiling
- LangChain baseline for comparison
- Seaborn visualizations for technical documentation

Results demonstrate Go SDK advantages:
- ~3,000x faster registration than LangChain at scale
- ~32x more memory efficient per handler
- ~520x higher theoretical throughput

* fix(sdk/python): optimize memory usage - 97% reduction vs baseline

Memory optimizations for Python SDK to significantly reduce memory footprint:

## Changes

### async_config.py
- Reduce result_cache_ttl: 600s -> 120s (2 min)
- Reduce result_cache_max_size: 20000 -> 5000
- Reduce cleanup_interval: 30s -> 10s
- Reduce max_completed_executions: 4000 -> 1000
- Reduce completed_execution_retention_seconds: 600s -> 60s

### client.py
- Add shared HTTP session pool (_shared_sync_session) for connection reuse
- Replace per-request Session creation with class-level shared session
- Add _init_shared_sync_session() and _get_sync_session() class methods
- Reduces connection overhead and memory from session objects

### execution_state.py
- Clear input_data after execution completion (set_result)
- Clear input_data after execution failure (set_error)
- Clear input_data after cancellation (cancel)
- Clear input_data after timeout (timeout_execution)
- Prevents large payloads from being retained in memory

### async_execution_manager.py
- Add 1MB buffer limit for SSE event stream
- Prevents unbounded buffer growth from malformed events

## Benchmark Results

Memory comparison (1000 iterations, ~10KB payloads):
- Baseline pattern: 47.76 MB (48.90 KB/iteration)
- Optimized SDK:     1.30 MB (1.33 KB/iteration)
- Improvement:      97.3% memory reduction

Added benchmark scripts for validation:
- memory_benchmark.py: Component-level memory testing
- benchmark_comparison.py: Full comparison with baseline patterns

* refactor(sdk): convert memory benchmarks to proper test suites

Replace standalone benchmark scripts with proper test suite integration:

## Python SDK
- Remove benchmark_comparison.py and memory_benchmark.py
- Add tests/test_memory_performance.py with pytest integration
- Tests cover AsyncConfig defaults, ExecutionState memory clearing,
  ResultCache bounds, and client session reuse
- Includes baseline comparison and memory regression tests

## Go SDK
- Add agent/memory_performance_test.go
- Benchmarks for InMemoryBackend Set/Get/List operations
- Memory efficiency tests with performance reporting
- ClearScope memory release verification (96.9% reduction)

## TypeScript SDK
- Add tests/memory_performance.test.ts with Vitest
- Agent creation and registration efficiency tests
- Large payload handling tests
- Memory leak prevention tests

All tests verify memory-optimized defaults and proper cleanup.

* feat(ci): add memory performance metrics workflow

Add GitHub Actions workflow that runs memory performance tests
and posts metrics as PR comments when SDK or control-plane changes.

Features:
- Runs Python, Go, TypeScript SDK memory tests
- Runs control-plane benchmarks
- Posts consolidated metrics table as PR comment
- Updates existing comment on subsequent runs
- Triggered on PRs affecting sdk/ or control-plane/

Metrics tracked:
- Heap allocation and per-iteration memory
- Memory reduction percentages
- Memory leak detection results

* feat(ci): enhance SDK performance metrics workflow

Comprehensive performance report for PR reviewers with:

## Quick Status Section
- Traffic light status for each component (✅/❌)
- Overall pass/fail summary at a glance

## Python SDK Metrics
- Lint status (ruff)
- Test count and duration
- Memory test status
- ExecutionState latency (avg/p99)
- Cache operation latency (avg/p99)

## Go SDK Metrics
- Lint status (go vet)
- Test count and duration
- Memory test status
- Heap usage
- ClearScope memory reduction %
- Benchmark: Set/Get ns/op, B/op

## TypeScript SDK Metrics
- Lint status
- Test count and duration
- Memory test status
- Agent creation memory
- Per-agent overhead
- Leak growth after 500 cycles

## Control Plane Metrics
- Build time and status
- Lint status
- Test count and duration

## Collapsible Details
- Each SDK has expandable details section
- Metric definitions table for reference
- Link to workflow logs for debugging

* feat(benchmarks): update with TypeScript SDK and optimized Python SDK

- Add TypeScript SDK benchmark (50K handlers in 16.7ms)
- Re-run all benchmarks with PR #137 Python memory optimizations
- Fix Go memory measurement to use HeapAlloc delta
- Regenerate all visualizations with seaborn

Results:
- Go: 100K handlers in 17.3ms, 280 bytes/handler, 8.2M req/s
- TypeScript: 50K handlers in 16.7ms, 276 bytes/handler
- Python SDK: 5K handlers in 2.97s, 127 MB total
- LangChain: 1K tools in 483ms, 10.8 KB/tool

* perf(python-sdk): optimize startup with lazy loading and add MCP/DID flags

Improvements:
- Implement lazy LiteLLM import in agent_ai.py (saves 10-20MB if AI not used)
- Add lazy loading for ai_handler and cli_handler properties
- Add enable_mcp (default: False) and enable_did (default: True) flags
- MCP disabled by default since not yet fully supported

Benchmark methodology fixes:
- Separate Agent init time from handler registration time
- Measure handler memory independently from Agent overhead
- Increase test scale to 10K handlers (from 5K)

Results:
- Agent Init: 1.07 ms (one-time overhead)
- Agent Memory: 0.10 MB (one-time overhead)
- Cold Start: 1.39 ms (Agent + 1 handler)
- Handler Registration: 0.58 ms/handler
- Handler Memory: 26.4 KB/handler (Pydantic + FastAPI overhead)
- Request Latency p99: 0.17 µs
- Throughput: 7.5M req/s (single-threaded theoretical)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* perf(python-sdk): Reduce per-handler memory from 26.4 KB to 7.4 KB

Architectural changes to reduce memory footprint:

1. Consolidated registries: Replace 3 separate data structures (reasoners list,
   _reasoner_vc_overrides, _reasoner_return_types) with single Dict[str, ReasonerEntry]
   using __slots__ dataclasses.

2. Removed Pydantic create_model(): Each handler was creating a Pydantic model
   class (~1.5-2 KB overhead). Now use runtime validation via _validate_handler_input()
   with type coercion support.

3. On-demand schema generation: Schemas are now generated only when the
   /discover endpoint is called, not stored per-handler. Added _types_to_json_schema()
   and _type_to_json_schema() helper methods.

4. Weakref closures: Use weakref.ref(self) in tracked_func closure to break
   circular references (Agent → tracked_func → Agent) and enable immediate GC.

Benchmark results (10,000 handlers):
- Memory: 26.4 KB/handler → 7.4 KB/handler (72% reduction)
- Registration: 5,797 ms → 624 ms

Also updated benchmark documentation to use neutral technical presentation
without comparative marketing language.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* ci: Redesign PR performance metrics for clarity and regression detection

Simplified the memory-metrics.yml workflow to be scannable and actionable:

- Single clean table instead of 4 collapsible sections
- Delta (Δ) column shows change from baseline
- Only runs benchmarks for affected SDKs (conditional execution)
- Threshold-based warnings: ⚠ at +10%, ✗ at +25% for memory
- Added baseline.json with current metrics for comparison

Example output:
| SDK    | Memory  | Δ    | Latency | Δ | Tests | Status |
|--------|---------|------|---------|---|-------|--------|
| Python | 7.4 KB  | -    | 0.21 µs | - | ✓     | ✓      |

✓ No regressions detected

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* refactor(benchmarks): Consolidate visualization to 2 scientific figures

- Reduce from 6 images to 2 publication-quality figures
- benchmark_summary.png: 2x2 grid with registration, memory, latency, throughput
- latency_comparison.png: CDF and box plot with proper legends
- Fix Python SDK validation error handling (proper HTTP 422 responses)
- Update tests to use new _reasoner_registry (replaces _reasoner_return_types)
- Clean up unused benchmark result files

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* fix(benchmarks): Re-run Python SDK benchmark with optimized code

- Updated AgentField_Python.json with fresh benchmark results
- Memory: 7.5 KB/handler (was 26.4 KB) - 30% better than LangChain
- Registration: 57ms for 1000 handlers (was 5796ms for 10000)
- Consolidated to single clean 2x2 visualization
- Removed comparative text, keeping neutral factual presentation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* feat(benchmarks): Add Pydantic AI comparison, improve visualization

- Add Pydantic AI benchmark (3.4 KB/handler, 0.17µs latency, 9M rps)
- Update color scheme: AgentField SDKs in blue family, others distinct
- Shows AgentField crushing LangChain on key metrics:
  - Latency: 0.21µs vs 118µs (560x faster)
  - Throughput: 6.7M vs 15K (450x higher)
  - Registration: 57ms vs 483ms (8x faster)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* chore(benchmarks): Remove Pydantic AI and CrewAI, keep only LangChain comparison

- Remove pydantic-ai-bench/ directory
- Remove crewai-bench/ directory
- Remove PydanticAI_Python.json results
- Update analyze.py to only include AgentField SDKs + LangChain
- Regenerate benchmark visualization

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* ci fixes

* fix: Python 3.8/3.9 compatibility for dataclass slots parameter

The `slots=True` parameter for dataclass was added in Python 3.10.
This fix conditionally applies slots only on Python 3.10+, maintaining
backward compatibility with Python 3.8 and 3.9 while preserving the
memory optimization on newer versions.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* fix(ci): Fix TypeScript benchmark and update baseline for CI environment

- Fix TypeScript benchmark failing due to top-level await in CJS mode
  - Changed from npx tsx -e to writing .mjs file and running with node
  - Now correctly reports memory (~219 B/handler) and latency metrics

- Update baseline.json to match CI environment (Python 3.11, ubuntu-latest)
  - Python baseline: 7.4 KB → 9.0 KB (reflects actual CI measurements)
  - Increased warning thresholds to 15% to account for cross-platform variance
  - The previous baseline was from Python 3.14/macOS which differs from CI

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* fix(ci): TypeScript benchmark now tests actual SDK instead of raw Map

The CI benchmark was incorrectly measuring a raw JavaScript Map instead
of the actual TypeScript SDK. This fix:

- Adds npm build step before benchmark
- Uses actual Agent class with agent.reasoner() registration
- Measures real SDK overhead (Agent + ReasonerRegistry)
- Updates baseline: 276 → 350 bytes/handler (actual SDK overhead)
- Aligns handler count with Python (1000) for consistency

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* feat(benchmarks): Add CrewAI and Mastra framework comparisons

Add benchmark comparisons for CrewAI (Python) and Mastra (TypeScript):
- CrewAI: AgentField is 3.5x faster registration, 1.9x less memory
- Mastra: AgentField is 27x faster registration, 6.5x less memory

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* docs: Add SDK performance benchmarks to README

Add benchmark comparison tables for Python (vs LangChain, CrewAI) and
TypeScript (vs Mastra) frameworks showing registration time, memory
per handler, and throughput metrics.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>
Co-authored-by: Abir Abbas <abirabbas1998@gmail.com> (8a7fded)

## [0.1.29] - 2026-01-11

## [0.1.29-rc.2] - 2026-01-11


### Fixed

- Fix(sdk): Update TypeScript SDK license to Apache-2.0 (#139)

Align the TypeScript SDK's package.json license field with the
project's root Apache 2.0 license. The SDK was incorrectly showing
MIT on npm.

Co-authored-by: Claude <noreply@anthropic.com> (8b1b1f7)

## [0.1.29-rc.1] - 2026-01-09


### Added

- Feat(ci): add contributor reminder and assignment tracking workflows (#132)

Add automated system to remind assigned contributors and free up stale assignments:

- contributor-reminders.yml: Scheduled daily check that:
  - Sends friendly reminder at 7 days without activity
  - Sends second reminder at 14 days with unassign warning
  - Unassigns and re-labels as 'help wanted' at 21 days
  - Skips issues with linked PRs or blocking labels
  - Supports dry-run mode for testing

- issue-assignment-tracking.yml: Real-time event handling that:
  - Welcomes new assignees with timeline expectations
  - Clears reminder labels when assignees comment
  - Clears labels when assignee opens linked PR
  - Auto-adds 'help wanted' when last assignee leaves

This improves contributor experience by setting clear expectations
while ensuring stale assignments don't block other contributors. (7bbac52)



### Documentation

- Docs: update Docker image references to Docker Hub (#134)

* docs: update Docker image references to Docker Hub

Update all references from ghcr.io/agent-field/agentfield-control-plane
to agentfield/control-plane (Docker Hub).

Files updated:
- deployments/kubernetes/base/control-plane-deployment.yaml
- deployments/helm/agentfield/values.yaml
- examples/python_agent_nodes/rag_evaluation/docker-compose.yml
- README.md
- docs/RELEASE.md (includes new DOCKERHUB_* secrets documentation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* fix: use real version numbers in RELEASE.md examples

Update example commands to use actual versions that exist:
- Docker: staging-0.1.28-rc.4 (not 0.1.19-rc.1)
- Install script: v0.1.28 and v0.1.28-rc.4

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (feeaa21)



### Other

- Add test connection_manager (#135) (247da4d)

## [0.1.28] - 2026-01-06

## [0.1.28-rc.4] - 2026-01-06


### Chores

- Chore(ci): migrate Docker publishing from GHCR to Docker Hub (#133)

- Change image path from ghcr.io/agent-field/agentfield-control-plane to agentfield/control-plane
- Update login step to use Docker Hub credentials (DOCKERHUB_USERNAME, DOCKERHUB_TOKEN)
- Remove unused OWNER env var from Docker metadata step

This enables Docker Hub analytics for image pulls. Requires adding
DOCKERHUB_USERNAME and DOCKERHUB_TOKEN secrets to the repository.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (e6abe54)



### Documentation

- Docs: add Discord community badge to README (#131)

Add a Discord badge near the top of README.md to invite users to join
the community. Uses Discord's official brand color (#5865F2) and matches
the existing badge styling.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude <noreply@anthropic.com> (81fb1c5)

## [0.1.28-rc.3] - 2026-01-05


### Fixed

- Fix(control-plane): enforce lifecycle_status consistency with agent state (#130)

When agents go offline, the control plane was incorrectly keeping
lifecycle_status as "ready" even though health_status correctly showed
"inactive". This caused observability webhooks to receive inconsistent
data where offline nodes appeared online based on lifecycle_status.

Changes:
- Add defensive lifecycle_status enforcement in persistStatus()
  to ensure consistency with agent state before writing to storage
- Update health_monitor.go fallback paths to also update lifecycle_status
- Add SystemStateSnapshot event type for periodic agent inventory
- Enhance execution events with full reasoner context and metadata
- Add ListAgents to ObservabilityWebhookStore interface for snapshots

The fix ensures both node_offline events and system_state_snapshot
events (every 60s) correctly report lifecycle_status: "offline" for
offline agents.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude <noreply@anthropic.com> (67c67c4)

## [0.1.28-rc.2] - 2026-01-05


### Other

- Switch hot-reload dev setup from Docker to native Air (#129)

Removes Docker-based dev setup in favor of running Air directly in the
host environment. This avoids networking issues between Docker and host
(especially on WSL2 where host.docker.internal has limitations).

Changes:
- Remove Dockerfile.dev and docker-compose.dev.yml
- Update dev.sh to run Air natively (auto-installs if missing)
- Update README.md with simplified instructions

Usage remains simple: ./dev.sh

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude <noreply@anthropic.com> (670c0ba)

## [0.1.28-rc.1] - 2026-01-05


### Other

- Hot reload controlplane local setup (#128) (690d481)

## [0.1.27] - 2026-01-02

## [0.1.27-rc.1] - 2026-01-01


### CI

- Ci: disable AI label workflow for fork compatibility

The AI label workflow fails on PRs from forked repositories because
GITHUB_TOKEN lacks write permissions. Since many contributions come
from forks, disabling the workflow until a proper solution (PAT or
GitHub App) is implemented.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com> (6dbc908)



### Other

- Add explicit return type to useFilterState hook (#127)

* Add explicit return type to useFilterState hook

* fix(types): use Partial<ExecutionFilters> in UseFilterStateReturn

The convertTagsToApiFormat function returns Partial<ExecutionFilters>,
so the return type interface must match to avoid TypeScript errors.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

---------

Co-authored-by: Abir Abbas <abirabbas1998@gmail.com>
Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (f2168e0)

## [0.1.26] - 2025-12-27

## [0.1.26-rc.3] - 2025-12-27


### Added

- Feat(sdk/go, control-plane): Add Vector Memory Ops (#124)

* chore(release): v0.1.26-rc.2

* feat: added vector memory ops

* test(handlers): add unit tests for GetVector and DeleteVector handlers

Add comprehensive test coverage for the new vector memory endpoints:

GetVectorHandler tests:
- TestGetVectorHandler_ReturnsVectorWithMetadata: Full happy path with scope/key/metadata
- TestGetVectorHandler_NotFound: 404 when vector doesn't exist
- TestGetVectorHandler_StorageError: 500 on database failure
- TestGetVectorHandler_DefaultScope: Scope resolution from headers

DeleteVectorHandler tests:
- TestDeleteVectorHandler_RESTfulDelete: DELETE with path parameter
- TestDeleteVectorHandler_BackwardCompatibilityWithBody: POST with JSON body
- TestDeleteVectorHandler_StorageError: 500 on database failure
- TestDeleteVectorHandler_MissingKey: 400 when key is missing

Also updated vectorStorageStub to track GetVector and DeleteVector parameters
for assertion verification.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

---------

Co-authored-by: github-actions[bot] <github-actions[bot]@users.noreply.github.com>
Co-authored-by: Abir Abbas <abirabbas1998@gmail.com>
Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (0dd4e62)



### Fixed

- Fix(ci): add issues:write permission for AI label workflow (#126)

The `gh pr edit --add-label` command requires `issues: write` permission
because labels are managed through the issues API in GitHub, even when
applied to pull requests. Without this permission, the workflow fails with:
"GraphQL: Resource not accessible by integration (addLabelsToLabelable)"

Added permissions:
- `issues: write` - Required for adding labels
- `contents: read` - Explicit permission for checkout

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude <noreply@anthropic.com> (26a71a4)

- Fix(ci): prevent shell injection in AI label workflow (#125)

The PR body was being directly interpolated into a shell variable using
`${{ github.event.pull_request.body }}`, which caused shell parsing of
the content. When PR descriptions contained filenames like `CHANGELOG.md`
on their own lines, the shell would attempt to execute them as commands.

This fix passes the PR body via the `env:` block instead, which properly
escapes the content as an environment variable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude <noreply@anthropic.com> (1e2225d)

## [0.1.26-rc.2] - 2025-12-27


### Added

- Feat(tests): add unit tests for vector memory handler functionality (#123) (9df214d)

- Feat: Add AI-assisted contribution guidelines, task issue template, AI labeling workflow, security, and support policies. (87c9297)

## [0.1.26-rc.1] - 2025-12-23


### Added

- Feat(observability): add webhook forwarding with dead letter queue (#102)

* feat(observability): add webhook forwarding with dead letter queue

Add observability webhook system for forwarding events to external endpoints:

- Configurable webhook URL with optional HMAC secret and custom headers
- Event batching with configurable size and timeout
- Automatic retry with exponential backoff
- Dead letter queue (DLQ) for failed events with redrive and clear capabilities
- Filter heartbeat events and minor health score fluctuations to reduce noise
- Settings UI page for configuration and DLQ management

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* test fixes

* fix(observability): node offline events now properly forwarded to webhook

Fixed a race condition where node offline events were not being forwarded
to the observability webhook. The issue was in UpdateAgentStatus which
called GetAgentStatus (performing a live health check) to get the "old"
status. By the time the health check completed, oldStatus == newStatus,
so no events were broadcast.

Changed to use GetAgentStatusSnapshot which returns cached/stored status
without a live check, preserving the true "old" state for comparison.

Also added observability webhook documentation to ARCHITECTURE.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* test(status-manager): add tests for node online/offline event broadcasting

Add comprehensive tests to verify status change events are properly
broadcast when nodes transition between active and inactive states:

- TestStatusManagerBroadcastsNodeOfflineEvent: Verifies NodeOffline/
  NodeUnifiedStatusChanged events are broadcast when node goes offline
- TestStatusManagerBroadcastsNodeOnlineEvent: Verifies NodeOnline/
  NodeUnifiedStatusChanged events are broadcast when node comes online
- TestStatusManagerPreservesOldStatusForEventBroadcast: Verifies the
  old status is correctly captured before updates, ensuring the fix
  for the GetAgentStatus race condition doesn't regress

These tests guard against the race condition where UpdateAgentStatus
was calling GetAgentStatus (with live health check) instead of
GetAgentStatusSnapshot, causing oldStatus == newStatus.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* docs: remove observability webhook documentation

The observability webhook feature remains functional but will not be publicly
documented at this time.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>

* bugfix on ts-sdk json schema return

* webhook secret fix

* refine webhook events

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (27cf1f7)



### Other

- Fix link to AI Backend blog post (fa379c6)

- Update links in README for IAM documentation (3a160de)

- Update README.md (fa62719)

## [0.1.25] - 2025-12-21

## [0.1.25-rc.2] - 2025-12-21


### Added

- Feat(dashboard): add comprehensive observability enhancements (#101) (d947542)

## [0.1.25-rc.1] - 2025-12-21


### Other

- Added note() method for fire-and-forget execution logging (#100) (55fdcf1)

- URL-encode badge endpoint and add cache control (8a4c970)

- Fix shields.io badge: separate badge.json and stats.json files (7d08183)

- Add workflow to update download stats (#87)

* Add workflow to update download stats

Adds a GitHub Actions workflow to automate the collection and updating of download statistics from GitHub releases, PyPI, and NPM.

This workflow:
- Runs every 6 hours or can be triggered manually.
- Fetches download counts from GitHub releases and aggregates them.
- Retrieves total downloads from Pepy.tech for PyPI.
- Collects lifetime download statistics from NPM.
- Calculates a combined total and updates a Gist file.
- The README's download badge is updated to point to this new Gist endpoint for more comprehensive stats.

* Add push trigger for download stats workflow

* Add permissions block to download stats workflow (d400e36)

## [0.1.24] - 2025-12-18

## [0.1.24-rc.3] - 2025-12-18


### Chores

- Chore(rag-eval): update default model to GPT-4o (#85)

- Set GPT-4o as default (reliable JSON output)
- Gemini 2.5 Flash as second option
- Move DeepSeek to last (can timeout)
- Remove old Gemini 2.0 Flash

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (f49959a)



### Documentation

- Docs: Add cross-reference links to RAG evaluator documentation (#83)

- Add docs link in examples/README.md table for rag_evaluation
- Add documentation callout in rag_evaluation/README.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (e51b8d4)

- Docs: Add website deployment guide links (#82)

Reference the full deployment guides at agentfield.ai for Kubernetes
and Helm deployment options.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (59eaf7f)



### Other

- Fix parent-child workflow tracking for direct reasoner calls via AgentRouter (#86)

* Fix parent-child workflow tracking for direct reasoner calls via AgentRouter

When reasoners registered via AgentRouter call other reasoners directly
(as normal async functions), the parent-child relationship was not being
captured in the workflow DAG. This happened because:

1. @router.reasoner() stored the original function but returned it unchanged
2. When include_router() later wrapped functions with tracking, closures in
   other reasoners still held references to the original unwrapped functions
3. Direct calls bypassed the tracked wrapper entirely

This fix implements lazy-binding in AgentRouter.reasoner():
- The decorator now returns a wrapper that looks up the tracked function
  at call time via router._tracked_functions
- include_router() registers tracked functions in this lookup table
- Direct reasoner-to-reasoner calls now go through the tracked wrapper,
  enabling proper parent_execution_id propagation

Changes:
- router.py: Add lazy-binding wrapper in reasoner() decorator
- agent.py: Register tracked functions in router._tracked_functions
- test_router.py: Update tests for new wrapper behavior
- test_workflow_parent_child.py: Add comprehensive tests for parent-child tracking

* Remove unused imports in test_workflow_parent_child.py (342af92)

## [0.1.24-rc.2] - 2025-12-17


### Other

- Deployments: Docker/Helm/Kustomize quickstarts + demo agents (#81)

* Update Docker deployment and configuration

Refactors the Docker deployment documentation and configuration to improve clarity and flexibility for setting up control planes and agents.

Key changes include:
- Enhancing the README for Docker deployments with more detailed instructions for running agents in Docker, distinguishing between agents on the host and agents within the same Docker Compose network.
- Adding specific guidance on using `host.docker.internal` for host-based agents and service names for agents within the same network.
- Introducing new Docker Compose services for a demo Go agent and a demo Python agent, enabling them to be run with Docker Compose profiles.
- Updating configuration options in `control-plane/internal/config/config.go` to include `mapstructure` tags, improving the flexibility of configuration loading.
- Adding a new test case `TestLoadConfig_VCRequirementsFromConfigFile` to verify loading VC requirements from a configuration file.
- Modifying the Python hello world example to use an environment variable for the AgentField server URL, making it more adaptable to different deployment scenarios.
- Updating the Dockerized README to include validation steps for execution paths and Verifiable Credentials (VCs).

* Update deployment documentation and manifests

Updates the README files for Docker, Helm, and Kubernetes deployments to improve clarity and provide more streamlined quick-start guides.

The changes include:
- Simplifying the Docker Compose setup instructions.
- Refining the Helm chart documentation to recommend PostgreSQL and the Python demo agent by default.
- Streamlining the Kubernetes manifests to suggest the Python demo agent overlay as a recommended starting point.
- Modifying the Python demo agent deployment in Kubernetes to directly install the AgentField SDK from PyPI instead of relying on a pre-built local image. This simplifies the local development workflow for the Python agent.

* Update documentation for deployment examples

Adds instructions for waiting for demo agents to become ready and for building/loading the Go demo agent image with Minikube.

Also includes an example of how to use the API key when authentication is enabled.

Updates the control plane deployment configuration to default `AGENTFIELD_CONFIG_FILE` to `/dev/null`.

Adjusts the kustomization file for the postgres demo overlay to use the standard `patches` key. (b6b0cd3)

## [0.1.24-rc.1] - 2025-12-17


### Added

- Feat(go-sdk): add ControlPlaneMemoryBackend for distributed memory (#80)

Add a new MemoryBackend implementation that delegates storage to the
control plane's /api/v1/memory/* endpoints. This enables distributed,
scope-aware memory across agents.

- Implements Set, Get, Delete, and List operations
- Maps SDK scopes (Workflow, Session, User, Global) to API scopes
- User scope maps to "actor" terminology in the API
- Includes comprehensive unit tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (6cd445c)

- Feat: add RAG evaluation example with multi-reasoner architecture (#79)

* feat: add RAG evaluation example with multi-reasoner architecture

Adds a comprehensive RAG evaluation agent demonstrating:
- Adversarial debate for faithfulness (prosecutor vs defender + judge)
- Multi-jury consensus for relevance (3 jurors vote on literal/intent/scope)
- Hybrid ML+LLM chain-of-verification for hallucination detection
- Configurable constitutional principles evaluation

Features:
- Docker Compose deployment (control plane + agent + UI)
- Next.js web interface with claim-level breakdown
- Domain-specific presets (medical, legal, financial)
- 3 evaluation modes: quick (4 calls), standard (14), thorough (20+)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* Remove ARCHITECTURE.md, link to website docs instead

The detailed architecture documentation is now on the website at
agentfield.dev/examples/complete-agents/rag-evaluator - no need
to duplicate content in the repo.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* Add examples README with quick reference table

Index of all examples across Python, TypeScript, and Go with:
- Quick reference table by use case and language
- Detailed per-language tables with key features
- Use case deep dives (RAG progression, multi-agent, serverless)
- Technology stack overview

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (862c41e)

## [0.1.23] - 2025-12-16

## [0.1.23-rc.1] - 2025-12-16


### Fixed

- Fix: use executions table for notes storage instead of workflow_executions (#75)

* fix: use executions table for notes storage instead of workflow_executions

The note handlers (AddExecutionNoteHandler, GetExecutionNotesHandler) were
querying the workflow_executions table, but execution data is actually stored
in the executions table. This caused "execution not found" errors when adding
or retrieving notes via app.note().

Changes:
- Add Notes field to types.Execution struct
- Add notes column to ExecutionRecordModel (GORM auto-migrates this)
- Update SQL queries in execution_records.go to include notes column
- Update scanExecution to deserialize notes JSON
- Change ExecutionNoteStorage interface to use GetExecutionRecord and
  UpdateExecutionRecord instead of GetWorkflowExecution and
  UpdateWorkflowExecution
- Update AddExecutionNoteHandler to use UpdateExecutionRecord
- Update GetExecutionNotesHandler to use GetExecutionRecord

This fixes both the SDK app.note() functionality and the UI notes panel
404 errors.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* fix: update execution notes tests to use correct storage methods

Tests were using WorkflowExecution type and StoreWorkflowExecution() to set up
test data, but the handlers now use Execution type and GetExecutionRecord()/
UpdateExecutionRecord() which query the executionRecords map.

- Change test setup from types.WorkflowExecution to types.Execution
- Change StoreWorkflowExecution() to CreateExecutionRecord()
- Change GetWorkflowExecution() verification to GetExecutionRecord()
- Rename workflowID to runID to match the Execution struct field

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (5dd327e)

## [0.1.22] - 2025-12-16

## [0.1.22-rc.4] - 2025-12-16


### Fixed

- Fix: wire up workflow notes SSE endpoint (#74)

The StreamWorkflowNodeNotesHandler existed but was never registered
in the routes. This adds the missing route registration for:
GET /api/ui/v1/workflows/:workflowId/notes/events

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (c6f31cb)

## [0.1.22-rc.3] - 2025-12-16


### Added

- Feat(go-sdk): add per-request API key override for AI client (#73)

Add WithAPIKey option to override the client's configured API key on a
per-request basis. This brings the Go SDK to parity with the Python SDK,
which supports api_key overrides in individual calls.

Changes:
- Add APIKeyOverride field to Request struct (excluded from JSON)
- Add WithAPIKey option function
- Update doRequest and StreamComplete to use override when provided
- Add test for API key override behavior

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (4dd8a70)

## [0.1.22-rc.2] - 2025-12-15


### Added

- Feat(go-sdk): add Memory and Note APIs for agent state and progress tracking (#71)

Add two major new capabilities to the Go SDK:

## Memory System
- Hierarchical scoped storage (workflow, session, user, global)
- Pluggable MemoryBackend interface for custom storage
- Default in-memory backend included
- Automatic scope ID resolution from execution context

## Note API
- Fire-and-forget progress/status messages to AgentField UI
- Note(ctx, message, tags...) and Notef(ctx, format, args...) methods
- Async HTTP delivery with proper execution context headers
- Silent failure mode to avoid interrupting workflows

These additions enable agents to:
- Persist state across handler invocations within a session
- Share data between workflows at different scopes
- Report real-time progress updates visible in the UI

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (1c48c1f)

## [0.1.22-rc.1] - 2025-12-15


### Added

- Feat: allow external contributors to run functional tests without API… (#70)

* feat: allow external contributors to run functional tests without API keys

Enable external contributors to run 92% of functional tests (24/26) without
requiring access to OpenRouter API keys. This makes it easier for the community
to contribute while maintaining full test coverage for maintainers.

Changes:
- Detect forked PRs and automatically skip OpenRouter-dependent tests
- Only 2 tests require OpenRouter (LLM integration tests)
- 24 tests validate all core infrastructure without LLM calls
- Update GitHub Actions workflow to conditionally set PYTEST_ARGS
- Update functional test README with clear documentation

Test coverage for external contributors:
✅ Control plane health and APIs
✅ Agent registration and discovery
✅ Multi-agent communication
✅ Memory system (all scopes)
✅ Workflow orchestration
✅ Go/TypeScript SDK integration
✅ Serverless agents
✅ Verifiable credentials

Skipped for external contributors (maintainers still run these):
⏭️  test_hello_world_with_openrouter
⏭️  test_readme_quick_start_summarize_flow

This change addresses the challenge of running CI for external contributors
without exposing repository secrets while maintaining comprehensive test
coverage for the core AgentField platform functionality.

* fix: handle push events correctly in functional tests workflow

The workflow was failing on push events (to main/testing branches) because
it relied on github.event.pull_request.head.repo.fork which is null for
push events. This caused the workflow to incorrectly fall into the else
branch and fail when OPENROUTER_API_KEY wasn't set.

Changes:
- Check github.event_name to differentiate between push, pull_request, and workflow_dispatch
- Explicitly handle push and workflow_dispatch events to run all tests with API key
- Preserve fork PR detection to skip OpenRouter tests for external contributors

Now properly handles:
✅ Fork PRs: Skip 2 OpenRouter tests, run 24/26 tests
✅ Internal PRs: Run all 26 tests with API key
✅ Push to main/testing: Run all 26 tests with API key
✅ Manual workflow dispatch: Run all 26 tests with API key

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

* fix: remove shell quoting from PYTEST_ARGS to prevent argument parsing errors

The PYTEST_ARGS variable contained single quotes around '-m "not openrouter" -v'
which would be included in the environment variable value. When passed to pytest
in the Docker container shell command, this caused the entire string to be treated
as a single argument instead of being properly split into separate arguments.

Changed from: '-m "not openrouter" -v'
Changed to:   -m not openrouter -v

This allows the shell's word splitting to correctly parse the arguments when
pytest $$PYTEST_ARGS is evaluated in the docker-compose command.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

* refactor: separate pytest marker expression from general args for proper quoting

The previous approach of embedding -m not openrouter inside PYTEST_ARGS was
fragile because shell word-splitting doesn't guarantee "not openrouter" stays
together as a single argument to the -m flag.

This change introduces PYTEST_MARK_EXPR as a dedicated variable for the marker
expression, which is then properly quoted when passed to pytest:
  pytest -m "$PYTEST_MARK_EXPR" $PYTEST_ARGS ...

Benefits:
- Marker expression is guaranteed to be treated as single argument to -m
- Clear separation between marker selection and general pytest args
- More maintainable for future marker additions
- Eliminates shell quoting ambiguity

Changes:
- workflow: Split PYTEST_ARGS into PYTEST_MARK_EXPR + PYTEST_ARGS
- docker-compose: Add PYTEST_MARK_EXPR env var and conditional -m flag
- docker-compose: Only apply -m when PYTEST_MARK_EXPR is non-empty

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

* fix: add proper event type checks before accessing pull_request context

Prevent errors when workflow runs on push events by:
- Check event_name == 'pull_request' before accessing pull_request.head.repo.fork
- Check event_name == 'workflow_dispatch' before accessing event.inputs
- Ensures all conditional expressions only access context properties when they exist

This prevents "Error: Cannot read properties of null (reading 'fork')" errors
on push events.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

---------

Co-authored-by: Claude Sonnet 4.5 <noreply@anthropic.com> (01668aa)



### Fixed

- Fix(python-sdk): move conditional imports to module level (#72)

The `serve()` method had `import os` and `import urllib.parse` statements
inside conditional blocks. When an explicit port was passed, the first
conditional block was skipped, but Python's scoping still saw the later
conditional imports, causing an `UnboundLocalError` when trying to use
`os.getenv()` at line 1140.

Error seen in Docker containers:
```
UnboundLocalError: cannot access local variable 'os' where it is not
associated with a value
```

This worked locally because `auto_port=True` executed the first code path
which included `import os`, but failed in Docker when passing an explicit
port value.

Fix: Move all imports to module level where they belong.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (a0d0538)

## [0.1.21] - 2025-12-14

## [0.1.21-rc.3] - 2025-12-14


### Other

- Test pr 68 init fix (#69)

* fix(cli): fix init command input handling issues

- Fix j/k keys not registering during text input
- Fix ctrl+c not cancelling properly
- Fix selected option shifting other items
- Filter special keys from text input
- Add ctrl+u to clear input line
- Add unit tests for init model

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* docs: add changelog entry for CLI init fixes

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* chore: trigger CI with secrets

* chore: remove manual changelog entry (auto-generated on release)

---------

Co-authored-by: fimbulwinter <sanandsankalp@gmail.com>
Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com> (55d0c61)

## [0.1.21-rc.2] - 2025-12-10


### Fixed

- Fix: correct parent execution ID for sub-calls in app.call() (#62)

When a reasoner calls a skill via app.call(), the X-Parent-Execution-ID
  header was incorrectly set to the inherited parent instead of the current
  execution. This caused workflow graphs to show incorrect parent-child
  relationships.

  The fix overrides X-Parent-Execution-ID to use the current execution's ID
  after to_headers() is called, ensuring sub-calls are correctly attributed
  as children of the calling execution.

Co-authored-by: Ivan Viljoen <8543825+ivanvza@users.noreply.github.com> (762142e)



### Other

- Update README to remove early adopter notice

Removed early adopter section from README. (054fc22)

- Update README.md (dae57c7)

- Update README.md (06e5cee)

- Update README.md (39c2da4)

## [0.1.21-rc.1] - 2025-12-06


### Other

- Add serverless agent examples and functional tests (#46)

* Add serverless agent examples and functional tests

* Add CLI support for serverless node registration

* Fix serverless execution payload initialization

* Harden serverless functional test to use CLI registration

* Broaden serverless CLI functional coverage

* Persist serverless invocation URLs

* Ensure serverless executions hit /execute

* Fix serverless agent metadata loading

* Derive serverless deployment for stored agents

* Honor serverless metadata during execution

* Backfill serverless invocation URLs on load

* Stabilize serverless agent runtime

* Harden serverless functional harness

* Support serverless agents via reasoners endpoint

* Log serverless reasoner responses for debugging

* Allow custom serverless adapters across SDKs

* Normalize serverless handler responses

* Fix Python serverless adapter typing

* Make serverless adapter typing py3.9-safe

* Fix Python serverless execution context

* Simplify Python serverless calls to sync

* Mark serverless Python agents connected for cross-calls

* Force sync execution path in serverless handler

* Handle serverless execute responses without result key

* Align serverless Python relay args with child signature

* feat: Add workflow performance visualizations, including agent health heatmap and execution scatter plot, and enhance UI mobile responsiveness.

* chore: Remove unused Badge import from ExecutionScatterPlot.tsx and add an empty line to .gitignore. (728e4e0)

- Added docker (74f111b)

- Update README.md (8b580cb)

## [0.1.20] - 2025-12-04

## [0.1.20-rc.3] - 2025-12-04


### Fixed

- Fix(sdk/typescript): add DID registration to enable VC generation (#60)

* fix(release): skip example requirements for prereleases

Restore the check to skip updating example requirements for prerelease
versions. Even though prereleases are now published to PyPI, pip install
excludes them by default per PEP 440. Users running `pip install -r
requirements.txt` would fail without the `--pre` flag.

Examples should always pin to stable versions so they work out of the box.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>

* fix(sdk/typescript): add DID registration to enable VC generation

The TypeScript SDK was not registering with the DID system, causing VC
generation to fail with "failed to resolve caller DID: DID not found".

This change adds DID registration to match the Python SDK's behavior:

- Add DIDIdentity types and registerAgent() to DidClient
- Create DidManager class to store identity package after registration
- Integrate DidManager into Agent.ts to auto-register on startup
- Update getDidInterface() to resolve DIDs from stored identity package

When didEnabled is true, the agent now:
1. Registers with /api/v1/nodes/register (existing)
2. Registers with /api/v1/did/register (new)
3. Stores identity package for DID resolution
4. Auto-populates callerDid/targetDid when generating VCs

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>

* feat(examples): add verifiable credentials TypeScript example

Add a complete VC example demonstrating:
- Basic text processing with explicit VC generation
- AI-powered analysis with VC audit trail
- Data transformation with integrity proof
- Multi-step workflow with chained VCs

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>

* fix(examples): fix linting errors in VC TypeScript example

- Remove invalid `note` property from workflow.progress calls
- Simplify AI response handling since schema already returns parsed type

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>

---------

Co-authored-by: Claude <noreply@anthropic.com> (bd097e1)

- Fix(release): skip example requirements for prereleases (#59)

Restore the check to skip updating example requirements for prerelease
versions. Even though prereleases are now published to PyPI, pip install
excludes them by default per PEP 440. Users running `pip install -r
requirements.txt` would fail without the `--pre` flag.

Examples should always pin to stable versions so they work out of the box.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude <noreply@anthropic.com> (1b7d9b8)

## [0.1.20-rc.2] - 2025-12-04


### Added

- Feat(release): unify PyPI publishing for all releases (#58)

Publish all Python SDK releases (both prerelease and stable) to PyPI
instead of using TestPyPI for prereleases.

Per PEP 440, prerelease versions (e.g., 0.1.20rc1) are excluded by
default from `pip install` - users must explicitly use `--pre` flag.
This simplifies the release process and removes the need for the
TEST_PYPI_API_TOKEN secret.

Changes:
- Merge TestPyPI and PyPI publish steps into single PyPI step
- Update release notes to show `pip install --pre` for staging
- Update install.sh staging output
- Re-enable example requirements updates for prereleases
- Update RELEASE.md documentation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-authored-by: Claude <noreply@anthropic.com> (ebf7020)



### Fixed

- Fix(release): fix example requirements and prevent future staging bumps (#56)

* fix(examples): revert to stable agentfield version (0.1.19)

The staging release bumped example requirements to 0.1.20-rc.1, but
RC versions are published to TestPyPI, not PyPI. This caused Railway
deployments to fail because pip couldn't find the package.

Revert to the last stable version (0.1.19) which is available on PyPI.

* fix(release): skip example requirements bump for prerelease versions

Prerelease versions are published to TestPyPI, not PyPI. If we bump
example requirements.txt files to require a prerelease version,
Railway deployments will fail because pip looks at PyPI by default.

Now bump_version.py only updates example requirements for stable
releases, ensuring deployed examples always use versions available
on PyPI. (c86bec5)

## [0.1.20-rc.1] - 2025-12-04


### Added

- Feat(release): add two-tier staging/production release system (#53)

* feat(release): add two-tier staging/production release system

Implement automatic staging releases and manual production releases:

- Staging: Automatic on push to main (PyPI prerelease, npm @next, staging-* Docker)
- Production: Manual workflow dispatch (PyPI, npm @latest, vX.Y.Z + latest Docker)

Changes:
- Add push trigger with path filters for automatic staging
- Replace release_channel with release_environment input
- Unified PyPI publishing for both staging (prerelease) and production
- Split npm publishing: @next tag (staging) vs @latest (production)
- Conditional Docker tagging: staging-X.Y.Z vs vX.Y.Z + latest
- Add install-staging.sh for testing prerelease binaries
- Update RELEASE.md with two-tier documentation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>

* refactor(install): consolidate staging into single install.sh with --staging flag

Instead of separate install.sh and install-staging.sh scripts:
- Single install.sh handles both production and staging
- Use --staging flag or STAGING=1 env var for prerelease installs
- Eliminates code drift between scripts

Usage:
  Production: curl -fsSL .../install.sh | bash
  Staging:    curl -fsSL .../install.sh | bash -s -- --staging

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>

---------

Co-authored-by: Claude <noreply@anthropic.com> (3bd748d)

- Feat(sdk/typescript): expand AI provider support to 10 providers

Add 6 new AI providers to the TypeScript SDK:
- Google (Gemini models)
- Mistral AI
- Groq
- xAI (Grok)
- DeepSeek
- Cohere

Also add explicit handling for OpenRouter and Ollama with sensible defaults.

Changes:
- Update AIConfig type with new provider options
- Refactor buildModel() with switch statement for all providers
- Refactor buildEmbeddingModel() with proper embedding support
  (Google, Mistral, Cohere have native embedding; others throw)
- Add 27 unit tests for provider selection and embedding support
- Install @ai-sdk/google, @ai-sdk/mistral, @ai-sdk/groq,
  @ai-sdk/xai, @ai-sdk/deepseek, @ai-sdk/cohere packages

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (b06b5b5)



### Other

- Update versions (a7912f5)

## [0.1.19] - 2025-12-04


### Fixed

- Fix(ui): add API key header to sidebar execution details fetch

The useNodeDetails hook was making a raw fetch() call without including
the X-API-Key header, causing 401 errors in staging where API key
authentication is enabled. Other API calls in the codebase use
fetchWrapper functions that properly inject the key. (f0ec542)

## [0.1.18] - 2025-12-03


### Fixed

- Fix(sdk): inject API key into all HTTP requests

The Python SDK was not including the X-API-Key header in HTTP requests
made through AgentFieldClient._async_request(), causing 401 errors when
the control plane has authentication enabled.

This fix injects the API key into request headers automatically when:
- The client has an api_key configured
- The header isn't already set (avoids overwriting explicit headers)

Fixes async status updates and memory operations (vector search, etc.)
that were failing with 401 Unauthorized.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (97673bc)

## [0.1.17] - 2025-12-03


### Fixed

- Fix(control-plane): remove redundant WebSocket origin check

The WebSocket upgrader's CheckOrigin was rejecting server-to-server
connections (like from Python SDK agents) that don't have an Origin
header. This caused 403 errors when agents tried to connect to memory
events WebSocket endpoint with auth enabled.

The origin check was redundant because:
1. Auth middleware already validates API keys before this handler
2. If auth is enabled, only valid API key holders reach this point
3. If auth is disabled, all connections are allowed anyway

Removes the origin checking logic and simplifies NewMemoryEventsHandler
to just take the storage provider.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (44f05c4)

## [0.1.16] - 2025-12-03


### Fixed

- Fix(example): use IPv4 binding for documentation-chatbot

The documentation chatbot was binding to `::` (IPv6 all interfaces) which
causes Railway internal networking to fail with "connection refused" since
Railway routes traffic over IPv4.

Removed explicit host parameter to use the SDK default of `0.0.0.0` which
binds to IPv4 all interfaces.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (2c1b205)

- Fix(python-sdk): include API key in memory events WebSocket connections

The MemoryEventClient was not including the X-API-Key header when
connecting to the memory events WebSocket endpoint, causing 401 errors
when the control plane has authentication enabled.

Changes:
- Add optional api_key parameter to MemoryEventClient constructor
- Include X-API-Key header in WebSocket connect() method
- Include X-API-Key header in history() method (both httpx and requests)
- Pass api_key from Agent to MemoryEventClient in both instantiation sites

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (eda95fc)



### Other

- Revert "fix(example): use IPv4 binding for documentation-chatbot"

This reverts commit 2c1b2053e37f4fcc968ad0805b71ef89cf9d6d9d. (576a96c)

## [0.1.15] - 2025-12-03


### Fixed

- Fix(python-sdk): update test mocks for api_key parameter

Update test helpers and mocks to accept the new api_key parameter:
- Add api_key field to StubAgent dataclass
- Add api_key parameter to _FakeDIDManager and _FakeVCGenerator
- Add headers parameter to VC generator test mocks

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (301e276)

- Fix(python-sdk): add missing API key headers to DID/VC and workflow methods

Comprehensive fix for API key authentication across all SDK HTTP requests:

DID Manager (did_manager.py):
- Added api_key parameter to __init__
- Added _get_auth_headers() helper method
- Fixed register_agent() to include X-API-Key header
- Fixed resolve_did() to include X-API-Key header

VC Generator (vc_generator.py):
- Added api_key parameter to __init__
- Added _get_auth_headers() helper method
- Fixed generate_execution_vc() to include X-API-Key header
- Fixed verify_vc() to include X-API-Key header
- Fixed get_workflow_vc_chain() to include X-API-Key header
- Fixed create_workflow_vc() to include X-API-Key header
- Fixed export_vcs() to include X-API-Key header

Agent Field Handler (agent_field_handler.py):
- Fixed _send_heartbeat() to include X-API-Key header

Agent (agent.py):
- Fixed emit_workflow_event() to include X-API-Key header
- Updated _initialize_did_system() to pass api_key to DIDManager and VCGenerator

All HTTP requests to AgentField control plane now properly include authentication headers when API key is configured.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (2517549)

- Fix(python-sdk): add missing API key headers to sync methods

Add authentication headers to register_node(), update_health(), and
get_nodes() methods that were missing X-API-Key headers in requests.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (0c2977d)



### Other

- Add Go SDK CallLocal workflow tracking (64c6217)

- Fix Python SDK to include API key in register/heartbeat requests

The SDK's AgentFieldClient stored the api_key but several methods were
not including it in their HTTP requests, causing 401 errors when
authentication is enabled on the control plane:

- register_agent()
- register_agent_with_status()
- send_enhanced_heartbeat() / send_enhanced_heartbeat_sync()
- notify_graceful_shutdown() / notify_graceful_shutdown_sync()

Also updated documentation-chatbot example to pass AGENTFIELD_API_KEY
from environment to the Agent constructor.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (1e6a095)

## [0.1.14] - 2025-12-03


### Added

- Feat: expose api_key at Agent level and fix test lint issues

- Add api_key parameter to Agent class constructor
- Pass api_key to AgentFieldClient for authentication
- Document api_key parameter in Agent docstring
- Fix unused loop variable in ensure_event_loop test fixture

Addresses reviewer feedback that api_key should be exposed at Agent
level since end users don't interact directly with AgentFieldClient.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (6567bd0)

- Feat: add API key authentication to control plane and SDKs

This adds optional API key authentication to the AgentField control plane
with support in all SDKs (Python, Go, TypeScript).

## Control Plane Changes

- Add `api_key` config option in agentfield.yaml
- Add HTTP auth middleware (X-API-Key header, Bearer token, query param)
- Add gRPC auth interceptor (x-api-key metadata, Bearer token)
- Skip auth for /api/v1/health, /metrics, and /ui/* paths
- UI prompts for API key when auth is required and stores in localStorage

## SDK Changes

- Python: Add `api_key` parameter to AgentFieldClient
- Go: Add `WithAPIKey()` option to client
- TypeScript: Add `apiKey` option to client config

## Tests

- Add comprehensive HTTP auth middleware tests (14 tests)
- Add gRPC auth interceptor tests (11 tests)
- Add Python SDK auth tests (17 tests)
- Add Go SDK auth tests (10 tests)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (3f8e45c)



### Fixed

- Fix: resolve flaky SSE decoder test in Go SDK

- Persist accumulated buffer across Decode() calls in SSEDecoder
- Check for complete messages in buffer before reading more data
- Add synchronization in test to prevent handler from closing early
- Update test expectation for multiple chunks (now correctly returns 2)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (32d6d6d)

- Fix: update test helper to accept api_key parameter

Update _FakeAgentFieldClient and _agentfield_client_factory to accept
the new api_key parameter that was added to AgentFieldClient.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (092f8e0)

- Fix: remove unused import and variable in test_client_auth

- Remove unused `requests` import
- Remove unused `result` variable assignment

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (8b93711)

- Fix: stop reasoner raw JSON editor from resetting (c604833)

- Fix(ci): add packages:write permission to publish job for GHCR push

The publish job had its own permissions block that overrode the
workflow-level permissions. Added packages:write to allow Docker
image push to ghcr.io.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (269ac29)



### Other

- Updated favcoin (d1712c2)



### Testing

- Test: add tests for Agent and AgentRouter api_key exposure

- Test Agent stores api_key and passes it to client
- Test Agent works without api_key
- Test AgentRouter delegates api_key to attached agent
- Test AgentRouter delegates client to attached agent
- Test unattached router raises RuntimeError

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (31cd0b1)

## [0.1.13] - 2025-12-02


### Other

- Release workflow fix (fde0309)

- Update README.md (c3cfca4)

## [0.1.12] - 2025-12-02


### Chores

- Chore: trigger Railway deployment for PR #39 fix (b4095d2)



### Documentation

- Docs(chatbot): add SDK search term relationship

Add search term mapping for SDK/language queries to improve RAG
retrieval when users ask about supported languages or SDKs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (87a4d90)

- Docs(chatbot): add TypeScript SDK to supported languages

Update product context to include TypeScript alongside Python and Go:
- CLI commands now mention all three language options
- Getting started section references TypeScript
- API Reference includes TypeScript SDK

This fixes the RAG chatbot returning only Python/Go when asked about
supported languages.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (9510d74)



### Fixed

- Fix(vector-store): fix PostgreSQL DeleteByPrefix and update namespace defaults

- Fix DeleteByPrefix to use PostgreSQL || operator for LIKE pattern
  (the previous approach with prefix+"%" in Go wasn't working correctly
  with parameter binding)
- Change default namespace from "documentation" to "website-docs" to
  match the frontend chat API expectations
- Add scope: "global" to clear_namespace API call to ensure proper
  scope matching

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (cbfdf7b)

- Fix(docs-chatbot): use correct start command

Change start command from `python -m agentfield.run` (doesn't exist)
to `python main.py` (the actual entry point).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (b71507c)

- Fix(docs-chatbot): override install phase for PyPI wait

The previous fix used buildCommand which runs AFTER pip install.
This fix overrides the install phase itself:

- Add nixpacks.toml with [phases.install] to run install.sh
- Update railway.json to point to nixpacks.toml
- Update install.sh to create venv before waiting for PyPI

The issue was that buildCommand runs after the default install phase,
so pip had already failed before our script ran.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (f8bf14b)

- Fix(docs-chatbot): use railway.json for Railpack PyPI wait

Railway now uses Railpack instead of Nixpacks. Update config:
- Replace nixpacks.toml with railway.json
- Force NIXPACKS builder with custom buildCommand
- Fix install.sh version check using pip --dry-run

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (8c22356)

## [0.1.11] - 2025-12-02


### Fixed

- Fix(docs-chatbot): handle PyPI race condition in Railway deploys

Add install script that waits for agentfield package to be available
on PyPI before installing. This fixes the race condition where Railway
deployment triggers before the release workflow finishes uploading to PyPI.

- Add install.sh with retry logic (30 attempts, 10s intervals)
- Add nixpacks.toml to use custom install script

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (e45f41d)

## [0.1.10] - 2025-12-02


### Added

- Feat: add delete-namespace endpoint for RAG reindexing

Adds a new DELETE /api/v1/memory/vector/namespace endpoint that allows
clearing all vectors with a given namespace prefix. This enables the
documentation chatbot to wipe and reindex its RAG data when docs change.

Changes:
- Add DeleteVectorsByPrefix to StorageProvider interface
- Implement DeleteByPrefix for SQLite and Postgres vector stores
- Add DeleteNamespaceVectorsHandler endpoint
- Add clear_namespace skill to documentation chatbot
- Update MemoryStorage interface with new method

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (bc1f41e)

- Feat(sdk-python): expose execution context via app.ctx property

Add a `ctx` property to the Agent class that provides direct access to
the current execution context during reasoner/skill execution. This
enables a more ergonomic API:

Before:
  from agentfield.execution_context import get_current_context
  ctx = get_current_context()
  workflow_id = ctx.workflow_id

After:
  workflow_id = app.ctx.workflow_id

The property returns None when accessed outside of an active execution
(e.g., at module level or after a request completes), matching the
behavior of app.memory. This prevents accidental use of stale or
placeholder context data.

Also fixes integration test fixtures to support the current monorepo
structure where control-plane lives at repo root instead of
apps/platform/agentfield.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (e01dcea)

- Feat(ts-sdk): add DID client and memory helpers (4b74998)

- Feat(ts-sdk): add heartbeat and local call coverage (cf228ec)

- Feat(ts-sdk): scaffold typescript sdk core (09dcc62)



### Chores

- Chore: ignore env files (3937821)

- Chore(ts-sdk): align heartbeat and memory clients, improve example env loading (fee2a7e)

- Chore(ts-sdk): load env config for simulation example (9715ac5)

- Chore(ts-sdk): remove AI stubs from simulation example (7b94190)

- Chore(ts-sdk): make simulation example runnable via build (9a87374)

- Chore(ts-sdk): fix typings, add heartbeat config, lock deps (f9af207)



### Fixed

- Fix: revert conftest changes to prevent CI failures

The integration tests should skip gracefully in CI when the control
plane cannot be built. Reverting conftest changes that caused the
tests to attempt building when they should skip.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (f86794c)

- Fix: remove unused import to pass linting

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (5a975fa)

- Fix flaky tests (bfb86cb)

- Fix(ts-sdk): normalize router IDs to align with control plane (7c36c8b)

- Fix(ts-sdk): register full reasoner definitions (e5cc44d)



### Other

- Ts sdk (ce3b965)

- Recover agent state on restart and speed up node status (7fa12ca)

- Remove unused configuration variables

Audit of agentfield.yaml revealed many config options that were defined
but never actually read or used by the codebase. This creates confusion
for users who set these values expecting them to have an effect.

Removed from YAML config:
- agentfield: mode, max_concurrent_requests, request_timeout,
  circuit_breaker_threshold (none were wired to any implementation)
- execution_queue: worker_count, request_timeout, lease_duration,
  max_attempts, failure_backoff, max_failure_backoff, poll_interval,
  result_preview_bytes, queue_soft_limit, waiter_map_limit
- ui: backend_url
- storage.local: cache_size, retention_days, auto_vacuum
- storage: config field
- agents section entirely (discovery/scaling never implemented)

Removed from Go structs:
- AgentsConfig, DiscoveryConfig, ScalingConfig
- CoreFeatures, EnterpriseFeatures
- DataDirectoriesConfig
- Unused fields from AgentFieldConfig, ExecutionQueueConfig,
  LocalStorageConfig, StorageConfig, UIConfig

The remaining config options are all actively used:
- agentfield.port, execution_cleanup.*, execution_queue webhook settings
- ui.enabled/mode/dev_port
- api.cors.*
- storage.mode/local.database_path/local.kv_store_path/vector.*
- features.did.* (all DID/VC settings)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (ee6e6e0)

- Adds more links to documentation

Adds several new links to the README.md file that direct users to more detailed documentation pages. These links cover production-ready features, comparisons with agent frameworks, the full feature set, and the core architecture. (d5a9922)

- Update documentation links

Updates several external links within the README to point to the correct documentation paths.

This ensures that users can navigate to the relevant guides and information seamlessly. (ac6f777)

- Updated arch (4ed9806)

- Improve README Quick Start guide

Updates the README's quick start section to provide a more comprehensive and user-friendly guide.

This revision clarifies the installation process, introduces a dedicated step for agent creation with a default configuration option using `af init --defaults`, and specifies the necessary command-line instructions for each terminal in the control plane + agent node architecture.

It also refines the example API call to use a more descriptive agent endpoint (`my-agent.demo_echo`) and adds examples for Go and TypeScript, as well as detailing how to use interactive mode for agent initialization. (4e897f0)

- Refactor README for clarity and expanded content

Updates the README to provide a more detailed explanation of AgentField's purpose and features.

Key changes include:
- Enhanced "What is AgentField?" section to emphasize its role as backend infrastructure for autonomous AI.
- Improved "Quick Start" section with clearer steps and usage examples.
- Expanded "Build Agents in Any Language" section to showcase Python, Go, TypeScript, and REST API examples.
- Introduced new sections like "The Production Gap" and "Identity & Trust" to highlight AgentField's unique value proposition.
- Refined "Who is this for?" and "Is AgentField for you?" sections for better audience targeting.
- Updated navigation links and visual elements for improved readability and user experience. (f05cd95)

- Typescript schema based formatting improvements (fcda991)

- Typescript release and init (218326b)

- Functional tests (99b6f9e)

- Add TS SDK CI and functional TS agent coverage (857191d)

- Add MCP integration (5bc36d7)

- Separate example freom sdk (909dc8c)

- Memory & Discovery (84ff093)

- TS SDK simulation flow working (5cab496)

- Add .env to git ignore (172e8a9)

- Update README.md (4e0b2e6)

- Fix MemoryEventClient init for sync contexts (1d246ec)

- Fix memory event client concurrency and compatibility (2d28571)

- Improve LLM prompt formatting and citations

Refactors the system and user prompts for the documentation chatbot to improve clarity and LLM performance. This includes:

- Restructuring and clarifying the prompt instructions for citations, providing explicit guidance on how to use and format them.
- Enhancing the citation key map format to be more descriptive and user-friendly for the LLM.
- Explicitly stating that the `citations` array in the response should be left empty by the LLM, as it will be injected by the system.
- Updating the `Citation` schema to correctly reflect that the `key` should not include brackets.
- Adding a specific "REFINEMENT MODE" instruction to the refined prompt to guide the LLM's behavior in a second retrieval attempt.
- Minor cleanup and adjustments to prompt text for better readability. (56246ad)

- Update dependencies for improved compatibility

Updates several npm package dependencies, including browserslist, caniuse-lite, and electron-to-chromium, to their latest versions.
This ensures better compatibility and incorporates recent improvements and bug fixes from these packages. (c72278c)

- Implement automatic agent method delegation

Improves the AgentRouter by implementing __getattr__ to automatically delegate any unknown attribute or method access to the attached agent. This eliminates the need for explicit delegation methods for agent functionalities like `ai()`, `call()`, `memory`, `note()`, and `discover()`.

This change simplifies the AgentRouter's interface and makes it more transparently proxy agent methods. Added tests to verify the automatic delegation for various agent methods and property access, as well as error handling when no agent is attached. (26c9288)



### Testing

- Tests hanging fix (dd2eb8d)

## [0.1.9] - 2025-11-25


### Other

- Un-hardcode agent request timeout (4b9789f)

- Remove --import-mode=importlib from pytest config

This flag was causing issues with functional tests in postgres mode.
The Python 3.8 PyO3 issue is already fixed by disabling coverage
for Python 3.8 in the CI workflow.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (629962e)

- Fix linting: Remove unused concurrent.futures import

The import was not needed for run_in_executor.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (6855ff9)

- Add Python 3.8 compatibility for asyncio.to_thread

asyncio.to_thread was added in Python 3.9. This commit adds a
compatibility shim using loop.run_in_executor for Python 3.8.

Fixes test failures:
- test_execute_async_falls_back_to_requests
- test_set_posts_payload
- test_async_request_falls_back_to_requests
- test_memory_round_trip

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (93031f0)

- Fix Python 3.8 CI: Disable coverage for Python 3.8

The PyO3 modules in pydantic-core can only be initialized once per
interpreter on Python 3.8. pytest-cov causes module reimports during
coverage collection, triggering this limitation.

Solution:
- Keep --import-mode=importlib for better import handling
- Disable coverage collection (--no-cov) only for Python 3.8 in CI
- Coverage still collected for Python 3.9-3.12

This is a known compatibility issue with PyO3 + Python 3.8 + pytest-cov.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (c97af63)

- Fix Python 3.8 CI: Add --import-mode=importlib to pytest config

Resolves PyO3 ImportError on Python 3.8 by configuring pytest to use
importlib import mode. This prevents PyO3 modules (pydantic-core) from
being initialized multiple times, which causes failures on Python 3.8.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (78f95b2)

- Fix linting error: Remove unused Dict import from pydantic_utils

The Dict type from typing was imported but never used in the file.
This was causing the CI to fail with ruff lint error F401.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (1e52294)

- Add Python 3.8+ support to Python SDK

Lower the minimum Python version requirement from 3.10 to 3.8 to improve
compatibility with systems running older Python versions.

Changes:
- Update pyproject.toml to require Python >=3.8
- Add Python 3.8, 3.9 to package classifiers
- Fix type hints incompatible with Python 3.8:
  - Replace list[T] with List[T]
  - Replace dict[K,V] with Dict[K,V]
  - Replace tuple[T,...] with Tuple[T,...]
  - Replace set[T] with Set[T]
  - Replace str | None with Optional[str]
- Update CI to test on Python 3.8, 3.9, 3.10, 3.11, 3.12
- Update documentation to reflect Python 3.8+ requirement

All dependencies (FastAPI, Pydantic v2, litellm, etc.) support Python 3.8+.
Tested and verified on Python 3.8.18.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com> (d797fc4)

- Update doc url (dc6f361)

- Fix README example: Use AIConfig for model configuration

- Changed from incorrect Agent(node_id='researcher', model='gpt-4o')
- To correct Agent(node_id='researcher', ai_config=AIConfig(model='gpt-4o'))
- Added AIConfig import to the example
- Model configuration should be passed through ai_config parameter, not directly to Agent (34bf018)

- Removes MCP documentation section

Removes the documentation section detailing the Model Context Protocol (MCP).
This section is no longer relevant to the current project structure. (3361f8c)

## [0.1.8] - 2025-11-23


### Other

- Automate changelog generation with git-cliff

Integrates git-cliff into the release workflow to automatically generate changelog entries from commit history. This streamlines the release process by eliminating manual changelog updates.

The CONTRIBUTING.md file has been updated to reflect this new process and guide contributors on how to structure their commits for effective changelog generation. A new script, `scripts/update_changelog.py`, is called to perform the changelog update during the release process. (d3e1146)

- Refactors agent AI token counting and trimming

Replaces lambda functions for `token_counter` and `trim_messages` with explicit function definitions in `AgentAI` to improve clarity and maintainability.

Additionally, this commit removes an unused import in `test_discovery_api.py` and cleans up some print statements and a redundant context manager wrapper in `test_go_sdk_cli.py` and `test_hello_world.py` respectively. (7880ff3)

- Remove unused Generator import

Removes the `Generator` type hint from the imports in `conftest.py`, as it is no longer being used. This is a minor cleanup to reduce unnecessary imports. (7270ce8)

- Final commit (1aa676e)

- Add discovery API endpoint

Introduces a new endpoint to the control plane for discovering agent capabilities.
This includes improvements to the Python SDK to support querying and parsing discovery results.

- Adds `InvalidateDiscoveryCache()` calls in node registration handlers to ensure cache freshness.
- Implements discovery routes in the control plane server.
- Enhances the Python SDK with `discover` method, including new types for discovery responses and improved `Agent` and `AgentFieldClient` classes.
- Refactors `AsyncExecutionManager` and `ResultCache` for lazy initialization of asyncio objects and `shutdown_event`.
- Adds new types for discovery API responses in `sdk/python/agentfield/types.py`.
- Introduces unit tests for the new `discover_capabilities` functionality in the client. (ab2417b)

- Updated (6f1f58d)

- Initial prd (4ed1ea5)

- Adds decorator-based API for global memory event listeners

Introduces a decorator to simplify subscribing to global memory change events,
enabling more readable and maintainable event-driven code.

Enhances test coverage by verifying event listener patterns via functional tests,
ensuring decorators correctly capture events under various scenarios. (608b8c6)

- Update functional tests and docker configuration

- Remove PRD_GO_SDK_CLI.md document
- Update docker compose configurations for local and postgres setups
- Modify test files for Go SDK CLI and memory events (4fa2bb7)

- Adds CLI support and configuration to agent module

Introduces options for registering CLI-accessible handlers, custom CLI formatting, and descriptions.
Adds a configuration struct for CLI behavior and presentation.
Refactors agent initialization to allow operation without a server URL in CLI mode.
Improves error handling and test coverage for new CLI logic. (54f483b)

- Prd doc (d258e72)

- Update README.md (3791924)

- Update README.md (b4bca5e)



### Testing

- Testing runs functional test still not working id errors (6da01e6)

## [0.1.2] - 2025-11-12
### Fixed
- Control-plane Docker image now builds with CGO enabled so SQLite works in containers like Railway.

## [0.1.1] - 2025-11-12
### Added
- Documentation chatbot + advanced RAG examples showcasing Python agent nodes.
- Vector memory storage backends and skill test scaffolding for SDK examples.

### Changed
- Release workflow improvements (selective publishing, prerelease support) and general documentation updates.

## [0.1.0] - 2024-XX-XX
### Added
- Initial open-source release with control plane, Go SDK, Python SDK, and deployment assets.

### Changed
- Cleaned repository layout for public distribution.

### Removed
- Private experimental artifacts and internal operational scripts.
