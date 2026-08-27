# Worker E report

## What changed and why

- Go agents now resolve a configurable 30-second graceful-shutdown timeout, notify the control plane first, stop accepting async dispatches, wait for detached 202 executions, and cancel them through their execution contexts on timeout.
- Go agents expose `POST /shutdown` with the Python-compatible body and unblock `Serve` after remote shutdown; `Config.ShutdownTimeout` overrides the shared environment variable.
- TypeScript `serve()` installs SIGTERM/SIGINT handlers by default with `handleSignals: false` opt-out, while idempotent `shutdown()` notifies the control plane, closes the listener, drains tracked detached executions, and cancels on timeout.
- Added the TypeScript node-shutdown client call and shared timeout parsing (`30`, `30s`, `5m`, invalid warning/default).
- Documented the shared Python/Go/TypeScript environment variable and Kubernetes termination-grace guidance in both SDK READMEs and the environment-variable reference.

## Round 2 reviewer fixes

- Go graceful HTTP and execution draining now share one deadline, so the configured budget is not applied twice. Immediate shutdown uses `server.Close()`, and HTTP shutdown errors are logged.
- Go cancellation now has a bounded five-second post-cancel settlement window; shutdown proceeds and logs the number of abandoned executions if a reasoner ignores cancellation.
- Added real-Agent/httptest coverage proving a 202 execution reports success before shutdown returns and a 200 ms timeout cancels a cooperative reasoner, reports a terminal status, and returns within one second.
- TypeScript now tracks detached work as `Map<executionId, Promise>`, cancels every in-flight ID on timeout (including non-paused work), cancels pauses, and awaits settlement.
- Default TypeScript signal handlers now call `process.exit(143)` for SIGTERM or `process.exit(130)` for SIGINT after shutdown resolves; programmatic `shutdown()` remains free of process exit.
- TypeScript `shutdown()` is a plain promise-returning method so repeated calls return the identical promise, with focused tests for ordering, listener ownership/removal, non-paused cancellation/settlement, and promise identity.

## Round 3 coverage fixes

- Added focused Go coverage for rejecting detached reasoner requests after shutdown begins, returning from `Serve` after remote immediate shutdown, and the `server.Close()` path.
- Made the post-cancel settlement bound package-configurable for tests and verified that a reasoner ignoring cancellation returns promptly while logging one abandoned execution.
- Added an assertion for the invalid `AGENTFIELD_SHUTDOWN_TIMEOUT` warning.
- Exact patch gate result: `| sdk-go | 89 | 88.00% | ✅ |` (80% required).

## Gates

- `./scripts/coverage-summary.sh && ./scripts/patch-coverage-gate.sh` — PASS (`sdk-go`: 88.00% on 89 touched lines).
- `cd sdk/go && go test ./... -count=1` — PASS (8 packages).
- `cd sdk/go && go mod tidy && go build ./... && go test ./... -count=1` — PASS (8 packages; agent package 36.797s).
- `cd sdk/go && go test ./agent -run 'TestShutdownWaits|TestShutdownTimeoutCancels' -count=1` — PASS (2 tests).
- `cd sdk/typescript && npm ci` — PASS (252 packages installed; npm reported one low-severity audit finding).
- `cd sdk/typescript && npm run build` — PASS (ESM and declarations).
- `cd sdk/typescript && npm test` — PASS (87 files, 936 tests).
- `gofmt -l $(git diff --name-only origin/main -- '*.go')` — PASS (no output).
- `git diff --check` — PASS (no output).

## Validation contract mapping

- Go timeout grammar/default/warning: `TestResolveShutdownTimeout`.
- Go `/shutdown` compatibility and immediate acceptance: `TestShutdownRouteAccepted`.
- Go accepted 202 completion and success callback before shutdown returns: `TestShutdownWaitsForAcceptedAsyncExecutionTerminalStatus`.
- Go deadline cancellation, bounded return, and terminal callback: `TestShutdownTimeoutCancelsAcceptedAsyncExecutionAndReportsTerminalStatus`.
- TypeScript timeout grammar/default/warning: `parseShutdownTimeout` tests in `tests/signals.test.ts`.
- TypeScript notify-close-drain ordering: `notifies before closing the server and waits for in-flight executions to settle`.
- TypeScript signal opt-out/default installation/removal: `serve signal handling is opt-out and shutdown removes default listeners`.
- TypeScript cancellation of non-paused work and post-cancel settlement: `timeout cancels a non-paused in-flight execution and awaits settlement`.
- TypeScript shutdown promise identity: `returns the same promise when shutdown is called twice`.

## Deviations from spec

- None.

## Found, not fixed

- `npm ci` reports one low-severity dependency vulnerability; dependency updates are outside this fix cluster.

## Reviewer decisions

- None.
