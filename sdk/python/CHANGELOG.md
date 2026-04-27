# Changelog

## [0.1.3] - 2025-11-12
### Changed
- Replaced generic `RuntimeError`/`Exception`/`TimeoutError` raises with the typed `AgentFieldError` hierarchy (`AgentFieldClientError`, `ExecutionTimeoutError`, etc.) across `agent`, `async_execution_manager`, `http_connection_manager`, `router`, and `utils`.

### Breaking
- `AsyncExecutionManager.wait_for_result` now raises `ExecutionTimeoutError` instead of the builtin `TimeoutError`, and `AgentFieldClientError` instead of `RuntimeError`, on timeout/failure/cancellation. Callers catching the builtin exceptions must update to the new types (or their `AgentFieldError` base).
- `ConnectionManager.start`/`get_session`, `AsyncExecutionManager.start`/`submit_execution`, `Agent.pause`/`Agent.discover`, and `AgentRouter` attribute access now raise `AgentFieldClientError` instead of `RuntimeError`.
- `agentfield.utils.get_free_port` now raises `AgentFieldError` instead of `RuntimeError`.

## [0.1.2] - 2025-11-12
### Changed
- Version bump to align with the control-plane Docker fix (no SDK behavior changes).

## [0.1.1] - 2025-11-12
### Added
- Expanded memory/client coverage in tests and documentation examples from latest examples directory.

### Changed
- Release tooling updates so prereleases publish to PyPI and skip unnecessary builds.

## [0.1.0] - 2024-XX-XX
- Initial public release of the AgentField Python SDK.
- Provides agent runtime, workflow helpers, async execution, and credential tooling.
