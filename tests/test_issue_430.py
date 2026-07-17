"""
Regression test for issue #430.

The control-plane batch status handler should fetch all requested executions in
one storage call instead of issuing one GetExecutionRecord call per execution ID.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_handle_batch_status_source() -> str:
    execute_go = _repo_root() / "control-plane" / "internal" / "handlers" / "execute.go"
    source = execute_go.read_text(encoding="utf-8")

    signature = (
        "func (c *executionController) handleBatchStatus(ctx *gin.Context) {"
    )
    start = source.find(signature)
    assert start != -1, "handleBatchStatus was not found in execute.go"

    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]

    raise AssertionError("handleBatchStatus body was not closed in execute.go")


def test_issue_430(monkeypatch):
    """
    Batch status polling for many executions must stay O(1) at the storage layer.

    The old implementation loops over request.ExecutionIDs and calls
    GetExecutionRecord once per ID, which creates the N+1 behavior described in
    issue #430. The fixed handler should call GetExecutionsByIDs once and render
    the returned records.
    """
    sdk_path = _repo_root() / "sdk" / "python"
    monkeypatch.syspath_prepend(str(sdk_path))
    module_path = sdk_path / "agentfield" / "openrouter_attribution.py"
    spec = importlib.util.spec_from_file_location(
        "agentfield.openrouter_attribution", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_OPENROUTER_APP_NAME == "AgentField AI"

    handler_source = _load_handle_batch_status_source()

    assert "GetExecutionsByIDs" in handler_source
    assert "for _, id := range request.ExecutionIDs" not in handler_source
    assert "GetExecutionRecord(reqCtx, id)" not in handler_source
