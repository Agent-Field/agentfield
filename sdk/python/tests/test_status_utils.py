from haxen_sdk.status import normalize_status, is_terminal, TERMINAL_STATUSES


def test_status_normalization_all_values():
    cases = {
        "  SUCCEEDED  ": "succeeded",
        "Success": "succeeded",
        "COMPLETED": "succeeded",
        "ok": "succeeded",
        "FAILED": "failed",
        "error": "failed",
        "canceled": "cancelled",
        "cancel": "cancelled",
        "timed_out": "timeout",
        "WAITING": "queued",
        "in_progress": "running",
        "": "unknown",
        None: "unknown",
        "mystery": "unknown",
    }

    for raw, expected in cases.items():
        assert normalize_status(raw) == expected


def test_is_terminal_aligns_with_terminal_set():
    for status in TERMINAL_STATUSES:
        assert is_terminal(status)

    non_terminals = ["pending", "queued", "running", "unknown", "mystery"]
    for status in non_terminals:
        assert not is_terminal(status)
