"""
Mock LLM Server - Simulates an LLM backend (like LiteLLM) that can be
toggled between healthy and unhealthy states.

Used to test:
- LLM health monitoring and circuit breaker behavior
- What happens when the LLM goes down mid-execution
- Recovery when the LLM comes back

Control the server state via:
- GET  /health          → returns health status
- POST /freeze          → makes the server stop responding (hangs)
- POST /unfreeze        → restores normal operation
- POST /error           → makes /health return 500
- POST /recover         → restores /health to 200
- GET  /v1/models       → fake model list (for LiteLLM compat)
- POST /v1/chat/completions → fake completion (or hang if frozen)
"""

import json
import signal
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

STATE = {
    "frozen": False,
    "error": False,
    "request_count": 0,
    "freeze_count": 0,
}
STATE_LOCK = threading.Lock()


class MockLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Prefix logs for clarity
        sys.stderr.write(f"[mock-llm] {format % args}\n")

    def do_GET(self):
        with STATE_LOCK:
            STATE["request_count"] += 1

        if self.path == "/health":
            self._handle_health()
        elif self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/state":
            self._handle_state()
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        with STATE_LOCK:
            STATE["request_count"] += 1

        if self.path == "/freeze":
            self._handle_freeze()
        elif self.path == "/unfreeze":
            self._handle_unfreeze()
        elif self.path == "/error":
            self._handle_set_error()
        elif self.path == "/recover":
            self._handle_recover()
        elif self.path == "/v1/chat/completions":
            self._handle_chat()
        else:
            self._respond(404, {"error": "not found"})

    def _handle_health(self):
        with STATE_LOCK:
            if STATE["frozen"]:
                pass  # Will hang below
            elif STATE["error"]:
                self._respond(500, {"status": "error", "message": "simulated LLM failure"})
                return
            else:
                self._respond(200, {"status": "ok"})
                return

        # Frozen: hang until unfrozen
        self._hang_until_unfrozen()

    def _handle_models(self):
        with STATE_LOCK:
            if STATE["frozen"]:
                pass  # hang
            else:
                self._respond(200, {
                    "data": [
                        {"id": "gpt-4o-mini", "object": "model"},
                        {"id": "gpt-4o", "object": "model"},
                    ]
                })
                return
        self._hang_until_unfrozen()

    def _handle_chat(self):
        with STATE_LOCK:
            if STATE["frozen"]:
                pass  # hang
            elif STATE["error"]:
                self._respond(500, {"error": {"message": "LLM backend error"}})
                return
            else:
                self._respond(200, {
                    "id": "mock-resp-001",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This is a mock LLM response."
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                })
                return
        self._hang_until_unfrozen()

    def _handle_freeze(self):
        with STATE_LOCK:
            STATE["frozen"] = True
            STATE["freeze_count"] += 1
        self._respond(200, {"action": "frozen", "message": "LLM server is now hanging on all requests"})

    def _handle_unfreeze(self):
        with STATE_LOCK:
            STATE["frozen"] = False
        self._respond(200, {"action": "unfrozen", "message": "LLM server resumed normal operation"})

    def _handle_set_error(self):
        with STATE_LOCK:
            STATE["error"] = True
        self._respond(200, {"action": "error_mode", "message": "LLM /health now returns 500"})

    def _handle_recover(self):
        with STATE_LOCK:
            STATE["error"] = False
        self._respond(200, {"action": "recovered", "message": "LLM /health now returns 200"})

    def _handle_state(self):
        with STATE_LOCK:
            state_copy = dict(STATE)
        self._respond(200, state_copy)

    def _hang_until_unfrozen(self):
        """Block the request until the server is unfrozen (check every 0.5s)."""
        while True:
            with STATE_LOCK:
                if not STATE["frozen"]:
                    break
            time.sleep(0.5)
        # After unfreezing, respond normally
        self._respond(200, {"status": "ok", "note": "resumed after freeze"})

    def _respond(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())


class ThreadedHTTPServer(HTTPServer):
    """Handle requests in a separate thread so freeze doesn't block control endpoints."""
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle_request_thread, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    server = ThreadedHTTPServer(("0.0.0.0", port), MockLLMHandler)

    def shutdown(sig, frame):
        print(f"\n[mock-llm] Shutting down...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"[mock-llm] Mock LLM server running on http://0.0.0.0:{port}")
    print(f"[mock-llm] Endpoints:")
    print(f"[mock-llm]   GET  /health              - Health check")
    print(f"[mock-llm]   GET  /state               - Current server state")
    print(f"[mock-llm]   POST /freeze              - Make server hang")
    print(f"[mock-llm]   POST /unfreeze            - Resume normal operation")
    print(f"[mock-llm]   POST /error               - Make /health return 500")
    print(f"[mock-llm]   POST /recover             - Restore /health to 200")
    print(f"[mock-llm]   POST /v1/chat/completions - Fake chat completion")
    server.serve_forever()


if __name__ == "__main__":
    main()
