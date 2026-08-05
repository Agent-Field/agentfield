from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class StubAgentRouter:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def reasoner(self):
        return lambda function: function


agentfield = types.ModuleType("agentfield")
agentfield.AgentRouter = StubAgentRouter
schemas = types.ModuleType("schemas")
for name in ("Citation", "ResearchFindings", "SearchQueries", "TaskResult"):
    setattr(schemas, name, type(name, (), {}))

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "deep_research_search", EXAMPLE_DIR / "routers" / "research.py"
)
assert SPEC and SPEC.loader
research = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"agentfield": agentfield, "schemas": schemas}):
    SPEC.loader.exec_module(research)


def parallel_envelope(results: list[dict]) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": json.dumps({"results": results})}]
        },
    }


class SearchProviderTests(unittest.TestCase):
    def test_search_provider_preserves_tavily_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(research._search_provider(), "tavily")
        with patch.dict(os.environ, {"SEARCH_PROVIDER": "  "}, clear=True):
            self.assertEqual(research._search_provider(), "tavily")

    def test_search_provider_accepts_parallel_and_rejects_unknown(self):
        with patch.dict(os.environ, {"SEARCH_PROVIDER": " Parallel "}, clear=True):
            self.assertEqual(research._search_provider(), "parallel")
        with patch.dict(os.environ, {"SEARCH_PROVIDER": "other"}, clear=True):
            with self.assertRaisesRegex(ValueError, "tavily, parallel"):
                research._search_provider()

    def test_parallel_request_uses_fixed_web_search_contract(self):
        self.assertEqual(
            research._parallel_request("latest MCP news", 7),
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "web_search",
                    "arguments": {
                        "objective": "latest MCP news",
                        "search_queries": ["latest MCP news"],
                    },
                },
            },
        )

    def test_parallel_results_normalize_and_skip_unusable_entries(self):
        envelope = parallel_envelope(
            [
                {
                    "title": "First",
                    "url": "https://example.com/first",
                    "excerpts": [" One ", "Two"],
                },
                {"title": "Missing URL", "excerpts": ["ignored"]},
                {
                    "title": "Second",
                    "url": "https://example.com/second",
                    "excerpts": "Only excerpt",
                },
            ]
        )

        self.assertEqual(
            research._parallel_results(envelope),
            [
                {
                    "title": "First",
                    "url": "https://example.com/first",
                    "content": "One\n\nTwo",
                    "raw_content": "One\n\nTwo",
                },
                {
                    "title": "Second",
                    "url": "https://example.com/second",
                    "content": "Only excerpt",
                    "raw_content": "Only excerpt",
                },
            ],
        )

    def test_parallel_results_reject_error_and_malformed_envelopes(self):
        cases = [
            ({"jsonrpc": "2.0", "error": {"code": -1}}, "JSON-RPC error"),
            ({"result": {"isError": True, "content": []}}, "MCP tool error"),
            ({"result": {"content": []}}, "invalid web_search content"),
            (
                {"result": {"content": [{"type": "text", "text": "not json"}]}},
                "invalid web_search content",
            ),
        ]
        for envelope, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    research._parallel_results(envelope)

    def test_tavily_path_preserves_key_and_request_options(self):
        calls = []

        class FakeTavilyClient:
            def __init__(self, api_key):
                self.assert_key = api_key

            def search(self, **kwargs):
                calls.append((self.assert_key, kwargs))
                return {"results": [{"title": "Tavily result"}]}

        tavily = types.ModuleType("tavily")
        tavily.TavilyClient = FakeTavilyClient
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=True),
            patch.dict(sys.modules, {"tavily": tavily}),
        ):
            results = research._execute_tavily_search(["query"])

        self.assertEqual(results, [{"results": [{"title": "Tavily result"}]}])
        self.assertEqual(
            calls,
            [
                (
                    "test-key",
                    {
                        "query": "query",
                        "search_depth": "advanced",
                        "max_results": 5,
                        "include_answer": False,
                        "include_raw_content": True,
                    },
                )
            ],
        )

    def test_tavily_path_still_requires_its_key(self):
        tavily = types.ModuleType("tavily")
        tavily.TavilyClient = object
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"tavily": tavily}),
        ):
            with self.assertRaisesRegex(ValueError, "TAVILY_API_KEY"):
                research._execute_tavily_search(["query"])


class AsyncSearchProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_search_routes_to_parallel_without_tavily_key(self):
        async def fake_parallel(queries):
            self.assertEqual(queries, ["query"])
            return [
                {
                    "results": [
                        {
                            "title": "Result",
                            "url": "https://example.com",
                            "content": "Excerpt",
                        }
                    ]
                }
            ]

        environment = {"SEARCH_PROVIDER": "parallel"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(research, "_execute_parallel_search", fake_parallel),
        ):
            result = await research.execute_search(["query"])

        self.assertEqual(
            result,
            {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com",
                        "content": "Excerpt",
                    }
                ],
                "queries": ["query"],
            },
        )

    async def test_parallel_search_sends_headers_and_isolates_failures(self):
        requests = []

        class FakeResponse:
            def __init__(self, status, envelope=None):
                self.status = status
                self.envelope = envelope

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def json(self, content_type=None):
                self_test.assertIsNone(content_type)
                return self.envelope

        class FakeSession:
            def __init__(self, **kwargs):
                self_test.assertEqual(kwargs["timeout"].total, 30)
                self.responses = [
                    FakeResponse(
                        200,
                        parallel_envelope(
                            [
                                {
                                    "title": "Good",
                                    "url": "https://example.com/good",
                                    "excerpts": ["text"],
                                }
                            ]
                        ),
                    ),
                    FakeResponse(503),
                ]

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            def post(self, url, *, json, headers):
                requests.append((url, json, headers))
                return self.responses.pop(0)

        class FakeTimeout:
            def __init__(self, total):
                self.total = total

        self_test = self
        aiohttp = types.ModuleType("aiohttp")
        aiohttp.ClientTimeout = FakeTimeout
        aiohttp.ClientSession = FakeSession

        with patch.dict(sys.modules, {"aiohttp": aiohttp}):
            results = await research._execute_parallel_search(["good", "unavailable"])

        self.assertEqual(requests[0][0], research.PARALLEL_MCP_URL)
        self.assertEqual(requests[0][1], research._parallel_request("good", 1))
        self.assertEqual(requests[1][1], research._parallel_request("unavailable", 2))
        self.assertEqual(
            requests[0][2],
            {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(results[0]["results"][0]["title"], "Good")
        self.assertEqual(
            results[1],
            {
                "error": "Parallel returned HTTP status 503",
                "query": "unavailable",
            },
        )


if __name__ == "__main__":
    unittest.main()
