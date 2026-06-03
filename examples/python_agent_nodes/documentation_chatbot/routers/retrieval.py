"""Vector retrieval reasoners."""

from __future__ import annotations

import asyncio
from typing import List

from agentfield import AgentRouter
from agentfield.logger import log_info

from embedding import embed_query
from pipeline_utils import deduplicate_results, filter_hits
from schemas import RetrievalResult

retrieval_router = AgentRouter(tags=["retrieval"])


async def _retrieve_for_query(
    global_memory,
    query: str,
    namespace: str,
    top_k: int,
    min_score: float,
) -> List[RetrievalResult]:
    """Single retrieval operation for one query."""

    embedding = embed_query(query)

    raw_hits = await global_memory.similarity_search(
        query_embedding=embedding, top_k=top_k * 2
    )

    filtered_hits = filter_hits(raw_hits, namespace=namespace, min_score=min_score)

    results: List[RetrievalResult] = []
    for hit in filtered_hits[:top_k]:
        metadata = hit.get("metadata", {})
        text = metadata.get("text", "").strip()
        if not text:
            continue

        relative_path = metadata.get("relative_path", "unknown")
        start_line = int(metadata.get("start_line", 0))
        end_line = int(metadata.get("end_line", 0))
        source = f"{relative_path}:{start_line}-{end_line}"

        results.append(
            RetrievalResult(
                text=text,
                source=source,
                score=float(hit.get("score", 0.0)),
                metadata=metadata,
            )
        )

    return results


@retrieval_router.reasoner()
async def parallel_retrieve(
    queries: List[str],
    namespace: str = "website-docs",
    top_k: int = 6,
    min_score: float = 0.35,
) -> List[RetrievalResult]:
    """Execute parallel retrieval for all queries and deduplicate results."""

    log_info(f"[parallel_retrieve] Running {len(queries)} queries in parallel")
    global_memory = retrieval_router.memory.global_scope

    tasks = [
        _retrieve_for_query(global_memory, query, namespace, top_k, min_score)
        for query in queries
    ]
    all_results_lists = await asyncio.gather(*tasks)

    all_results: List[RetrievalResult] = []
    for results in all_results_lists:
        all_results.extend(results)

    log_info(
        f"[parallel_retrieve] Retrieved {len(all_results)} total chunks before deduplication"
    )

    deduplicated = deduplicate_results(all_results)

    log_info(f"[parallel_retrieve] Returning {len(deduplicated)} unique chunks")

    return deduplicated
