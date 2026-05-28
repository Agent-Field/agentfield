"""Ingestion reasoners for the documentation chatbot."""

from __future__ import annotations

import os
from pathlib import Path

from agentfield import AgentRouter
from agentfield.logger import log_info

from chunking import chunk_markdown_text, is_supported_file, read_text
from embedding import embed_texts
from schemas import IngestReport

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is installed in runtime environments
    httpx = None

ingestion_router = AgentRouter(tags=["ingestion"])


async def _clear_namespace_via_api(namespace: str) -> dict:
    """Call control plane delete-namespace endpoint."""
    if httpx is None:
        raise RuntimeError("httpx is required for clear_namespace")

    base_url = os.getenv("CONTROL_PLANE_URL") or os.getenv("AGENTFIELD_SERVER") or ""
    base_url = base_url.rstrip("/")
    if not base_url:
        raise ValueError("CONTROL_PLANE_URL (or AGENTFIELD_SERVER) is required to clear namespace")

    api_key = os.getenv("CONTROL_PLANE_API_KEY") or os.getenv("AGENTFIELD_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url}/api/v1/memory/vector/namespace"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.delete(url, json={"namespace": namespace, "scope": "global"}, headers=headers)
        resp.raise_for_status()
        return resp.json()


@ingestion_router.skill()
async def clear_namespace(namespace: str = "website-docs") -> dict:
    """Wipe all vectors for a namespace before re-indexing."""
    result = await _clear_namespace_via_api(namespace)
    deleted = result.get("deleted", 0)
    log_info(f"Cleared namespace '{namespace}' (deleted {deleted} vectors)")
    return {"namespace": namespace, "deleted": deleted}


@ingestion_router.reasoner()
async def ingest_folder(
    folder_path: str,
    namespace: str = "website-docs",
    glob_pattern: str = "**/*",
    chunk_size: int = 1200,
    chunk_overlap: int = 250,
) -> IngestReport:
    """
    Chunk + embed every supported file inside ``folder_path``.

    Uses two-tier storage:
    1. Store full document text ONCE in regular memory
    2. Store chunk vectors with reference to document
    """

    root = Path(folder_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted(p for p in root.glob(glob_pattern) if p.is_file())
    supported_files = [p for p in files if is_supported_file(p)]
    skipped = [p.as_posix() for p in files if not is_supported_file(p)]

    if not supported_files:
        return IngestReport(
            namespace=namespace, file_count=0, chunk_count=0, skipped_files=skipped
        )

    global_memory = ingestion_router.memory.global_scope

    total_chunks = 0
    for file_path in supported_files:
        relative_path = file_path.relative_to(root).as_posix()
        try:
            full_text = read_text(file_path)
        except Exception as exc:  # pragma: no cover - defensive
            skipped.append(f"{relative_path} (error: {exc})")
            continue

        # TIER 1: Store full document ONCE
        document_key = f"{namespace}:doc:{relative_path}"
        await global_memory.set(
            key=document_key,
            data={
                "full_text": full_text,
                "relative_path": relative_path,
                "namespace": namespace,
                "file_size": len(full_text),
            },
        )

        # Create chunks
        doc_chunks = chunk_markdown_text(
            full_text,
            relative_path=relative_path,
            namespace=namespace,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        if not doc_chunks:
            continue

        # TIER 2: Store chunk vectors with document reference
        embeddings = embed_texts([chunk.text for chunk in doc_chunks])
        for idx, (chunk, embedding) in enumerate(zip(doc_chunks, embeddings)):
            vector_key = f"{namespace}|{chunk.chunk_id}"
            metadata = {
                "text": chunk.text,
                "namespace": namespace,
                "relative_path": chunk.relative_path,
                "section": chunk.section,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "document_key": document_key,
                "chunk_index": idx,
                "total_chunks": len(doc_chunks),
            }
            await global_memory.set_vector(
                key=vector_key, embedding=embedding, metadata=metadata
            )
            total_chunks += 1

    log_info(
        f"Ingested {total_chunks} chunks from {len(supported_files)} files into namespace '{namespace}'"
    )

    return IngestReport(
        namespace=namespace,
        file_count=len(supported_files),
        chunk_count=total_chunks,
        skipped_files=skipped,
    )
