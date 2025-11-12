"""Documentation chatbot agent with inline citations and no router prefixes."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable, Dict, List, Sequence, Tuple

from agentfield import AIConfig, Agent
from agentfield.logger import log_info

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

from chunking import chunk_markdown_text, is_supported_file, read_text
from embedding import embed_query, embed_texts
from schemas import (
    AnswerCheck,
    Citation,
    ContextChunk,
    ContextWindow,
    DocAnswer,
    IngestReport,
    InlineAnswer,
    QueryPlan,
    QuestionFocus,
    SearchAngles,
)

app = Agent(
    node_id="documentation-chatbot",
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("AI_MODEL", "openrouter/openai/gpt-oss-120b"),
        temperature=0.2,
    ),
)


# ========================= Ingestion Skill =========================


@app.skill()
async def ingest_folder(
    folder_path: str,
    namespace: str = "documentation",
    glob_pattern: str = "**/*",
    chunk_size: int = 1200,
    chunk_overlap: int = 250,
) -> IngestReport:
    """Chunk + embed every supported file inside ``folder_path``."""

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

    global_memory = app.memory.global_scope

    total_chunks = 0
    for file_path in supported_files:
        relative_path = file_path.relative_to(root).as_posix()
        try:
            text = read_text(file_path)
        except Exception as exc:  # pragma: no cover - defensive
            skipped.append(f"{relative_path} (error: {exc})")
            continue

        doc_chunks = chunk_markdown_text(
            text,
            relative_path=relative_path,
            namespace=namespace,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        if not doc_chunks:
            continue

        embeddings = embed_texts([chunk.text for chunk in doc_chunks])
        for chunk, embedding in zip(doc_chunks, embeddings):
            vector_key = f"{namespace}|{chunk.chunk_id}"
            metadata = {
                "text": chunk.text,
                "namespace": namespace,
                "relative_path": chunk.relative_path,
                "section": chunk.section,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
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


# ========================= QA Reasoners =========================


def _filter_hits(
    hits: Sequence[Dict],
    *,
    namespace: str,
    min_score: float,
) -> List[Dict]:
    filtered: List[Dict] = []
    for hit in hits:
        metadata = hit.get("metadata", {})
        if metadata.get("namespace") != namespace:
            continue
        if hit.get("score", 0.0) < min_score:
            continue
        filtered.append(hit)
    return filtered


def _alpha_key(index: int) -> str:
    if index < 0:
        raise ValueError("Index must be non-negative")

    letters: List[str] = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        letters.append(chr(ord("A") + remainder))
        if current == 0:
            break
        current -= 1
    return "".join(reversed(letters))


def _build_context_entries(hits: Sequence[Dict]) -> List[ContextChunk]:
    entries: List[ContextChunk] = []
    for hit in hits:
        metadata = hit.get("metadata", {})
        text = metadata.get("text", "").strip()
        if not text:
            continue
        key = _alpha_key(len(entries))
        citation = Citation(
            key=key,
            relative_path=metadata.get("relative_path", "unknown"),
            start_line=int(metadata.get("start_line", 0)),
            end_line=int(metadata.get("end_line", 0)),
            section=metadata.get("section"),
            preview=text[:200],
            score=float(hit.get("score", 0.0)),
        )
        entries.append(ContextChunk(key=key, text=text, citation=citation))
    return entries


def _context_prompt(entries: Sequence[ContextChunk]) -> str:
    if not entries:
        return "(no context available)"
    blocks: List[str] = []
    for entry in entries:
        citation = entry.citation
        section = f" · {citation.section}" if citation.section else ""
        location = f"{citation.relative_path}:{citation.start_line}-{citation.end_line}{section}"
        blocks.append(f"[{entry.key}] {location}\n{entry.text}")
    return "\n\n".join(blocks)


def _filter_citations_by_keys(
    entries: Sequence[ContextChunk], keys: Sequence[str]
) -> List[Citation]:
    lookup = {entry.key: entry.citation for entry in entries}
    unique_keys: List[str] = []
    for key in keys:
        if key not in lookup:
            continue
        if key in unique_keys:
            continue
        unique_keys.append(key)
    return [lookup[key] for key in unique_keys]


def _ensure_plan(data: Any) -> QueryPlan:
    if isinstance(data, QueryPlan):
        return data
    return QueryPlan.model_validate(data)


def _ensure_window(data: Any) -> ContextWindow:
    if isinstance(data, ContextWindow):
        return data
    return ContextWindow.model_validate(data)


def _ensure_angles(data: Any) -> SearchAngles:
    if isinstance(data, SearchAngles):
        return data
    return SearchAngles.model_validate(data)


def _merge_lists(base: List[str], additions: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for value in base + additions:
        value_clean = value.strip()
        if not value_clean:
            continue
        if value_clean.lower() in seen:
            continue
        seen.add(value_clean.lower())
        merged.append(value_clean)
    return merged


def _literal_mismatches(answer: str, contexts: Sequence[ContextChunk]) -> List[str]:
    literals = re.findall(r"`([^`]+)`", answer)
    if not literals:
        return []

    corpus = "\n".join(entry.text for entry in contexts).lower()
    gaps: List[str] = []
    for literal in literals:
        cleaned = literal.strip()
        if not cleaned:
            continue
        if cleaned.lower() not in corpus:
            gaps.append(f"`{cleaned}` not found in context")
    return gaps


def _context_inventory(contexts: Sequence[ContextChunk]) -> List[str]:
    inventory: List[str] = []
    for entry in contexts:
        citation = entry.citation
        descriptor = f"{citation.relative_path}:{citation.section or 'section?'}"
        inventory.append(descriptor)
    return inventory


def _extract_terms(items: Sequence[str]) -> List[str]:
    """Reduce critique strings into concise search tokens."""

    extracted: List[str] = []
    for item in items:
        literals = re.findall(r"`([^`]+)`", item)
        if literals:
            extracted.extend(literals)
        else:
            extracted.append(item)
    cleaned = [value.strip() for value in extracted if value.strip()]
    return cleaned


ReasonerFunc = Callable[..., Awaitable[Any]]


async def _call_reasoner(
    reasoner_name: str,
    func: ReasonerFunc,
    **kwargs: Any,
) -> Any:
    """
    Try cross-agent call first for observability; fall back to local execution if control plane is down.
    """

    full_id = f"{app.node_id}.{reasoner_name}"
    try:
        return await app.call(full_id, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on network state
        if "AgentField server unavailable" not in str(exc):
            raise
        log_info(
            f"[fallback] Control plane unavailable for {reasoner_name}; running locally."
        )
        return await func(**kwargs)


@app.reasoner()
async def qa_focus_question(user_input: str) -> QuestionFocus:
    """Reduce conversation blobs into a crisp current question + key terms."""

    return await app.ai(
        system=(
            "You extract the actionable question from chat transcripts. "
            "Ignore refusal text. Return a short clean question, 2-4 key terms, and a note if additional context matters."
        ),
        user=(
            "Conversation fragment:\n"
            f"{user_input}\n\n"
            "Identify the user's latest question in plain language. "
            "List essential nouns/phrases that MUST appear in the answer. "
            "Mention any context that should influence retrieval (e.g., product area, feature set)."
        ),
        schema=QuestionFocus,
    )


@app.reasoner()
async def qa_plan(question: str) -> QueryPlan:
    """Analyze the question and return retrieval instructions."""

    return await app.ai(
        system="You design retrieval plans for documentation search.",
        user=f"""Question: {question}

Return focused search terms (2-4), critical words that must appear,
an answer style (direct, step_by_step, or comparison),
and a refusal condition describing when to say you lack information.""",
        schema=QueryPlan,
    )


@app.reasoner()
async def qa_generate_queries(question: str, plan: Dict[str, Any]) -> SearchAngles:
    """Produce complementary search angles to run in parallel."""

    plan_obj = _ensure_plan(plan)
    return await app.ai(
        system=(
            "You propose extra focused search queries for documentation retrieval. "
            "Keep them short (<=6 words) and targeted."
        ),
        user=(
            f"Question: {question}\n"
            f"Existing search terms: {plan_obj.search_terms}\n"
            "List 2-3 complementary search queries that cover different facets or synonyms.\n"
            "Also explain in one short phrase what they focus on."
        ),
        schema=SearchAngles,
    )


@app.reasoner()
async def qa_refine_queries(
    question: str,
    plan: Dict[str, Any],
    critique: Dict[str, Any],
    known_queries: List[str],
    context_summary: List[str],
) -> SearchAngles:
    """Suggest additional focused queries based on critique feedback."""

    plan_obj = _ensure_plan(plan)
    check = AnswerCheck.model_validate(critique)

    return await app.ai(
        system=(
            "You propose new search phrases to fill evidence gaps. "
            "Avoid repeating known queries."
        ),
        user=(
            f"Question: {question}\n"
            f"Plan search terms: {plan_obj.search_terms}\n"
            f"Known queries: {known_queries}\n"
            f"Context inventory: {context_summary}\n"
            f"Critique verdict: {check.verdict}\n"
            f"Missing terms: {check.missing_terms}\n"
            f"Unsupported claims: {check.unsupported_claims}\n"
            "Return 2-3 short queries plus a one-line focus on what they target."
        ),
        schema=SearchAngles,
    )


@app.reasoner()
async def qa_retrieve(
    question: str,
    namespace: str,
    plan: Dict[str, Any],
    queries: Dict[str, Any],
    top_k: int = 6,
    min_score: float = 0.35,
) -> ContextWindow:
    """Retrieve the highest-signal snippets for the current plan."""

    plan_obj = _ensure_plan(plan)
    angles = _ensure_angles(queries)

    search_basket = _merge_lists(
        [question], plan_obj.search_terms + angles.queries
    )
    global_memory = app.memory.global_scope

    best_by_key: Dict[str, Dict] = {}
    for text in search_basket:
        embedding = embed_query(text)
        raw_hits = await global_memory.similarity_search(
            query_embedding=embedding, top_k=top_k * 2
        )
        filtered_hits = _filter_hits(raw_hits, namespace=namespace, min_score=min_score)
        for hit in filtered_hits:
            key = hit.get("key")
            if not key:
                continue
            existing = best_by_key.get(key)
            if not existing or hit.get("score", 0) > existing.get("score", 0):
                best_by_key[key] = hit

    sorted_hits = sorted(
        best_by_key.values(), key=lambda h: h.get("score", 0), reverse=True
    )
    context_entries = _build_context_entries(sorted_hits[:top_k])
    return ContextWindow(contexts=context_entries)


@app.reasoner()
async def qa_synthesize(
    question: str,
    plan: Dict[str, Any],
    contexts: Dict[str, Any],
) -> InlineAnswer:
    """Generate a markdown answer using the supplied snippets."""

    plan_obj = _ensure_plan(plan)
    context_window = _ensure_window(contexts)

    if not context_window.contexts:
        return InlineAnswer(
            answer=(
                "I could not find any matching documentation yet. "
                f"({plan_obj.refusal_condition})"
            ),
            cited_keys=[],
        )

    context_prompt = _context_prompt(context_window.contexts)
    snippets_json = json.dumps(
        {entry.key: entry.text for entry in context_window.contexts}, indent=2
    )

    return await app.ai(
        system=(
            "You are a precise documentation assistant. Answer ONLY when the info is in the context map. "
            "Always respond using GitHub-flavored Markdown (2-4 concise sentences or bullets) and keep citation keys inline like [A] or [B][D]. "
            "Only mention API names, CLI commands, or config values if the exact literal string appears in the snippets. "
            "If the context is insufficient, respond with a short markdown note explaining that."
        ),
        user=(
            f"Question: {question}\n"
            f"Answer style: {plan_obj.answer_style}\n"
            f"Critical terms that must appear: {', '.join(plan_obj.must_include) or 'none'}\n"
            "Context map (JSON where each key maps to a snippet):\n"
            f"{snippets_json}\n\n"
            "Readable context with locations:\n"
            f"{context_prompt}\n\n"
            "Respond with a concise markdown answer (<= 6 sentences) keeping the citation keys inline."
        ),
        schema=InlineAnswer,
    )


@app.reasoner()
async def qa_review(
    question: str,
    plan: Dict[str, Any],
    contexts: Dict[str, Any],
    answer: str,
) -> AnswerCheck:
    """Meta-review the draft answer for completeness and grounding."""

    plan_obj = _ensure_plan(plan)
    context_window = _ensure_window(contexts)
    context_prompt = _context_prompt(context_window.contexts)

    return await app.ai(
        system=(
            "You audit documentation answers for completeness and hallucinations. "
            "Be strict: mark needs_more_context whenever key terms are missing OR the context lacks the cited facts. "
            "List every unsupported_claim (claims in the draft that you cannot locate verbatim or in paraphrased form inside the context). "
            "Do not invent facts; if the answer overreaches, flag it."
        ),
        user=(
            f"Question: {question}\n"
            f"Plan search terms: {plan_obj.search_terms}\n"
            f"Plan must include: {plan_obj.must_include}\n\n"
            f"Draft answer:\n{answer}\n\n"
            "Context provided:\n"
            f"{context_prompt}\n\n"
            "Decide if the answer is well-supported. "
            "If missing details, list the concrete topics or entities that need more retrieval. "
            "For unsupported_claims, quote short snippets from the answer that are NOT present anywhere in the context."
        ),
        schema=AnswerCheck,
    )


async def _run_iteration(
    *,
    question: str,
    namespace: str,
    plan: QueryPlan,
    angles: SearchAngles,
    top_k: int,
    min_score: float,
) -> tuple[ContextWindow, InlineAnswer, AnswerCheck]:
    plan_payload = plan.model_dump()
    angle_payload = angles.model_dump()

    context_data = await _call_reasoner(
        "qa_retrieve",
        qa_retrieve,
        question=question,
        namespace=namespace,
        plan=plan_payload,
        queries=angle_payload,
        top_k=top_k,
        min_score=min_score,
    )
    context_window = _ensure_window(context_data)

    inline_data = await _call_reasoner(
        "qa_synthesize",
        qa_synthesize,
        question=question,
        plan=plan_payload,
        contexts=context_window.model_dump(),
    )
    inline_answer = InlineAnswer.model_validate(inline_data)

    critique_data = await _call_reasoner(
        "qa_review",
        qa_review,
        question=question,
        plan=plan_payload,
        contexts=context_window.model_dump(),
        answer=inline_answer.answer,
    )
    critique = AnswerCheck.model_validate(critique_data)

    return context_window, inline_answer, critique


@app.reasoner()
async def qa_answer(
    question: str,
    namespace: str = "documentation",
    top_k: int = 6,
    min_score: float = 0.35,
) -> DocAnswer:
    """Orchestrate planning → retrieval → synthesis → self-review."""

    focus_data = await _call_reasoner(
        "qa_focus_question", qa_focus_question, user_input=question
    )
    focus = QuestionFocus.model_validate(focus_data)
    core_question = focus.question.strip() or question

    plan_data = await _call_reasoner("qa_plan", qa_plan, question=core_question)
    plan = _ensure_plan(plan_data)
    if focus.key_terms:
        plan = plan.model_copy(
            update={
                "must_include": _merge_lists(plan.must_include, focus.key_terms),
                "search_terms": _merge_lists(plan.search_terms, focus.key_terms),
            }
        )
    angles_data = await _call_reasoner(
        "qa_generate_queries",
        qa_generate_queries,
        question=core_question,
        plan=plan.model_dump(),
    )
    angles = _ensure_angles(angles_data)
    visited_queries = list(angles.queries)

    max_attempts = 3
    attempt = 0
    latest_contexts = ContextWindow(contexts=[])
    latest_answer = InlineAnswer(answer="I do not know yet.", cited_keys=[])
    latest_critique = AnswerCheck(
        verdict="insufficient",
        needs_more_context=True,
        missing_terms=[],
        unsupported_claims=[],
    )

    while attempt < max_attempts:
        attempt += 1
        latest_contexts, latest_answer, latest_critique = await _run_iteration(
            question=core_question,
            namespace=namespace,
            plan=plan,
            angles=angles,
            top_k=top_k,
            min_score=min_score,
        )

        literal_gaps = _literal_mismatches(
            latest_answer.answer, latest_contexts.contexts
        )
        if literal_gaps:
            latest_critique.unsupported_claims = _merge_lists(
                latest_critique.unsupported_claims, literal_gaps
            )
            latest_critique.needs_more_context = True
            log_info(
                f"[qa_answer] Inline literal gaps detected: {literal_gaps}. "
                "Marking review as needing more context."
            )

        if not latest_critique.needs_more_context:
            break

        if not (latest_critique.missing_terms or latest_critique.unsupported_claims):
            # No guidance on what to fetch—stop to avoid loops.
            break

        raw_expansions = latest_critique.missing_terms + latest_critique.unsupported_claims
        expansions = _extract_terms(raw_expansions)
        plan = plan.model_copy(
            update={
                "search_terms": _merge_lists(plan.search_terms, expansions),
                "must_include": _merge_lists(
                    plan.must_include, latest_critique.missing_terms
                ),
            }
        )
        log_info(
            f"[qa_answer] Critique requested more context ({latest_critique.missing_terms}); "
            "expanding search terms and retrying."
        )
        refinement = await _call_reasoner(
            "qa_refine_queries",
            qa_refine_queries,
            question=core_question,
            plan=plan.model_dump(),
            critique=latest_critique.model_dump(),
            known_queries=visited_queries,
            context_summary=_context_inventory(latest_contexts.contexts),
        )
        new_angles = _ensure_angles(refinement)
        merged_queries = _merge_lists(visited_queries, new_angles.queries)
        visited_queries = merged_queries
        angles = SearchAngles(
            queries=merged_queries,
            focus=new_angles.focus or angles.focus,
        )

    if not latest_contexts.contexts:
        refusal = (
            "I did not find that in the documentation yet. "
            f"(Plan refusal condition: {plan.refusal_condition})"
        )
        return DocAnswer(answer=refusal, citations=[], plan=plan)

    final_literal_gaps = _literal_mismatches(
        latest_answer.answer, latest_contexts.contexts
    )
    if final_literal_gaps:
        latest_critique.unsupported_claims = _merge_lists(
            latest_critique.unsupported_claims, final_literal_gaps
        )
        log_info(
            f"[qa_answer] Final literal mismatch check failed: {final_literal_gaps}"
        )

    if latest_critique.unsupported_claims:
        refusal = (
            "I cannot answer from the documentation because these statements lack evidence: "
            + "; ".join(latest_critique.unsupported_claims)
        )
        return DocAnswer(answer=refusal, citations=[], plan=plan)

    if latest_critique.needs_more_context:
        refusal = (
            "I could not gather enough grounded context to answer. "
            f"(Reason: {latest_critique.verdict})"
        )
        return DocAnswer(answer=refusal, citations=[], plan=plan)

    citations = _filter_citations_by_keys(
        latest_contexts.contexts, latest_answer.cited_keys
    )
    if not citations:
        citations = [entry.citation for entry in latest_contexts.contexts]

    return DocAnswer(
        answer=latest_answer.answer.strip(),
        citations=citations,
        plan=plan,
    )


# ========================= Bootstrapping =========================


def _warmup_embeddings() -> None:
    try:
        embed_texts(["doc-chatbot warmup"])
        log_info("FastEmbed model warmed up for documentation chatbot")
    except Exception as exc:  # pragma: no cover - best-effort
        log_info(f"FastEmbed warmup failed: {exc}")


if __name__ == "__main__":
    _warmup_embeddings()

    print("📚 Documentation Chatbot Agent")
    print("🧠 Node ID: documentation-chatbot")
    print(f"🌐 Control Plane: {app.agentfield_server}")
    print("Endpoints:")
    print("  • /skills/ingest_folder → documentation-chatbot.ingest_folder")
    print("  • /reasoners/qa_focus_question → documentation-chatbot.qa_focus_question")
    print("  • /reasoners/qa_plan → documentation-chatbot.qa_plan")
    print("  • /reasoners/qa_generate_queries → documentation-chatbot.qa_generate_queries")
    print("  • /reasoners/qa_refine_queries → documentation-chatbot.qa_refine_queries")
    print("  • /reasoners/qa_retrieve → documentation-chatbot.qa_retrieve")
    print("  • /reasoners/qa_synthesize → documentation-chatbot.qa_synthesize")
    print("  • /reasoners/qa_review → documentation-chatbot.qa_review")
    print("  • /reasoners/qa_answer → documentation-chatbot.qa_answer")
    app.run(auto_port=True)
