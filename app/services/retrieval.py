from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Iterable
import math

from app.services.embeddings import OllamaEmbeddingsClient
from app.services.llm_generate import OllamaLLMClient
from app.services.qdrant_store import QdrantStore, build_filter


def _cosine(a: List[float], b: List[float]) -> float:
    # cosine similarity
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(min(len(a), len(b))):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def mmr_select(
    query_vec: List[float],
    candidates: List[Dict[str, Any]],
    k: int = 5,
    lambda_mult: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    candidates: list of {"hit": <qdrant hit>, "text": str, "vec": [..], ...}
    Returns k selected candidates using MMR.
    """
    if not candidates:
        return []
    selected: List[Dict[str, Any]] = []
    remaining = candidates[:]

    # precompute sim(query, doc)
    for c in remaining:
        c["sim_q"] = _cosine(query_vec, c["vec"])

    while remaining and len(selected) < k:
        best = None
        best_score = -1e9
        for c in remaining:
            if not selected:
                score = c["sim_q"]
            else:
                sim_to_selected = max(_cosine(c["vec"], s["vec"]) for s in selected)
                score = lambda_mult * c["sim_q"] - (1.0 - lambda_mult) * sim_to_selected

            if score > best_score:
                best_score = score
                best = c

        selected.append(best)
        remaining.remove(best)

    return selected


def generate_multi_queries(question: str, n: int = 3) -> List[str]:
    """
    Use LLM to create diverse reformulations for better recall.
    Very short, safe prompts.
    """
    llm = OllamaLLMClient(model="llama3.1:8b")
    prompt = f"""Generate {n} short alternative search queries in French for the following user question.
Return ONLY the queries, one per line, no numbering, no extra text.

Question: {question}
"""
    raw = llm.generate(prompt).strip()
    lines = [l.strip(" -\t") for l in raw.splitlines() if l.strip()]
    # fallback
    if not lines:
        return [question]
    # keep unique, keep original first
    uniq = []
    for q in [question] + lines:
        if q and q not in uniq:
            uniq.append(q)
    return uniq[: max(1, n)]


def retrieve_candidates(
    store: QdrantStore,
    embedder: OllamaEmbeddingsClient,
    question: str,
    doc_id: str | None,
    tags: List[str] | None,
    top_k: int,
    candidates_k: int = 30,
    min_score: float = 0.0,
    multiquery_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Returns list of candidate dicts (deduped) with payload text + vectors.
    We embed and fetch vectors for MMR using the same embedder.
    """
    qfilter = build_filter(doc_id=doc_id, tags=tags)

    # 1) multi-query for better recall
    queries = generate_multi_queries(question, n=multiquery_n)

    all_hits = []
    qvecs = []
    for q in queries:
        qvec = embedder.embed_one(q)
        qvecs.append(qvec)
        hits = store.search(query_vector=qvec, top_k=candidates_k, qfilter=qfilter)
        all_hits.extend(hits)

    # 2) dedupe by (doc_id, page, chunk_id) OR hash(text)
    seen = set()
    deduped = []
    for hit in all_hits:
        payload = getattr(hit, "payload", None) or {}
        text = (payload.get("text", "") or "").strip()
        if not text:
            continue
        score = float(getattr(hit, "score", 0.0) or 0.0)
        if score < min_score:
            continue

        key = (
            payload.get("doc_id", ""),
            int(payload.get("page", -1)),
            payload.get("chunk_id", str(getattr(hit, "id", ""))),
            hash(text),
        )
        if key in seen:
            continue
        seen.add(key)

        deduped.append({"hit": hit, "text": text, "payload": payload})

    # 3) embed candidate texts for MMR (lightweight; OK for 30-90 items)
    texts = [c["text"] for c in deduped]
    vecs = embedder.embed_batch(texts) if texts else []
    for i, v in enumerate(vecs):
        deduped[i]["vec"] = v

    # 4) MMR on the ORIGINAL question vector (first one)
    base_qvec = qvecs[0] if qvecs else embedder.embed_one(question)
    selected = mmr_select(base_qvec, deduped, k=top_k, lambda_mult=0.7)

    return selected