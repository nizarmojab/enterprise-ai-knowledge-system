from app.models.query import QueryRequest, QueryResponse, SourceRef
from app.services.embeddings import OllamaEmbeddingsClient
from app.services.qdrant_store import QdrantStore
from app.services.llm_generate import OllamaLLMClient
from app.services.retrieval import retrieve_candidates


def build_prompt(question: str, contexts: list[dict]) -> str:
    ctx_block = "\n\n".join([f"{c['tag']}\n{c['text']}" for c in contexts])

    return f"""You are an enterprise document analyst.
Use ONLY the provided sources. If the answer is not in the sources, say you don't know.

Question:
{question}

Sources:
{ctx_block}

Output format (French):
1) Résumé (3-5 lignes)
2) Points clés (5 bullets max)
3) Concepts/termes importants (liste)
4) Limites / informations manquantes (1-3 bullets)

Rules:
- Every bullet MUST include citations like [S1], [S2].
- No external knowledge, no web, no Wikipedia.
- Be concise and faithful to sources.
"""


def query_rag(req: QueryRequest) -> QueryResponse:
    embedder = OllamaEmbeddingsClient()
    store = QdrantStore()

    # 🔥 Retrieval avancé
    selected = retrieve_candidates(
        store=store,
        embedder=embedder,
        question=req.question,
        doc_id=req.doc_id,
        tags=req.tags,
        top_k=req.top_k,
        candidates_k=max(30, req.top_k * 6),  # retrieve wide
        min_score=req.min_score,
        multiquery_n=3,
    )

    # Construire contexts + sources
    contexts: list[dict] = []
    sources: list[SourceRef] = []

    MAX_CONTEXT_CHARS = 6000
    used_chars = 0

    for i, c in enumerate(selected, start=1):
        payload = c["payload"]
        text = c["text"]

        doc_id = payload.get("doc_id", "")
        page = int(payload.get("page", -1))
        chunk_id = payload.get("chunk_id", str(getattr(c["hit"], "id", "")))
        extraction_type = payload.get("extraction_type")
        source_name = payload.get("source")
        score = float(getattr(c["hit"], "score", 0.0) or 0.0)

        if used_chars + len(text) > MAX_CONTEXT_CHARS:
            break

        contexts.append({"tag": f"[S{i}] (doc={doc_id}, page={page})", "text": text})
        used_chars += len(text)

        sources.append(
            SourceRef(
                doc_id=doc_id,
                page=page,
                chunk_id=chunk_id,
                extraction_type=extraction_type,
                score=score,
                source=source_name,
            )
        )

    # Guardrail anti-hallucination
    if not contexts:
        return QueryResponse(
            answer="Aucune information pertinente trouvée dans les documents indexés pour répondre à cette question.",
            sources=[],
            debug={
                "top_k": req.top_k,
                "doc_filter": req.doc_id,
                "hits_returned": 0,
            },
        )

    # LLM
    llm = OllamaLLMClient(model="llama3.1:8b")
    prompt = build_prompt(req.question, contexts)
    answer = llm.generate(prompt).strip()

    return QueryResponse(
        answer=answer,
        sources=sources,
        debug={
            "top_k": req.top_k,
            "doc_filter": req.doc_id,
            "hits_returned": len(sources),
            "context_chars": used_chars,
            "retrieval_mode": "multiquery+mmr",
        },
    )