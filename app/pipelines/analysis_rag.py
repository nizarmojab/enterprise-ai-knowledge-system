import json
import re
from fastapi import HTTPException

from app.models.query import QueryRequest
from app.models.analysis import AnalysisResponse, Citation, KeyPoint
from app.services.embeddings import OllamaEmbeddingsClient
from app.services.qdrant_store import QdrantStore
from app.services.llm_generate import OllamaLLMClient
from app.services.retrieval import retrieve_candidates


def _confidence_from_scores(scores: list[float]) -> float:
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    conf = (avg - 0.45) / (0.75 - 0.45)
    return max(0.0, min(1.0, conf))


def _extract_json_block(text: str) -> str:
    """
    Extract the first JSON object from model output.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


def _safe_json_loads(s: str) -> dict:
    """
    Robust JSON parse:
    - fixes invalid backslash escapes
    - removes trailing commas (common LLM mistake)
    """
    # 1) fix invalid \ escapes by doubling backslashes that are not valid escapes
    # valid escapes in JSON: \" \\ \/ \b \f \n \r \t \uXXXX
    s = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)

    # 2) remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    return json.loads(s)


def _repair_json_with_llm(llm: OllamaLLMClient, raw: str) -> dict:
    """
    Second-pass repair: ask model to output strictly valid JSON.
    """
    prompt = f"""You will be given a JSON-like text that may be invalid.
Fix it and output ONLY valid JSON (no markdown, no explanations).

Text:
{raw}
"""
    fixed = llm.generate(prompt, max_tokens=450).strip()
    block = _extract_json_block(fixed)
    return _safe_json_loads(block)


def build_prompt(question: str, contexts: list[dict]) -> str:
    ctx_block = "\n\n".join([f"{c['tag']}\n{c['text']}" for c in contexts])

    return f"""You are an enterprise document analyst.
Use ONLY the provided sources. If information is missing, say so.

Question:
{question}

Sources:
{ctx_block}

Return ONLY valid JSON with this schema:
{{
  "summary": "string",
  "key_points": [{{"text": "string", "citations": ["S1","S2"]}}],
  "concepts": ["string"],
  "limitations": ["string"]
}}

Rules:
- Respond in French.
- Every key_point MUST have at least 1 citation.
- Citations MUST reference only the provided tags S1..Sn.
- Do NOT add extra keys. Do NOT add markdown. JSON only.
"""


def analysis_rag(req: QueryRequest) -> AnalysisResponse:
    embedder = OllamaEmbeddingsClient()
    store = QdrantStore()

    selected = retrieve_candidates(
        store=store,
        embedder=embedder,
        question=req.question,
        doc_id=req.doc_id,
        tags=req.tags,
        top_k=req.top_k,
        candidates_k=max(30, req.top_k * 6),
        min_score=req.min_score,
        multiquery_n=3,
    )

    contexts = []
    citations = []
    scores = []

    MAX_CONTEXT_CHARS = 6000
    used_chars = 0

    for i, c in enumerate(selected, start=1):
        payload = c["payload"]
        text = c["text"]

        doc_id = payload.get("doc_id", "")
        page = int(payload.get("page", -1))
        chunk_id = payload.get("chunk_id", str(getattr(c["hit"], "id", "")))
        score = float(getattr(c["hit"], "score", 0.0) or 0.0)

        if used_chars + len(text) > MAX_CONTEXT_CHARS:
            break

        tag = f"S{i}"
        contexts.append({"tag": f"[{tag}] (doc={doc_id}, page={page})", "text": text})
        used_chars += len(text)

        citations.append(Citation(source_id=tag, doc_id=doc_id, page=page, chunk_id=chunk_id))
        scores.append(score)

    if not contexts:
        return AnalysisResponse(
            summary="Aucune information pertinente trouvée dans les documents indexés.",
            key_points=[],
            concepts=[],
            limitations=["Aucune source récupérée pour cette question."],
            confidence=0.0,
            sources=[],
        )

    llm = OllamaLLMClient(model="llama3.1:8b")
    prompt = build_prompt(req.question, contexts)

    raw = llm.generate(prompt, max_tokens=550).strip()

    try:
        block = _extract_json_block(raw)
        data = _safe_json_loads(block)
    except Exception:
        # Second pass repair
        try:
            data = _repair_json_with_llm(llm, raw)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"LLM JSON parsing failed even after repair. Error: {str(e)}",
            )

    # validate + normalize
    key_points = [KeyPoint(**kp) for kp in data.get("key_points", [])]

    return AnalysisResponse(
        summary=data.get("summary", ""),
        key_points=key_points,
        concepts=data.get("concepts", []),
        limitations=data.get("limitations", []),
        confidence=_confidence_from_scores(scores),
        sources=citations,
    )