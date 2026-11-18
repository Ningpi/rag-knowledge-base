"""检索服务：FastAPI 暴露 /search 与 /ask，索引进程内单例。"""
import os

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel

from rag import Retriever

app = FastAPI(title="rag-knowledge-base")
retriever = Retriever()

LLM_BASE = os.environ.get("LLM_BASE_URL", "")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")


class AskReq(BaseModel):
    question: str
    top_k: int = 5


def llm_answer(question: str, contexts: list[dict]) -> str | None:
    """可选生成层：配置了 OpenAI 兼容接口就调用，否则走抽取式回答。"""
    if not (LLM_BASE and LLM_KEY and LLM_MODEL):
        return None
    ctx = "\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(contexts))
    payload = {
        "model": LLM_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system",
             "content": "只能依据给定资料回答；资料不足就说不知道；每个结论标注来源编号。"},
            {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{question}"},
        ],
    }
    try:
        resp = httpx.post(
            f"{LLM_BASE.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_KEY}"},
            json=payload, timeout=20.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = 5, mode: str = "hybrid"):
    return {"query": q, "hits": retriever.search(q, k=k, mode=mode)}


@app.post("/ask")
def ask(req: AskReq):
    hits = retriever.search(req.question, k=req.top_k)
    if not hits:
        return {"answer": "知识库中未找到相关内容。", "citations": [], "grounded": False}
    generated = llm_answer(req.question, hits)
    citations = [{"id": h["doc_id"], "title": h["title"]} for h in hits]
    if generated:
        return {"answer": generated, "citations": citations, "grounded": True}
    body = "\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    return {"answer": body, "citations": citations, "grounded": True}
