"""
Lawsy → 源内 ExApp ブリッジ API

源内の ExApp プロトコルに準拠した FastAPI アプリ。
Lawsy の法令 Deep Research を源内のチーム環境から呼び出せるようにする。

起動方法:
    uvicorn api:app --host 0.0.0.0 --port 8000

源内 ExApp への登録:
    エンドポイント URL: https://<your-host>/requests
    入力定義: exapp_definition.json の内容を貼り付ける
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import dotenv
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from lawsy.ai.outline_creater import OutlineCreater
from lawsy.ai.query_expander import QueryExpander
from lawsy.ai.query_refiner import QueryRefiner
from lawsy.ai.report_writer import StreamConclusionWriter, StreamLeadWriter, StreamSectionWriter
from lawsy.app.config import get_config
from lawsy.app.utils.lm import load_lm
from lawsy.app.utils.preload import load_text_encoder, load_vector_search_article_retriever
from lawsy.app.utils.web_retreiver import load_web_retriever
from lawsy.utils.logging import logger

dotenv.load_dotenv()

app = FastAPI(title="Lawsy ExApp API")

_jobs: dict[str, dict[str, Any]] = {}
_encoder = None
_retriever = None


@app.on_event("startup")
async def startup() -> None:
    global _encoder, _retriever
    logger.info("Loading text encoder and vector retriever...")
    _encoder = load_text_encoder()
    _retriever = load_vector_search_article_retriever()
    logger.info("Ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _construct_fusion_query(expanded_queries: list[str]) -> str:
    query = expanded_queries[0]
    topics = expanded_queries[1:]
    return "\n".join(
        [
            "以下の内容に関する法令解説文書を作るにあたって参考になるWebページや法令がほしい",
            "",
            "主題となるクエリー: " + query,
            "関連するトピック:",
        ]
        + ["- " + q for q in topics]
    )


async def _collect(gen) -> str:
    text = ""
    if hasattr(gen, "__aiter__"):
        async for chunk in gen:
            text += chunk
    else:
        for chunk in gen:
            text += chunk
    return text


def _set_progress(job: dict, msg: str) -> None:
    job["progress"] = msg
    job["updated_at"] = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Research pipeline (Streamlit 非依存)
# ---------------------------------------------------------------------------

async def _run_research(job_id: str, query: str) -> None:
    job = _jobs[job_id]
    job["status"] = "IN_PROGRESS"

    try:
        lm_name = os.getenv("LAWSY_LM", "openai/gpt-4o-mini")
        web_engine = os.getenv("LAWSY_WEB_SEARCH_ENGINE", "DuckDuckGo")
        lm = load_lm(lm_name)
        web_retriever = load_web_retriever(web_engine)

        # 1. クエリ精錬
        if len(query) >= 64:
            _set_progress(job, "クエリーを検索向けに変換中...")
            refined_query: str = QueryRefiner(lm=lm)(query=query).refined_query
            logger.info(f"[{job_id}] refined: {refined_query}")
        else:
            refined_query = query

        # 2. Web 検索
        web_search_results = []
        if get_config("free_web_search_enabled", True):
            _set_progress(job, "Web 検索中（フリードメイン）...")
            web_search_results.extend(web_retriever.search(refined_query, k=10))

        domains = get_config("web_search_domains") or []
        if domains:
            _set_progress(job, "Web 検索中（ドメイン指定）...")
            web_search_results.extend(web_retriever.search(refined_query, k=10, domains=domains))

        # 3. クエリ展開
        _set_progress(job, "クエリーを展開中...")
        web_results_text = "\n\n".join(
            [f"[{i}] {r.title}\n{r.snippet}" for i, r in enumerate(web_search_results, 1)]
        )
        expander_result = QueryExpander(lm=lm)(query=query, web_search_results=web_results_text)
        expanded_queries = [query] + expander_result.topics

        # 4. 法令ベクトル検索
        _set_progress(job, "法令データベースを検索中...")
        article_results = []
        query_vecs = _encoder.get_query_embeddings(expanded_queries)
        for q, qvec in zip(expanded_queries, query_vecs):
            article_results.extend(_retriever.search(qvec, k=10))
            logger.info(f"[{job_id}] vector search: {q}")

        # 5. リランキング
        _set_progress(job, "ナレッジをリランキング中...")
        url_to_article = {r.url: r for r in article_results}
        unique_articles = list(url_to_article.values())
        unique_web = [r for r in web_search_results if r.url not in url_to_article]

        dim = _retriever.vector_dim
        rich_qvec = _encoder.get_query_embeddings([_construct_fusion_query(expanded_queries)])[0][:dim]
        web_vecs = _encoder.get_document_embeddings(
            [r.title + "\n" + r.snippet for r in unique_web]
        )[:, :dim] if unique_web else np.empty((0, dim))
        article_vecs = np.asarray([_retriever.get_vector(r) for r in unique_articles]) if unique_articles else np.empty((0, dim))

        all_results = unique_web + unique_articles
        if all_results:
            vecs = np.vstack([web_vecs, article_vecs])
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            cossims = vecs.dot(rich_qvec / (np.linalg.norm(rich_qvec) + 1e-9))
            all_results = [all_results[i] for i in np.argsort(cossims)[::-1]]

        # 6. 参照ナレッジ抽出（最大 100k 字）
        references: list[str] = []
        seen: set = set()
        total_len = 0
        for i, result in enumerate(all_results, start=1):
            if result.source_type == "article":
                key = (result.rev_id, result.anchor)
                if key in seen:
                    continue
                body = "\n".join(result.snippet.split("\n")[1:])
                ref = f"[{i}] {result.title}\n{body[:1024]}"
                seen.add(key)
            elif result.source_type == "web":
                if result.url in seen:
                    continue
                ref = f"[{i}] {result.title}\n{result.snippet}"
                seen.add(result.url)
            else:
                continue
            references.append(ref)
            total_len += len(ref)
            if len(seen) >= 200 or total_len >= 100000:
                break

        # 7. アウトライン生成
        _set_progress(job, "アウトラインを生成中...")
        outline = OutlineCreater(lm=lm)(
            query=query, topics=expander_result.topics, references=references
        ).outline
        id2result = {i: r for i, r in enumerate(all_results, 1)}

        # 8. セクション並列生成
        _set_progress(job, "各セクションを生成中...")
        section_writers = [StreamSectionWriter(lm=lm) for _ in outline.section_outlines]
        tasks = []
        for writer, sec_outline in zip(section_writers, outline.section_outlines):
            ref_ids = sorted({rid for sub in sec_outline.subsection_outlines for rid in sub.reference_ids})
            refs = "\n\n".join(
                [f"[{rid}] {id2result[rid].title}\n{id2result[rid].snippet}"
                 for rid in ref_ids if rid in id2result]
            )
            tasks.append(_collect(writer(query, refs, sec_outline.to_text())))

        section_texts: list[str] = list(await asyncio.gather(*tasks))

        report_draft = "\n".join(["# " + outline.title] + section_texts)

        # 9. 結論生成
        _set_progress(job, "結論を生成中...")
        conclusion_writer = StreamConclusionWriter(lm)
        conclusion = await _collect(conclusion_writer(query, report_draft))

        # 10. リード生成
        _set_progress(job, "リードを生成中...")
        lead_writer = StreamLeadWriter(lm=lm)
        lead = await _collect(
            lead_writer(query=query, title=outline.title, draft=report_draft + "\n## 結論\n" + conclusion)
        )

        # 11. レポート組み立て
        report_md = "\n\n".join(
            [
                "# " + outline.title,
                lead,
                *section_texts,
                "## 結論",
                conclusion,
                "---",
                "## 参考文献",
                "\n".join(
                    [f"- [{r.title}]({r.url})" for r in all_results[:30]]
                ),
            ]
        )

        job["status"] = "COMPLETED"
        job["outputs"] = report_md
        _set_progress(job, "処理が完了しました。")
        logger.info(f"[{job_id}] completed")

    except Exception:
        logger.exception(f"[{job_id}] error")
        job["status"] = "ERROR"
        job["error"] = {"message": "処理中にエラーが発生しました。", "details": "ログを確認してください。"}
        _set_progress(job, "エラーが発生しました。")


# ---------------------------------------------------------------------------
# ExApp エンドポイント
# ---------------------------------------------------------------------------

class RequestBody(BaseModel):
    inputs: dict[str, Any]


@app.post("/requests", status_code=202)
async def create_request(
    body: RequestBody,
    x_api_key: str = Header(default=None),
) -> dict[str, Any]:
    query: str = (body.inputs.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query は必須です")

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _jobs[job_id] = {
        "request_id": job_id,
        "status": "PENDING",
        "progress": "リクエストを受け付けました",
        "created_at": now,
        "updated_at": now,
    }
    asyncio.create_task(_run_research(job_id, query))

    return {
        "outputs": "リクエストを受け付けました",
        "request_id": job_id,
        "status": "PENDING",
        "status_url": f"/status/{job_id}",
    }


@app.get("/status/{request_id}")
async def get_status(
    request_id: str,
    x_api_key: str = Header(default=None),
) -> dict[str, Any]:
    job = _jobs.get(request_id)
    if not job:
        raise HTTPException(status_code=404, detail="指定されたリクエストが見つかりません")

    resp: dict[str, Any] = {
        "request_id": job["request_id"],
        "status": job["status"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    if job["status"] == "COMPLETED":
        resp["outputs"] = job["outputs"]
    if job["status"] == "ERROR":
        resp["error"] = job["error"]
    return resp
