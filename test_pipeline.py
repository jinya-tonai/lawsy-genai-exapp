"""
Lawsy パイプライン単体テスト
実行方法: python test_pipeline.py
"""

import os
import sys

import dotenv

dotenv.load_dotenv()

QUERY = "個人情報保護法における第三者提供の要件"

# -----------------------------------------------------------------------
# Step 1: エンコーダ・リトリーバのロード
# -----------------------------------------------------------------------
print("\n[Step 1] エンコーダ・リトリーバをロード中...")
try:
    from lawsy.app.utils.preload import load_text_encoder, load_vector_search_article_retriever

    encoder = load_text_encoder()
    retriever = load_vector_search_article_retriever()
    print("  OK: encoder =", type(encoder).__name__)
    print("  OK: retriever =", type(retriever).__name__)
    print("  OK: vector_dim =", retriever.vector_dim)
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

# -----------------------------------------------------------------------
# Step 2: LM ロード
# -----------------------------------------------------------------------
print("\n[Step 2] LM をロード中...")
try:
    from lawsy.app.utils.lm import load_lm

    lm_name = os.getenv("LAWSY_LM", "openai/gpt-4o-mini")
    lm = load_lm(lm_name)
    print("  OK:", lm_name)
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

# -----------------------------------------------------------------------
# Step 3: QueryRefiner
# -----------------------------------------------------------------------
print("\n[Step 3] QueryRefiner...")
try:
    from lawsy.ai.query_refiner import QueryRefiner

    result = QueryRefiner(lm=lm)(query=QUERY)
    refined = result.refined_query
    print("  OK:", refined)
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

# -----------------------------------------------------------------------
# Step 4: Web 検索
# -----------------------------------------------------------------------
print("\n[Step 4] Web 検索...")
try:
    from lawsy.app.utils.web_retreiver import load_web_retriever

    web_engine = os.getenv("LAWSY_WEB_SEARCH_ENGINE", "DuckDuckGo")
    web_retriever = load_web_retriever(web_engine)
    hits = web_retriever.search(refined, k=3)
    print(f"  OK: {len(hits)} 件")
    for h in hits[:2]:
        print(f"    - {h.title}")
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

# -----------------------------------------------------------------------
# Step 5: QueryExpander
# -----------------------------------------------------------------------
print("\n[Step 5] QueryExpander...")
try:
    from lawsy.ai.query_expander import QueryExpander

    web_text = "\n\n".join([f"[{i}] {h.title}\n{h.snippet}" for i, h in enumerate(hits, 1)])
    exp_result = QueryExpander(lm=lm)(query=QUERY, web_search_results=web_text)
    topics = exp_result.topics
    print(f"  OK: {len(topics)} トピック展開")
    for t in topics[:2]:
        print(f"    - {t}")
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

# -----------------------------------------------------------------------
# Step 6: ベクトル検索
# -----------------------------------------------------------------------
print("\n[Step 6] 法令ベクトル検索...")
try:
    expanded = [QUERY] + topics
    query_vecs = encoder.get_query_embeddings(expanded)
    articles = retriever.search(query_vecs[0], k=5)
    print(f"  OK: {len(articles)} 件")
    for a in articles[:2]:
        print(f"    - {a.title}")
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

# -----------------------------------------------------------------------
# Step 7: OutlineCreater
# -----------------------------------------------------------------------
print("\n[Step 7] OutlineCreater...")
try:
    from lawsy.ai.outline_creater import OutlineCreater

    refs = [f"[{i}] {a.title}\n{a.snippet[:500]}" for i, a in enumerate(articles, 1)]
    outline_result = OutlineCreater(lm=lm)(query=QUERY, topics=topics, references=refs)
    outline = outline_result.outline
    print(f"  OK: '{outline.title}' / セクション {len(outline.section_outlines)} 個")
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

# -----------------------------------------------------------------------
# Step 8: StreamSectionWriter（1セクションだけ）
# -----------------------------------------------------------------------
print("\n[Step 8] StreamSectionWriter（同期/非同期確認）...")
try:
    import asyncio
    from lawsy.ai.report_writer import StreamSectionWriter

    writer = StreamSectionWriter(lm=lm)
    sec = outline.section_outlines[0]
    gen = writer(QUERY, refs[0], sec.to_text())

    if hasattr(gen, "__aiter__"):
        print("  → 非同期ジェネレータ")

        async def collect_async():
            text = ""
            async for chunk in gen:
                text += chunk
            return text

        text = asyncio.run(collect_async())
    else:
        print("  → 同期ジェネレータ")
        text = "".join(gen)

    print(f"  OK: {len(text)} 文字生成")
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

# -----------------------------------------------------------------------
# Step 9: StreamConclusionWriter
# -----------------------------------------------------------------------
print("\n[Step 9] StreamConclusionWriter（同期/非同期確認）...")
try:
    from lawsy.ai.report_writer import StreamConclusionWriter

    draft = "# テスト\n" + text
    cw = StreamConclusionWriter(lm)
    gen = cw(QUERY, draft)

    if hasattr(gen, "__aiter__"):
        print("  → 非同期ジェネレータ")

        async def collect_conclusion():
            t = ""
            async for chunk in gen:
                t += chunk
            return t

        conclusion = asyncio.run(collect_conclusion())
    else:
        print("  → 同期ジェネレータ")
        conclusion = "".join(gen)

    print(f"  OK: {len(conclusion)} 文字生成")
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

print("\n" + "=" * 50)
print("全ステップ通過。api.py を起動できます。")
print("=" * 50)
