# lawsy-genai-exapp

[Lawsy](https://github.com/digital-go-jp/lawsy) を[デジタル庁ガバメントAI「源内」](https://github.com/digital-go-jp/genai-web)の ExApp として動かすための FastAPI ブリッジです。

Lawsy の法令 Deep Research パイプラインをチームの源内 UI から呼び出せるようになります。

## アーキテクチャ

```
源内 Web UI
    │
    ▼  ExApp プロトコル（POST /requests → GET /status/{id}）
FastAPI ブリッジ（api.py）
    │
    ▼
Lawsy パイプライン
（QueryRefiner → Web検索 → QueryExpander → FAISS検索 → レポート生成）
```

## 前提

- [Lawsy](https://github.com/digital-go-jp/lawsy) がセットアップ済みであること（`make install` 完了）
- Lawsy の公開データ（法令インデックス）がダウンロード済みであること
- Python 3.12 / uv

## セットアップ

```bash
# api.py を Lawsy リポジトリ直下にコピー
cp api.py ~/lawsy/

# FastAPI を Lawsy の venv に追加インストール
~/lawsy/.venv/bin/uv pip install fastapi "uvicorn[standard]"
~/lawsy/.venv/bin/uv pip install -e .

# .env を設定
cat > ~/lawsy/.env <<EOF
OPENAI_API_KEY=sk-...
LAWSY_LM=openai/gpt-4o-mini
LAWSY_WEB_SEARCH_ENGINE=DuckDuckGo
LAWSY_OUTPUT_DIR=./outputs
EOF
```

## 起動

```bash
cd ~/lawsy
LAWSY_OUTPUT_DIR=./outputs .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

## 動作確認

```bash
# リクエスト投入
curl -X POST http://localhost:8000/requests \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"query": "個人情報保護法の第三者提供の要件"}}'

# ステータス確認（2〜3分後）
curl http://localhost:8000/status/<request_id>
```

`"status": "COMPLETED"` になったら `outputs` に Markdown レポートが入っています。

## 源内への登録

源内管理画面の ExApp 登録フォームに `exapp_definition.json` の内容を貼り付けてください。

- **エンドポイント URL**: `http://<サーバーIP>:8000`
- **入力定義**: `exapp_definition.json` を参照

## ブランチ戦略

GitHub Flow を採用しています。

- `main` は常にデプロイ可能な状態を保つ
- 作業は `feature/<機能名>` ブランチで行う
- PR を通じて `main` にマージする
- マージ後はブランチを削除する

## ライセンス

MIT
