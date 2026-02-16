# OpenVINO + LangChain Agent MVP

このMVPは `OpenVINO/Qwen3-8B-int8-ov` と `LangGraph` を前提にしたエージェント構成です。
現在は **インターネット検索なし** で、`文書作成` と `ローカル検索` を実装済みです。
LLM実行の互換セットは `torch==2.10.0 / optimum-intel==1.25.2 / transformers==4.53.3` で固定しています。

## 実装済み機能
- `document_create_tool`
  - 入力: `title`, `content`, `format(md|txt)`, `output_dir(任意)`
  - 出力: 保存先 `saved_path`
- `file_search_tool`
  - 入力: `root_path`, `pattern`, `max_results`
  - 出力: `path`, `size`, `mtime` の配列
  - `root_path=this_pc` または「このコンピュータ」でPC全体検索
- `MVPAgent.run_prompt(prompt)`
  - `OpenVINO/Qwen3-8B-int8-ov` による **LLM tool-calling** で1ターン1ツールを自動選択して実行
  - 実行フロー: `plan -> (tool | respond) -> finalize` のLangGraph

## セットアップ
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

既存環境で互換性エラーが出る場合（推奨）:
```powershell
pip uninstall -y torch optimum-intel transformers openvino huggingface-hub
pip install --no-cache-dir -r requirements.txt
```

`PIP_NO_INDEX=1` が設定されている環境では、利用可能な社内/ローカルインデックスに依存します。
その場合は以下で現在設定を確認してください。
```powershell
python -m pip config list
```

`langgraph` が未導入の場合でも、内部フォールバック実行器で同じフローを維持して実行します。

`MODEL_ID`がローカルに存在しない場合、初回実行時にHugging Faceから自動ダウンロードします。
キャッシュ先を固定したい場合は `MODEL_CACHE_DIR` を指定してください。
```powershell
$env:MODEL_ID="OpenVINO/Qwen3-8B-int8-ov"
$env:MODEL_CACHE_DIR="C:\\models\\hf-cache"
```

モデルの事前取得（推奨）:
```powershell
python -m app.main download-model
```

認証が必要なモデルの場合:
```powershell
$env:HF_TOKEN="hf_xxx"
python -m app.main download-model
```

## 実行例（エージェント自動選択）
`chat` はLLMプランナーが `file_search_tool` / `document_create_tool` のどちらか1つを選んで実行します。

文書作成を自動選択:
```powershell
python -m app.main chat --prompt "議事録を作成して notesフォルダ に md で保存して"
```

ローカル検索を自動選択:
```powershell
python -m app.main chat --prompt "app以下のpythonファイルを教えて"
python -m app.main chat --prompt "このコンピュータの中から *.py を検索して 20件 返して"
```

## 手動実行（デバッグ用）
```powershell
python -m app.main create --title "調査メモ" --content "OpenVINOでMVP作成" --format md --output-dir notes
python -m app.main search --root-path app --pattern "*.py" --max-results 20
python -m app.main search --root-path this_pc --pattern "*.py" --max-results 20
```

生成先の許可ルートは既定で `workspace` 配下です。

## テスト
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## 主なファイル
- `app/agent/runner.py`: LLMプランナー + 1ターン1ツール実行ロジック
- `app/tools/document_create.py`: 文書作成ツール
- `app/tools/file_search.py`: ローカル検索ツール
- `app/main.py`: CLIエントリ（chat/create/search）
