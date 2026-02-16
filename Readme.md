# OpenVINO + LangChain Agent MVP

このMVPは `OpenVINO/Qwen3-8B-int8-ov` を前提にしたエージェント構成です。
現在は **インターネット検索なし** で、`文書作成` と `ローカル検索` を実装済みです。

## 実装済み機能
- `document_create_tool`
  - 入力: `title`, `content`, `format(md|txt)`, `output_dir(任意)`
  - 出力: 保存先 `saved_path`
- `file_search_tool`
  - 入力: `root_path`, `pattern`, `max_results`
  - 出力: `path`, `size`, `mtime` の配列
  - `root_path=this_pc` または「このコンピュータ」でPC全体検索
- `MVPAgent.run_prompt(prompt)`
  - 自然言語プロンプトからツールを自動選択して実行

## セットアップ
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 実行例（エージェント自動選択）
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
- `app/agent/runner.py`: 自動ツール選択ロジック
- `app/tools/document_create.py`: 文書作成ツール
- `app/tools/file_search.py`: ローカル検索ツール
- `app/main.py`: CLIエントリ（chat/create/search）
