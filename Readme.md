# OpenVINO + LangChain Agent MVP

このMVPは `OpenVINO/Qwen3-8B-int8-ov` を前提にしたエージェント構成です。
現在は **インターネット検索なし** で、`文書作成` を実装済みです。

## 実装済み機能
- `document_create_tool`
  - 入力: `title`, `content`, `format(md|txt)`, `output_dir(任意)`
  - 出力: 保存先 `saved_path`
- `file_search`（ローカル検索の最小実装）
- `OpenVINO/Qwen3-8B-int8-ov` 用のLLMラッパー（遅延ロード）

## セットアップ
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 実行例（文書作成）
```powershell
python -m app.main --title "調査メモ" --content "OpenVINOでMVP作成" --format md --output-dir notes
```

生成先は既定で `workspace` 配下です。

## テスト
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## 主なファイル
- `app/tools/document_create.py`: 文書作成ツール本体
- `app/agent/runner.py`: MVPエージェント
- `app/llm/openvino_qwen.py`: OpenVINO Qwenラッパー
- `app/main.py`: CLIエントリ
