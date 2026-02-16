from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from app.agent.runner import MVPAgent


class AgentAutoToolSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("workspace")
        self.base.mkdir(exist_ok=True)
        (self.base / "notes").mkdir(exist_ok=True)
        (self.base / "notes" / "sample.md").write_text("# sample\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self.base.exists():
            shutil.rmtree(self.base)

    def test_auto_selects_document_create(self) -> None:
        agent = MVPAgent()
        prompt = "議事録を作成して notesフォルダ に md で保存して"
        result = agent.run_prompt(prompt)

        self.assertEqual(result.data["selected_tool"], "document_create_tool")
        saved_path = Path(result.data["tool_output"]["saved_path"])
        self.assertTrue(saved_path.exists())

    def test_auto_selects_file_search(self) -> None:
        agent = MVPAgent()
        prompt = "workspace/notes配下で *.md を検索して 5件 返して"
        result = agent.run_prompt(prompt)

        self.assertEqual(result.data["selected_tool"], "file_search_tool")
        self.assertGreaterEqual(len(result.data["tool_output"]), 1)

    def test_auto_selects_file_search_for_python_listing(self) -> None:
        agent = MVPAgent()
        prompt = "app以下のpythonファイルを教えて"
        result = agent.run_prompt(prompt)

        self.assertEqual(result.data["selected_tool"], "file_search_tool")
        self.assertEqual(result.data["tool_input"]["root_path"], "app")
        self.assertEqual(result.data["tool_input"]["pattern"], "*.py")
        self.assertGreaterEqual(len(result.data["tool_output"]), 1)

    def test_auto_uses_this_pc_keyword(self) -> None:
        agent = MVPAgent()
        params = agent._extract_search_params("このコンピュータの中から *.py を検索して")
        self.assertEqual(params["root_path"], "this_pc")


if __name__ == "__main__":
    unittest.main()
