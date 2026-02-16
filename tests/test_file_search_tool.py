from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from app.agent.runner import MVPAgent
from app.tools.file_search import _expand_search_roots, file_search


class FileSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("workspace")
        self.base.mkdir(exist_ok=True)
        (self.base / "notes").mkdir(exist_ok=True)
        (self.base / "notes" / "a.md").write_text("# a\n", encoding="utf-8")
        (self.base / "notes" / "b.txt").write_text("b\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self.base.exists():
            shutil.rmtree(self.base)

    def test_file_search_returns_matches(self) -> None:
        results = file_search(root_path="workspace/notes", pattern="*.md", max_results=10)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["path"].endswith("a.md"))

    def test_expand_search_roots_this_pc(self) -> None:
        roots = _expand_search_roots("this_pc")
        self.assertGreaterEqual(len(roots), 1)
        self.assertTrue(all(root.is_absolute() for root in roots))

    def test_agent_can_call_search_tool(self) -> None:
        agent = MVPAgent()
        result = agent.search_files(root_path="workspace/notes", pattern="*.md", max_results=10)
        self.assertEqual(result.message, "Found 1 file(s)")
        self.assertIsInstance(result.data, list)


if __name__ == "__main__":
    unittest.main()
