from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from app.agent.runner import MVPAgent


class FakePlanner:
    def __init__(self, decision: dict):
        self._decision = decision

    def plan(self, user_prompt: str) -> dict:
        return self._decision


class BrokenPlanner:
    def plan(self, user_prompt: str) -> dict:
        raise ValueError("Planner returned invalid JSON")


class RuntimeBrokenPlanner:
    def plan(self, user_prompt: str) -> dict:
        raise RuntimeError("Missing dependencies. Install: pip install transformers optimum-intel openvino")


class AgentAutoToolSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("workspace")
        self.base.mkdir(exist_ok=True)
        (self.base / "notes").mkdir(exist_ok=True)
        (self.base / "notes" / "sample.md").write_text("# sample\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self.base.exists():
            shutil.rmtree(self.base)

    def test_runs_document_create_from_planner_decision(self) -> None:
        planner = FakePlanner(
            {
                "action": "use_tool",
                "tool_name": "document_create_tool",
                "arguments": {
                    "title": "議事録",
                    "content": "内容",
                    "format": "md",
                    "output_dir": "notes",
                },
            }
        )
        agent = MVPAgent(planner=planner)
        result = agent.run_prompt("dummy")

        self.assertEqual(result.data["selected_tool"], "document_create_tool")
        saved_path = Path(result.data["tool_output"]["saved_path"])
        self.assertTrue(saved_path.exists())

    def test_runs_file_search_from_planner_decision(self) -> None:
        planner = FakePlanner(
            {
                "action": "use_tool",
                "tool_name": "file_search_tool",
                "arguments": {
                    "root_path": "workspace/notes",
                    "pattern": "*.md",
                    "max_results": 5,
                },
            }
        )
        agent = MVPAgent(planner=planner)
        result = agent.run_prompt("dummy")

        self.assertEqual(result.data["selected_tool"], "file_search_tool")
        self.assertGreaterEqual(len(result.data["tool_output"]), 1)

    def test_returns_direct_response_when_planner_responds(self) -> None:
        planner = FakePlanner({"action": "respond", "answer": "これはツール不要です"})
        agent = MVPAgent(planner=planner)
        result = agent.run_prompt("dummy")

        self.assertEqual(result.message, "これはツール不要です")
        self.assertIsNone(result.data["selected_tool"])

    def test_falls_back_when_planner_output_is_invalid(self) -> None:
        agent = MVPAgent(planner=BrokenPlanner())
        result = agent.run_prompt("app以下のpythonファイルを教えて")

        self.assertEqual(result.data["selected_tool"], "file_search_tool")
        self.assertIn("fallback planner used", result.message)
        self.assertIn("invalid JSON", result.data["fallback_reason"])

    def test_runtime_error_falls_back(self) -> None:
        agent = MVPAgent(planner=RuntimeBrokenPlanner())
        result = agent.run_prompt("app以下のpythonファイルを教えて")
        self.assertEqual(result.data["selected_tool"], "file_search_tool")
        self.assertIn("fallback planner used", result.message)


if __name__ == "__main__":
    unittest.main()
