from __future__ import annotations

import unittest

from app.agent.runner import LLMToolPlanner


class StubLLM:
    def __init__(self, response: str):
        self.response = response

    def invoke(self, prompt: str) -> str:
        return self.response


class LLMToolPlannerTests(unittest.TestCase):
    def test_parses_use_tool_json(self) -> None:
        llm = StubLLM('{"action":"use_tool","tool_name":"file_search_tool","arguments":{"root_path":"app","pattern":"*.py","max_results":3}}')
        planner = LLMToolPlanner(llm=llm)
        decision = planner.plan("app以下のpythonファイル")

        self.assertEqual(decision["action"], "use_tool")
        self.assertEqual(decision["tool_name"], "file_search_tool")

    def test_parses_json_inside_markdown_fence(self) -> None:
        llm = StubLLM('```json\n{"action":"respond","answer":"ok"}\n```')
        planner = LLMToolPlanner(llm=llm)
        decision = planner.plan("hello")

        self.assertEqual(decision["action"], "respond")
        self.assertEqual(decision["answer"], "ok")

    def test_raises_on_invalid_action(self) -> None:
        llm = StubLLM('{"action":"unknown"}')
        planner = LLMToolPlanner(llm=llm)
        with self.assertRaises(ValueError):
            planner.plan("x")


if __name__ == "__main__":
    unittest.main()
