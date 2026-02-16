from __future__ import annotations

from dataclasses import dataclass
import re

from app.tools.document_create import create_document
from app.tools.file_search import file_search


@dataclass
class AgentResult:
    message: str
    data: dict | list | None = None


class MVPAgent:
    """Small MVP agent focused on document creation and local file search."""

    def create_document(self, title: str, content: str, format: str = "md", output_dir: str | None = None) -> AgentResult:
        data = create_document(title=title, content=content, format=format, output_dir=output_dir)
        return AgentResult(message=f"Document created: {data['saved_path']}", data=data)

    def search_files(self, root_path: str = ".", pattern: str = "*.md", max_results: int = 20) -> AgentResult:
        data = file_search(root_path=root_path, pattern=pattern, max_results=max_results)
        return AgentResult(message=f"Found {len(data)} file(s)", data=data)

    def run_prompt(self, prompt: str) -> AgentResult:
        tool_name = self._select_tool(prompt)
        if tool_name == "file_search_tool":
            params = self._extract_search_params(prompt)
            result = self.search_files(**params)
            return AgentResult(
                message=f"Auto selected: {tool_name}. {result.message}",
                data={"selected_tool": tool_name, "tool_input": params, "tool_output": result.data},
            )

        params = self._extract_create_params(prompt)
        result = self.create_document(**params)
        return AgentResult(
            message=f"Auto selected: document_create_tool. {result.message}",
            data={"selected_tool": "document_create_tool", "tool_input": params, "tool_output": result.data},
        )

    def _select_tool(self, prompt: str) -> str:
        text = prompt.lower()
        create_keywords = ["作成", "保存", "まとめ", "書いて", "文書", "ドキュメント", "メモ", "議事録", "report"]
        search_keywords = ["検索", "探し", "見つけ", "一覧", "どこ", "find", "search"]

        create_score = sum(1 for k in create_keywords if k in text)
        search_score = sum(1 for k in search_keywords if k in text)

        if ("ファイル" in prompt and ("教えて" in prompt or "一覧" in prompt or "表示" in prompt)) or "*.py" in text or "python" in text:
            search_score += 2

        if search_score > create_score:
            return "file_search_tool"
        return "document_create_tool"

    def _extract_search_params(self, prompt: str) -> dict[str, str | int]:
        lower = prompt.lower()

        pattern_match = re.search(r"(\*\.[a-zA-Z0-9]+)", prompt)
        if pattern_match:
            pattern = pattern_match.group(1)
        elif ".py" in lower or "python" in lower:
            pattern = "*.py"
        elif "txt" in lower:
            pattern = "*.txt"
        elif "md" in lower or "markdown" in lower:
            pattern = "*.md"
        else:
            pattern = "*"

        root_path = "."
        if any(token in prompt for token in ["このコンピュータ", "PC全体", "pc全体"]):
            root_path = "this_pc"
        else:
            path_match = re.search(r"([A-Za-z0-9_.\\/-]+)\s*(?:フォルダ|folder|配下|以下)", prompt)
            if path_match:
                root_path = path_match.group(1)

        max_results = 20
        max_match = re.search(r"(\d+)\s*(?:件|個|results?)", lower)
        if max_match:
            max_results = max(1, min(200, int(max_match.group(1))))

        return {"root_path": root_path, "pattern": pattern, "max_results": max_results}

    def _extract_create_params(self, prompt: str) -> dict[str, str | None]:
        lower = prompt.lower()
        fmt = "txt" if "txt" in lower else "md"

        output_dir = None
        path_match = re.search(r"([A-Za-z0-9_.\\/-]+)\s*(?:フォルダ|folder|配下|以下)", prompt)
        if path_match:
            output_dir = path_match.group(1)

        quoted = re.findall(r"[\"'「](.*?)[\"'」]", prompt)
        if len(quoted) >= 2:
            title = quoted[0].strip() or "Agent_Note"
            content = quoted[1].strip() or prompt.strip()
        elif len(quoted) == 1:
            title = quoted[0].strip() or "Agent_Note"
            content = prompt.strip()
        else:
            title = self._derive_title(prompt)
            content = prompt.strip()

        return {
            "title": title,
            "content": content,
            "format": fmt,
            "output_dir": output_dir,
        }

    def _derive_title(self, prompt: str) -> str:
        cleaned = re.sub(r"\s+", " ", prompt).strip()
        if not cleaned:
            return "Agent_Note"
        return cleaned[:40]
