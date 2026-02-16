from __future__ import annotations

from dataclasses import dataclass

from app.tools.document_create import create_document


@dataclass
class AgentResult:
    message: str
    data: dict | list | None = None


class MVPAgent:
    """Small MVP agent focused on document creation and local file search."""

    def create_document(self, title: str, content: str, format: str = "md", output_dir: str | None = None) -> AgentResult:
        data = create_document(title=title, content=content, format=format, output_dir=output_dir)
        return AgentResult(message=f"Document created: {data['saved_path']}", data=data)
