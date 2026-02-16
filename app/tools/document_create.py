from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from pydantic import BaseModel, Field

from app.config import ALLOWED_OUTPUT_ROOT, SUPPORTED_FORMATS


class DocumentCreateInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    format: str = Field(default="md", pattern="^(md|txt)$")
    output_dir: str | None = None


def _sanitize_title(title: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", title.strip())
    safe = safe.strip("._")
    return safe or "document"


def _resolve_output_dir(output_dir: str | None) -> Path:
    base = ALLOWED_OUTPUT_ROOT
    base.mkdir(parents=True, exist_ok=True)

    if not output_dir:
        return base

    target = Path(output_dir)
    if not target.is_absolute():
        target = base / target

    target = target.resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"output_dir must be inside allowed root: {base}")

    target.mkdir(parents=True, exist_ok=True)
    return target


def create_document(title: str, content: str, format: str = "md", output_dir: str | None = None) -> dict[str, str]:
    fmt = format.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format: {fmt}; expected one of {sorted(SUPPORTED_FORMATS)}")

    out_dir = _resolve_output_dir(output_dir)
    file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_sanitize_title(title)}.{fmt}"
    path = out_dir / file_name

    if fmt == "md":
        body = f"# {title}\n\n{content.rstrip()}\n"
    else:
        body = f"{title}\n{'=' * len(title)}\n\n{content.rstrip()}\n"

    path.write_text(body, encoding="utf-8")
    return {"saved_path": str(path), "format": fmt}


try:
    from langchain_core.tools import StructuredTool
except Exception:  # pragma: no cover
    StructuredTool = None


def build_document_create_tool():
    """Return a LangChain StructuredTool when langchain-core is installed."""
    if StructuredTool is None:
        return None

    return StructuredTool.from_function(
        func=create_document,
        name="document_create_tool",
        description=(
            "Create a local document in md or txt format. "
            "Inputs: title, content, format(md|txt), optional output_dir."
        ),
        args_schema=DocumentCreateInput,
    )
