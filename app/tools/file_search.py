from __future__ import annotations

import os
from pathlib import Path
import string

from pydantic import BaseModel, Field


class FileSearchInput(BaseModel):
    root_path: str = Field(default=".")
    pattern: str = Field(default="*.md")
    max_results: int = Field(default=20, ge=1, le=200)


def _expand_search_roots(root_path: str) -> list[Path]:
    key = root_path.strip().lower()
    if key in {"this_pc", "computer", "all", "このコンピュータ", "pc全体"}:
        if os.name == "nt":
            roots = [Path(f"{drive}:/") for drive in string.ascii_uppercase if Path(f"{drive}:/").exists()]
            return roots or [Path(Path.cwd().anchor)]
        return [Path("/")]

    target = Path(root_path).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    else:
        target = target.resolve()
    return [target]


def file_search(root_path: str = ".", pattern: str = "*.md", max_results: int = 20) -> list[dict[str, str]]:
    roots = _expand_search_roots(root_path)
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    for root in roots:
        if not root.exists():
            continue
        try:
            iterator = root.rglob(pattern)
            for path in iterator:
                try:
                    if not path.is_file():
                        continue
                    real = str(path.resolve())
                    if real in seen:
                        continue
                    stat = path.stat()
                    results.append(
                        {
                            "path": real,
                            "size": str(stat.st_size),
                            "mtime": str(int(stat.st_mtime)),
                        }
                    )
                    seen.add(real)
                    if len(results) >= max_results:
                        return results
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            continue

    return results


try:
    from langchain_core.tools import StructuredTool
except Exception:  # pragma: no cover
    StructuredTool = None


def build_file_search_tool():
    """Return a LangChain StructuredTool when langchain-core is installed."""
    if StructuredTool is None:
        return None

    return StructuredTool.from_function(
        func=file_search,
        name="file_search_tool",
        description=(
            "Search files on this computer. "
            "Use root_path=this_pc for whole computer search. "
            "Inputs: root_path, pattern, max_results."
        ),
        args_schema=FileSearchInput,
    )
