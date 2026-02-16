from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class FileSearchInput(BaseModel):
    root_path: str = Field(default="workspace")
    pattern: str = Field(default="*.md")
    max_results: int = Field(default=20, ge=1, le=200)


def file_search(root_path: str = "workspace", pattern: str = "*.md", max_results: int = 20) -> list[dict[str, str]]:
    root = Path(root_path).resolve()
    if not root.exists():
        return []

    results: list[dict[str, str]] = []
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        stat = path.stat()
        results.append(
            {
                "path": str(path),
                "size": str(stat.st_size),
                "mtime": str(int(stat.st_mtime)),
            }
        )
        if len(results) >= max_results:
            break

    return results
