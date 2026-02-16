from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from app.tools.document_create import create_document


class DocumentCreateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("workspace")
        self.base.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        if self.base.exists():
            shutil.rmtree(self.base)

    def test_create_markdown_document(self) -> None:
        result = create_document("MVP Report", "This is a test.", "md", "notes")
        saved = Path(result["saved_path"])
        self.assertTrue(saved.exists())
        self.assertEqual(saved.suffix, ".md")
        self.assertIn("MVP Report", saved.read_text(encoding="utf-8"))

    def test_unsupported_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_document("bad", "x", "pdf")


if __name__ == "__main__":
    unittest.main()
