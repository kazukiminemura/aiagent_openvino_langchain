from __future__ import annotations

from pathlib import Path
import os

MODEL_ID = os.getenv("MODEL_ID", "OpenVINO/Qwen3-8B-int8-ov")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "")
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "AUTO:NPU,GPU")
ALLOWED_OUTPUT_ROOT = Path(os.getenv("ALLOWED_OUTPUT_ROOT", "workspace")).resolve()
DEFAULT_DOC_FORMAT = os.getenv("DEFAULT_DOC_FORMAT", "md")

SUPPORTED_FORMATS = {"md", "txt"}
