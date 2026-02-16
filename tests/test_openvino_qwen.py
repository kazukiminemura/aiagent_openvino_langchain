from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from app.llm.openvino_qwen import OpenVINOQwen, OpenVINOQwenConfig


class OpenVINOQwenResolveModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path("workspace") / "model_local"
        self.temp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        base = Path("workspace")
        if base.exists():
            shutil.rmtree(base)

    def test_resolve_model_source_prefers_existing_local_path(self) -> None:
        cfg = OpenVINOQwenConfig(model_id=str(self.temp))
        llm = OpenVINOQwen(cfg=cfg)
        source = llm._resolve_model_source()
        self.assertEqual(source, str(self.temp.resolve()))

    def test_ensure_model_downloaded_uses_resolver(self) -> None:
        cfg = OpenVINOQwenConfig(model_id=str(self.temp))
        llm = OpenVINOQwen(cfg=cfg)
        source = llm.ensure_model_downloaded()
        self.assertEqual(source, str(self.temp.resolve()))

    def test_default_device_is_non_cpu(self) -> None:
        cfg = OpenVINOQwenConfig(model_id=str(self.temp))
        self.assertNotEqual(cfg.device.upper(), "CPU")

    def test_patch_torch_onnx_compat_exposes_required_symbols(self) -> None:
        llm = OpenVINOQwen(cfg=OpenVINOQwenConfig(model_id=str(self.temp)))
        llm._patch_torch_onnx_compat()

        import torch.onnx.symbolic_opset14 as compat

        required = [
            "_attention_scale",
            "_causal_attention_mask",
            "_onnx_symbolic",
            "_type_utils",
            "jit_utils",
            "symbolic_helper",
        ]
        for name in required:
            self.assertTrue(hasattr(compat, name), f"Missing symbol after patch: {name}")


if __name__ == "__main__":
    unittest.main()
