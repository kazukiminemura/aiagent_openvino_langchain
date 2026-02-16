from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from app.config import MODEL_CACHE_DIR, MODEL_ID


@dataclass
class OpenVINOQwenConfig:
    model_id: str = MODEL_ID
    model_cache_dir: str = MODEL_CACHE_DIR
    max_new_tokens: int = 512
    temperature: float = 0.2


class OpenVINOQwen:
    """Lazy OpenVINO text-generation wrapper for OpenVINO/Qwen3-8B-int8-ov."""

    def __init__(self, cfg: OpenVINOQwenConfig | None = None) -> None:
        self.cfg = cfg or OpenVINOQwenConfig()
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return

        self._patch_torch_onnx_compat()

        try:
            from transformers import AutoTokenizer, pipeline
            try:
                from optimum.intel.openvino import OVModelForCausalLM
            except Exception:
                from optimum.intel import OVModelForCausalLM
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LLM backend unavailable (missing dependencies or version mismatch). "
                "Try: pip install -U transformers optimum-intel openvino"
            ) from exc

        model_source = self._resolve_model_source()
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        model = OVModelForCausalLM.from_pretrained(model_source, trust_remote_code=True)
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            do_sample=self.cfg.temperature > 0,
        )

    def _patch_torch_onnx_compat(self) -> None:
        """Patch torch.onnx.symbolic_opset14 private symbols for newer torch versions."""
        try:
            import torch.onnx.symbolic_opset14 as compat
            from torch.onnx._internal.torchscript_exporter import symbolic_opset14 as internal
        except Exception:
            return

        required = [
            "_attention_scale",
            "_causal_attention_mask",
            "_onnx_symbolic",
            "_type_utils",
            "jit_utils",
            "symbolic_helper",
        ]
        for name in required:
            if not hasattr(compat, name) and hasattr(internal, name):
                setattr(compat, name, getattr(internal, name))

    def _resolve_model_source(self) -> str:
        model_id = self.cfg.model_id

        if Path(model_id).exists():
            return str(Path(model_id).resolve())

        try:
            from huggingface_hub import snapshot_download
        except Exception:
            return model_id

        cache_dir = self.cfg.model_cache_dir.strip() or None
        kwargs = {"repo_id": model_id}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        token = os.getenv("HF_TOKEN", "").strip()
        if token:
            kwargs["token"] = token

        try:
            return snapshot_download(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download model '{model_id}'. "
                "Check network/authentication, or set MODEL_ID to a local path."
            ) from exc

    def ensure_model_downloaded(self) -> str:
        """Resolve and download the model if needed, returning local model path or model id."""
        return self._resolve_model_source()

    def invoke(self, prompt: str) -> str:
        self._load()
        assert self._pipe is not None
        out = self._pipe(prompt)
        if not out:
            return ""
        generated = out[0].get("generated_text", "")
        return generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
