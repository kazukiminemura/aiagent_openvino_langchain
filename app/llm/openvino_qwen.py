from __future__ import annotations

from dataclasses import dataclass

from app.config import MODEL_ID


@dataclass
class OpenVINOQwenConfig:
    model_id: str = MODEL_ID
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

        try:
            from transformers import AutoTokenizer, pipeline
            from optimum.intel.openvino import OVModelForCausalLM
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependencies. Install: transformers optimum[openvino] openvino"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, trust_remote_code=True)
        model = OVModelForCausalLM.from_pretrained(self.cfg.model_id, trust_remote_code=True)
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            do_sample=self.cfg.temperature > 0,
        )

    def invoke(self, prompt: str) -> str:
        self._load()
        assert self._pipe is not None
        out = self._pipe(prompt)
        if not out:
            return ""
        generated = out[0].get("generated_text", "")
        return generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
