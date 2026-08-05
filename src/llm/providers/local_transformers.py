"""
A local Hugging Face `transformers` vision-language model, served in-process.

This is the no-external-API path: the weights load onto whatever GPU (or CPU)
is available and every call runs locally. It is also the most involved
provider, because it is the only one that has to manage model loading,
dtype/attention-backend fallbacks, and image preprocessing itself — an API
provider gets all of that for free from the service it calls.
"""

import time
from typing import Any, Dict, List, Optional

from ..base import LLMClient, ContentBlock, fit_image, load_image


class LocalTransformersClient(LLMClient):
    """Loads one `transformers` checkpoint and serves every call from it."""

    def __init__(
        self,
        model_id: str,
        fallback_ids: Optional[List[str]] = None,
        dtype: str = "bfloat16",
        use_flash_attention: bool = False,
        device_map: str = "auto",
        hf_token: Optional[str] = None,
        max_image_side: int = 1_280,
        logger=None,
    ):
        super().__init__()
        self.requested_model_id = model_id
        self.fallback_ids = fallback_ids or []
        self.dtype_name = dtype
        self.use_flash_attention = use_flash_attention
        self.device_map = device_map
        self.hf_token = hf_token
        self.max_image_side = max_image_side
        self.log = logger

        self.model = None
        self.processor = None
        self.model_id = None
        self._load()

    # ---------------------------------------------------------------- load --
    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor

        try:  # transformers >= 4.45
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            self.dtype_name, torch.float32
        )

        # transformers renamed `torch_dtype` to `dtype` in 4.56. Getting this
        # wrong is not an error: the unknown keyword is absorbed into **kwargs
        # and the model silently loads in float32 -- twice the VRAM, half the
        # speed, no warning. So pick the spelling this version understands.
        import transformers

        version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kwarg = "dtype" if version >= (4, 56) else "torch_dtype"
        self._log(f"transformers {transformers.__version__} -> using {dtype_kwarg}=")

        backends = (["flash_attention_2"] if self.use_flash_attention else []) + [
            "sdpa", "eager",
        ]

        # If this machine cannot pull the requested checkpoint, step down the
        # fallback ladder rather than dying -- but the model actually loaded
        # is printed in every stage report, so a downgrade can never pass
        # unnoticed.
        candidates = [self.requested_model_id] + [
            m for m in self.fallback_ids if m != self.requested_model_id
        ]

        t0 = time.time()
        last_error = None
        for model_id in candidates:
            for backend in backends:
                try:
                    self._log(f"Loading {model_id} (attn={backend}) ...")
                    self.model = AutoVLM.from_pretrained(
                        model_id,
                        device_map=self.device_map,
                        attn_implementation=backend,
                        token=self.hf_token,
                        **{dtype_kwarg: dtype},
                    )
                    self.processor = AutoProcessor.from_pretrained(
                        model_id, token=self.hf_token)
                    self.model_id = model_id
                    break
                except Exception as exc:
                    last_error = exc
                    self._log(
                        f"  {model_id} / attn={backend}: "
                        f"{type(exc).__name__}: {str(exc)[:150]}", level="warning")
            if self.model is not None:
                break
            self._log(f"Could not load {model_id}; trying the next candidate",
                      level="warning")

        if self.model is None:
            raise RuntimeError(
                f"No usable model. Requested {self.requested_model_id}; none of "
                f"{candidates} could be loaded. Last error: {last_error}")

        if self.model_id != self.requested_model_id:
            self._log(f"RUNNING ON {self.model_id}, NOT the requested "
                      f"{self.requested_model_id}. Set LLM_MODEL once the "
                      f"intended checkpoint is reachable.", level="warning")

        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        self._log(f"Ready in {time.time() - t0:.1f}s: {self.model_id} "
                  f"({n_params / 1e9:.1f}B params)")

    def _log(self, message: str, level: str = "info") -> None:
        if self.log is not None:
            getattr(self.log, level)(message)

    # ------------------------------------------------------------- chatting --
    def _chat(self, content: List[ContentBlock], system: Optional[str],
             max_tokens: int, temperature: float) -> str:
        import torch

        with torch.inference_mode():
            images = []
            blocks: List[Dict[str, Any]] = []

            for block in content:
                if block.get("type") == "image":
                    img = fit_image(load_image(block["image"]).convert("RGB"),
                                    self.max_image_side)
                    images.append(img)
                    blocks.append({"type": "image"})
                else:
                    blocks.append({"type": "text", "text": block.get("text", "")})

            messages: List[Dict[str, Any]] = []
            if system:
                messages.append({"role": "system",
                                 "content": [{"type": "text", "text": system}]})
            messages.append({"role": "user", "content": blocks})

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            proc_kwargs: Dict[str, Any] = {"text": [text], "return_tensors": "pt"}
            if images:
                proc_kwargs["images"] = images
            inputs = self.processor(**proc_kwargs).to(self.model.device)

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                # Passing temperature with do_sample=False is ignored and
                # warns, so only set it when it will actually be used.
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            output_ids = self.model.generate(**inputs, **gen_kwargs)

            trimmed = [out[len(inp):] for inp, out in
                      zip(inputs["input_ids"], output_ids)]
            return self.processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0].strip()

    def cleanup(self) -> None:
        import torch

        del self.model, self.processor
        self.model = self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log("Model released.")
