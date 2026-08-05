"""
The provider-agnostic model layer.

Every pipeline stage generates through an `LLMClient`, obtained once from
`build_llm_client()`. Which provider that actually is -- local weights,
Anthropic, Groq, a self-hosted vLLM/Ollama/whatever OpenAI-compatible
server -- is decided entirely by `.env` (`LLM_PROVIDER` and friends), not by
any import in pipeline code.
"""

from .base import LLMClient, extract_json_object
from .factory import build_llm_client

__all__ = ["LLMClient", "extract_json_object", "build_llm_client"]
