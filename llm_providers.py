# -*- coding: utf-8 -*-
"""
LLM provider abstraction: same prompts and workflow for OpenAI or HuggingFace (e.g. Qwen).
"""
from abc import ABC, abstractmethod
from typing import Any, Optional

# Lazy imports per provider to avoid loading heavy deps when not used
def _get_openai_llm(
    model: str,
    temperature: float,
    api_key: Optional[str],
    openai_api_base: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    from langchain_openai import ChatOpenAI
    params = dict(model=model, temperature=temperature, api_key=api_key, **kwargs)
    if openai_api_base:
        params["openai_api_base"] = openai_api_base.rstrip("/") + "/"
    return ChatOpenAI(**params)


def _get_huggingface_llm(model: str, temperature: float, api_key: Optional[str], **kwargs) -> Any:
    """Use langchain-huggingface (ChatHuggingFace) or fallback to local pipeline."""
    try:
        from langchain_huggingface import ChatHuggingFace
        return ChatHuggingFace(
            model=model,
            temperature=temperature,
            huggingfacehub_api_token=api_key,
            **kwargs,
        )
    except ImportError:
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            raise ImportError(
                "HuggingFace LLM requires: pip install langchain-huggingface (or transformers torch langchain-community)"
            )
        tokenizer = AutoTokenizer.from_pretrained(model, **kwargs.get("tokenizer_kwargs", {}))
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=kwargs.get("device_map", "auto"),
        )
        pipe = pipeline(
            "text-generation",
            model=model_obj,
            tokenizer=tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", 2048),
            temperature=temperature,
            do_sample=temperature > 0,
        )
        return HuggingFacePipeline(pipeline=pipe)


class BaseLLMAdapter(ABC):
    """Base for an LLM that can be used with LangChain chains (prompt | llm)."""

    @abstractmethod
    def get_langchain_model(self, temperature: float = 0) -> Any:
        """Return a LangChain Runnable (e.g. BaseChatModel) for use in chains."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return provider/model name for logging and cost tracking."""
        pass


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI or OpenAI-compatible gateway (e.g. Smallcase AI Gateway)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.openai_api_base = openai_api_base
        self.extra_kwargs = kwargs

    def get_langchain_model(self, temperature: float = 0) -> Any:
        return _get_openai_llm(
            self.model_name,
            temperature,
            self.api_key,
            openai_api_base=self.openai_api_base,
            **self.extra_kwargs,
        )

    def get_name(self) -> str:
        return f"openai:{self.model_name}"


class HuggingFaceAdapter(BaseLLMAdapter):
    """HuggingFace models (e.g. Qwen from HuggingFace)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.extra_kwargs = kwargs

    def get_langchain_model(self, temperature: float = 0) -> Any:
        return _get_huggingface_llm(
            self.model_name,
            temperature,
            self.api_key,
            **self.extra_kwargs,
        )

    def get_name(self) -> str:
        return f"huggingface:{self.model_name}"


def get_llm(
    provider: str = "openai",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMAdapter:
    """
    Factory: return an LLM adapter for the given provider.
    - provider: "openai" | "huggingface"
    - model_name: e.g. "gpt-4o-mini" or "Qwen/Qwen2.5-7B-Instruct"
    - api_key: used for OpenAI or HuggingFace token (or set in .env).
    """
    if provider.lower() == "openai":
        from config import OPENAI_API_BASE, OPENAI_API_KEY, DEFAULT_OPENAI_MODEL
        key = api_key or OPENAI_API_KEY
        model = model_name or DEFAULT_OPENAI_MODEL
        base = kwargs.pop("openai_api_base", None) or OPENAI_API_BASE
        return OpenAIAdapter(
            model_name=model,
            api_key=key,
            openai_api_base=base if base else None,
            **kwargs,
        )
    if provider.lower() in ("huggingface", "hf"):
        from config import DEFAULT_HF_MODEL, HUGGINGFACEHUB_API_TOKEN
        model = model_name or DEFAULT_HF_MODEL
        key = api_key or HUGGINGFACEHUB_API_TOKEN
        return HuggingFaceAdapter(model_name=model, api_key=key, **kwargs)
    raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'huggingface'.")
