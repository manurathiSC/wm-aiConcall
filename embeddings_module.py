# -*- coding: utf-8 -*-
"""
Extract embeddings for a DataFrame column. Supports multiple embedding models
(e.g. OpenAI, HuggingFace sentence-transformers, BGE, Instructor).
"""
from typing import Any, List, Optional, Union

import pandas as pd


def get_embeddings_for_column(
    df: pd.DataFrame,
    column_name: str,
    model_type: str = "openai",
    model_name: Optional[str] = None,
    batch_size: int = 32,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Add an embedding column for the given text column.

    - df: DataFrame containing the column.
    - column_name: Name of the column with text to embed.
    - model_type: "openai" | "huggingface" | "sentence_transformers" | "bge" | "instructor"
    - model_name: Model id (e.g. "text-embedding-3-small", "sentence-transformers/all-MiniLM-L6-v2").
    - batch_size: Batch size for embedding calls (where supported).
    - **kwargs: Passed to the underlying embedder (e.g. api_key, device).

    Returns a copy of df with an extra column: `{column_name}_embeddings` (list of floats per row).
    """
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name!r} not in DataFrame. Columns: {list(df.columns)}")

    texts = df[column_name].astype(str).fillna("").tolist()

    if model_type.lower() == "openai":
        embeddings = _embed_openai(texts, model_name=model_name, **kwargs)
    elif model_type.lower() in ("huggingface", "sentence_transformers", "sentence_transformers"):
        embeddings = _embed_sentence_transformers(texts, model_name=model_name, **kwargs)
    elif model_type.lower() == "bge":
        embeddings = _embed_bge(texts, model_name=model_name, **kwargs)
    elif model_type.lower() == "instructor":
        embeddings = _embed_instructor(texts, model_name=model_name, **kwargs)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Use openai, huggingface, bge, or instructor."
        )

    out = df.copy()
    out[f"{column_name}_embeddings"] = list(embeddings)
    return out


def _embed_openai(
    texts: List[str],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    from langchain_openai import OpenAIEmbeddings
    from config import OPENAI_API_KEY, OPENAI_API_BASE
    key = api_key or OPENAI_API_KEY
    model = model_name or "text-embedding-3-small"
    base = openai_api_base if openai_api_base is not None else OPENAI_API_BASE
    params = dict(model=model, api_key=key, **kwargs)
    if base:
        params["openai_api_base"] = base.rstrip("/") + "/"
    emb = OpenAIEmbeddings(**params)
    return emb.embed_documents(texts)


def _embed_sentence_transformers(
    texts: List[str],
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer
    model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    st = SentenceTransformer(model, **kwargs)
    arr = st.encode(texts, show_progress_bar=kwargs.get("show_progress_bar", True))
    if arr.ndim == 1:
        return [arr.tolist()]
    return arr.tolist()


def _embed_bge(
    texts: List[str],
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    model = model_name or "BAAI/bge-small-en-v1.5"
    emb = HuggingFaceBgeEmbeddings(model_name=model, **kwargs)
    return emb.embed_documents(texts)


def _embed_instructor(
    texts: List[str],
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    from langchain_community.embeddings import HuggingFaceInstructEmbeddings
    model = model_name or "hkunlp/instructor-base"
    emb = HuggingFaceInstructEmbeddings(model_name=model, **kwargs)
    return emb.embed_documents(texts)
