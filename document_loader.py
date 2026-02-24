# -*- coding: utf-8 -*-
"""Document loading and text splitting utilities."""
import os
from typing import List, Optional

import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def set_docs(
    stored_file_path: str = "",
    stored_folder_path: str = "",
    loaded_file: Optional[str] = None,
    weblink: str = "",
) -> List:
    """Load documents from PDF file, folder, or URL."""
    docs = []
    if stored_file_path:
        docs.extend(PyPDFLoader(stored_file_path).load())
    if stored_folder_path:
        docs.extend(DirectoryLoader(stored_folder_path).load())
    if loaded_file is not None:
        docs.extend(PyPDFLoader(loaded_file).load())
    if weblink:
        docs.extend(WebBaseLoader(weblink).load())
    return docs


def get_recursive_text_splitter(
    documents: List,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> pd.DataFrame:
    """Split documents into chunks and return a DataFrame with Doc and Page columns."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    doc_df = pd.DataFrame()
    doc_df["origRawChunk"] = [doc.page_content for doc in docs]
    doc_df["page"] = [doc.metadata.get("page", 0) for doc in docs]
    return doc_df
