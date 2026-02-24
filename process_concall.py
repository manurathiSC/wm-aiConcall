# -*- coding: utf-8 -*-
"""
Earnings call processing pipeline: context summary -> subchunks -> filtered themes (neg/pos/exec/plan).
Uses LLM abstraction and cost tracking.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import Runnable
from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm

from cost_tracker import CostTracker
from llm_providers import BaseLLMAdapter, get_llm
from prompts import PromptCollections


def _ensure_list_of_dicts(out: Any) -> List[dict]:
    """Normalize LLM output to list of dicts (for subchunk parsing)."""
    if isinstance(out, list):
        if all(isinstance(item, dict) for item in out):
            return out
        if all(isinstance(item, list) for item in out):
            return sum(out, [])
        return list(out) if out else []
    if isinstance(out, dict):
        return [out]
    return []


class ProcessConcall:
    def __init__(
        self,
        llm_adapter: Optional[BaseLLMAdapter] = None,
        cost_tracker: Optional[CostTracker] = None,
        provider: str = "openai",
        model_name: Optional[str] = None,
    ):
        self.prmpt_obj = PromptCollections()
        self.cost_tracker = cost_tracker or CostTracker(run_name="concall")
        self.llm_adapter = llm_adapter or get_llm(provider=provider, model_name=model_name)
        self._model_name = self.llm_adapter.get_name()

    def _chain_with_parser(self, prompt_template: Runnable, temperature: float = 0) -> Runnable:
        """Build chain: prompt | llm | json parser."""
        llm = self.llm_adapter.get_langchain_model(temperature=temperature)
        return prompt_template | llm | JsonOutputParser()

    def _chain_no_parser(self, prompt_template: Runnable, temperature: float = 0) -> Runnable:
        """Build chain: prompt | llm (returns message with .content)."""
        llm = self.llm_adapter.get_langchain_model(temperature=temperature)
        return prompt_template | llm

    def _chain_str_parser(self, prompt_template: Runnable, temperature: float = 0) -> Runnable:
        """Build chain: prompt | llm | str parser (returns plain string)."""
        llm = self.llm_adapter.get_langchain_model(temperature=temperature)
        return prompt_template | llm | StrOutputParser()

    def get_context_summary(
        self,
        document_df: pd.DataFrame,
        tmp: float = 0,
    ) -> pd.DataFrame:
        """Produce parent-chunk summaries for each row (with prev/next context)."""
        def process_row(i: int) -> str:
            doc = document_df.loc[i, "origRawChunk"]
            try:
                if i == 0:
                    context_txt = self.prmpt_obj.get_context_summary("F")
                    pr = self.prmpt_obj.get_langchain_supported_prompt(["doc", "nxtDoc"], context_txt)
                    chain = self._chain_no_parser(pr, tmp)
                    with get_openai_callback() as cb:
                        res = chain.invoke({
                            "doc": doc,
                            "nxtDoc": document_df.loc[i + 1, "origRawChunk"] + " " + document_df.loc[i + 2, "origRawChunk"],
                        })
                        self.cost_tracker.add_from_openai_callback("context_summary", cb, self._model_name)
                    return getattr(res, "content", str(res))
                elif i == 1:
                    context_txt = self.prmpt_obj.get_context_summary("F")
                    pr = self.prmpt_obj.get_langchain_supported_prompt(["prevDoc", "doc", "nxtDoc"], context_txt)
                    chain = self._chain_no_parser(pr, tmp)
                    with get_openai_callback() as cb:
                        res = chain.invoke({
                            "prevDoc": document_df.loc[i - 1, "origRawChunk"],
                            "doc": doc,
                            "nxtDoc": document_df.loc[i + 1, "origRawChunk"] + " " + document_df.loc[i + 2, "origRawChunk"],
                        })
                        self.cost_tracker.add_from_openai_callback("context_summary", cb, self._model_name)
                    return getattr(res, "content", str(res))
                elif i == document_df.shape[0] - 2:
                    context_txt = self.prmpt_obj.get_context_summary("L")
                    pr = self.prmpt_obj.get_langchain_supported_prompt(["prevDoc", "doc", "nxtDoc"], context_txt)
                    chain = self._chain_no_parser(pr, tmp)
                    with get_openai_callback() as cb:
                        res = chain.invoke({
                            "prevDoc": document_df.loc[i - 2, "origRawChunk"] + " " + document_df.loc[i - 1, "origRawChunk"],
                            "doc": doc,
                            "nxtDoc": document_df.loc[i + 1, "origRawChunk"],
                        })
                        self.cost_tracker.add_from_openai_callback("context_summary", cb, self._model_name)
                    return getattr(res, "content", str(res))
                elif i == document_df.shape[0] - 1:
                    context_txt = self.prmpt_obj.get_context_summary("L")
                    pr = self.prmpt_obj.get_langchain_supported_prompt(["doc", "prevDoc"], context_txt)
                    chain = self._chain_no_parser(pr, tmp)
                    with get_openai_callback() as cb:
                        res = chain.invoke({
                            "doc": document_df.iloc[i]["origRawChunk"],
                            "prevDoc": document_df.iloc[i - 1]["origRawChunk"] + " " + document_df.loc[i - 2, "origRawChunk"],
                        })
                        self.cost_tracker.add_from_openai_callback("context_summary", cb, self._model_name)
                    return getattr(res, "content", str(res))
                else:
                    context_txt = self.prmpt_obj.get_context_summary("M")
                    pr = self.prmpt_obj.get_langchain_supported_prompt(["prevDoc", "doc", "nxtDoc"], context_txt)
                    chain = self._chain_no_parser(pr, tmp)
                    with get_openai_callback() as cb:
                        res = chain.invoke({
                            "prevDoc": document_df.loc[i - 2, "origRawChunk"] + " " + document_df.loc[i - 1, "origRawChunk"],
                            "doc": doc,
                            "nxtDoc": document_df.loc[i + 1, "origRawChunk"] + " " + document_df.loc[i + 2, "origRawChunk"],
                        })
                        self.cost_tracker.add_from_openai_callback("context_summary", cb, self._model_name)
                    return getattr(res, "content", str(res))
            except Exception as e:
                return f"Error: {str(e)}"

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                tqdm(
                    executor.map(process_row, range(document_df.shape[0])),
                    total=document_df.shape[0],
                    desc="Context Summary",
                )
            )
        document_df = document_df.copy()
        document_df["parentChunk"] = results
        return document_df

    def get_subchunk(self, x_df: pd.DataFrame, tmp: float = 0) -> Tuple[pd.DataFrame, None]:
        """Extract subchunks (themes with sentiment/tags) per row."""
        sub_prompt = self.prmpt_obj.get_subchunks()
        pr = self.prmpt_obj.get_langchain_supported_prompt(["doc"], sub_prompt)
        chain = self._chain_with_parser(pr, tmp)

        out_list: List[dict] = []
        ind_list: List[int] = []

        def process_subchunk(row: pd.Series) -> Any:
            try:
                with get_openai_callback() as cb:
                    out = chain.invoke({"doc": row["origRawChunk"]})
                    self.cost_tracker.add_from_openai_callback("subchunk", cb, self._model_name)
                return out
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=8) as executor:
            rows = [row for _, row in x_df.iterrows()]
            results = list(tqdm(executor.map(process_subchunk, rows), total=x_df.shape[0], desc="Subchunks"))

        for ind, out in zip(x_df.index, results):
            items = _ensure_list_of_dicts(out)
            for _ in items:
                ind_list.append(ind)
            out_list.extend(items)

        if not out_list:
            main_df = pd.DataFrame(columns=["childChunk", "positive", "negative", "neutral", "tag", "keywords"])
            main_df = pd.merge(main_df, x_df, left_index=True, right_index=True, how="right")
            return main_df, None

        main_df = pd.DataFrame(out_list, index=ind_list)
        main_df = pd.merge(main_df, x_df, left_index=True, right_index=True)
        return main_df, None

    def get_filtered_df(self, main_df: pd.DataFrame, key: str) -> List[dict]:
        """Filter mainDf by key and return list of {childChunk, parentChunk} for theme prompts."""
        if "childChunk" not in main_df.columns or "parentChunk" not in main_df.columns:
            # Allow summary_x / summary_y if not yet renamed
            if "summary_x" in main_df.columns and "summary_y" in main_df.columns:
                main_df = main_df.rename(columns={"summary_x": "childChunk", "summary_y": "parentChunk"})
            else:
                return []
        if key == "negative":
            filt = main_df[main_df["negative"] >= 0.5]
        elif key == "positive":
            filt = main_df[main_df["positive"] >= 0.7]
        elif key == "executed":
            filt = main_df[main_df["tag"] == "E"]
        elif key == "planned":
            filt = main_df[main_df["tag"] == "P"]
        else:
            return []
        if filt.empty:
            return []
        result = (
            filt.groupby(filt.index, sort=False)
            .agg(childChunk=("childChunk", list), parentChunk=("parentChunk", "first"))
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        return result

    def process_neg(self, main_df: pd.DataFrame, tmp: float = 0) -> Tuple[Any, None]:
        neg_prompt = self.prmpt_obj.get_neg_theme()
        pr = self.prmpt_obj.get_langchain_supported_prompt(["docs"], neg_prompt)
        chain = self._chain_with_parser(pr, tmp)
        result = self.get_filtered_df(main_df, "negative")
        if not result:
            return pd.DataFrame(), None
        with get_openai_callback() as cb:
            out = chain.invoke({"docs": result})
            self.cost_tracker.add_from_openai_callback("process_neg", cb, self._model_name)
        return out, None

    def process_pos(self, main_df: pd.DataFrame, tmp: float = 0) -> Tuple[Any, None]:
        pos_prompt = self.prmpt_obj.get_pos_theme()
        pr = self.prmpt_obj.get_langchain_supported_prompt(["docs"], pos_prompt)
        chain = self._chain_with_parser(pr, tmp)
        result = self.get_filtered_df(main_df, "positive")
        if not result:
            return pd.DataFrame(), None
        with get_openai_callback() as cb:
            out = chain.invoke({"docs": result})
            self.cost_tracker.add_from_openai_callback("process_pos", cb, self._model_name)
        return out, None

    def process_exec(self, main_df: pd.DataFrame, tmp: float = 0) -> Tuple[Any, None]:
        exec_prompt = self.prmpt_obj.get_executed_theme()
        pr = self.prmpt_obj.get_langchain_supported_prompt(["docs"], exec_prompt)
        chain = self._chain_with_parser(pr, tmp)
        result = self.get_filtered_df(main_df, "executed")
        if not result:
            return pd.DataFrame(), None
        with get_openai_callback() as cb:
            out = chain.invoke({"docs": result})
            self.cost_tracker.add_from_openai_callback("process_exec", cb, self._model_name)
        return out, None

    def extract_all_themes(
        self,
        main_df: pd.DataFrame,
        tmp: float = 0,
    ) -> List[dict]:
        """
        Single LLM call over all unique parentChunks from main_df.
        Returns up to 10 distinctive analyst-relevant themes, each with 4-5 bullet points.
        """
        theme_prompt = self.prmpt_obj.get_all_themes()
        pr = self.prmpt_obj.get_langchain_supported_prompt(["docs"], theme_prompt)
        chain = self._chain_with_parser(pr, tmp)

        parent_chunks = (
            main_df.drop_duplicates(subset=["parentChunk"])["parentChunk"]
            .dropna()
            .tolist()
        )
        docs = "\n\n".join(f"[Section {i+1}]\n{text}" for i, text in enumerate(parent_chunks))

        print(f"Extracting themes from {len(parent_chunks)} sections...")
        try:
            with get_openai_callback() as cb:
                out = chain.invoke({"docs": docs})
                self.cost_tracker.add_from_openai_callback("extract_themes", cb, self._model_name)
            return out if isinstance(out, list) else []
        except Exception as e:
            print(f"Warning: Theme extraction failed: {e}")
            return []

    def extract_overall_summary(self, themes: List[dict], tmp: float = 0) -> str:
        """
        Single LLM call that takes extracted themes and returns a 150-180 word
        analyst narrative paragraph (headline + anchor figures + tone + watchpoint).
        """
        if not themes:
            return ""
        summary_prompt = self.prmpt_obj.get_overall_summary()
        pr = self.prmpt_obj.get_langchain_supported_prompt(["themes"], summary_prompt)
        chain = self._chain_str_parser(pr, tmp)
        print("Generating overall summary...")
        try:
            with get_openai_callback() as cb:
                out = chain.invoke({"themes": themes})
                self.cost_tracker.add_from_openai_callback("overall_summary", cb, self._model_name)
            return out.strip()
        except Exception as e:
            print(f"Warning: Overall summary failed: {e}")
            return ""

    def process_plan(self, main_df: pd.DataFrame, tmp: float = 0) -> Tuple[Any, None]:
        plan_prompt = self.prmpt_obj.get_planned_theme()
        pr = self.prmpt_obj.get_langchain_supported_prompt(["docs"], plan_prompt)
        chain = self._chain_with_parser(pr, tmp)
        result = self.get_filtered_df(main_df, "planned")
        if not result:
            return pd.DataFrame(), None
        with get_openai_callback() as cb:
            out = chain.invoke({"docs": result})
            self.cost_tracker.add_from_openai_callback("process_plan", cb, self._model_name)
        return out, None
