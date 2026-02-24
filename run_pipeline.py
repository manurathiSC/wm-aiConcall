# -*- coding: utf-8 -*-
"""
Run the full earnings-call pipeline: load PDF -> context summary -> subchunks -> themes -> overall summary -> embeddings.
Cost is tracked and written to output. Keys are read from .env.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import pandas as pd

from config import OPENAI_API_KEY
from cost_tracker import CostTracker
from document_loader import get_recursive_text_splitter, set_docs
from embeddings_module import get_embeddings_for_column
from process_concall import ProcessConcall


# Default embedding model (can be overridden via env or args)
DEFAULT_EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "openai")
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")


def run_pipeline(
    pdf_path: str,
    output_dir: str,
    llm_provider: str = "openai",
    llm_model_name: str | None = None,
    temperature: float = 0.0,
    embedding_model_type: str = DEFAULT_EMBEDDING_MODEL_TYPE,
    embedding_model_name: str | None = DEFAULT_EMBEDDING_MODEL_NAME,
    run_name: str | None = None,
    compute_embeddings: bool = False,
) -> dict:
    """
    Run full pipeline and return summary dict (mainDf, plannedDf, executedDf, cost_tracker, paths).
    Theme extraction (up to 10 analyst themes) and overall summary always run.
    Set compute_embeddings=True to additionally generate embeddings for parentChunk and childChunk columns.
    """
    pdf_path = Path(pdf_path).resolve()
    output_dir = Path(output_dir).resolve() if Path(output_dir).is_absolute() else _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    run_name = run_name or pdf_path.stem
    cost_tracker = CostTracker(run_name=run_name)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    start_time = time.time()

    # 1) Load documents and split
    documents = set_docs(stored_file_path=str(pdf_path))
    x_df = get_recursive_text_splitter(documents)

    # 2) Process with chosen LLM
    process = ProcessConcall(
        cost_tracker=cost_tracker,
        provider=llm_provider,
        model_name=llm_model_name,
    )

    doc_df = process.get_context_summary(x_df, tmp=temperature)
    main_df, _ = process.get_subchunk(doc_df, tmp=temperature)

    # Rename columns to match downstream (childChunk, parentChunk)
    rename_map = {}
    if "Doc" in main_df.columns:
        rename_map["Doc"] = "origRawChunk"
    if "Page" in main_df.columns:
        rename_map["Page"] = "page"
    if "summary_y" in main_df.columns:
        rename_map["summary_y"] = "parentChunk"
    if "summary_x" in main_df.columns:
        rename_map["summary_x"] = "childChunk"
    if "summary" in main_df.columns and "summary_x" not in main_df.columns:
        rename_map["summary"] = "childChunk"
    main_df = main_df.rename(columns=rename_map)
    if "parentChunk" not in main_df.columns and "summary_y" in main_df.columns:
        main_df = main_df.rename(columns={"summary_y": "parentChunk"})
    if "childChunk" not in main_df.columns and "summary" in main_df.columns:
        main_df = main_df.rename(columns={"summary": "childChunk"})

    # 3) Theme extraction: negative, positive, planned, executed
    neg_output, _ = process.process_neg(main_df, tmp=temperature)
    pos_output, _ = process.process_pos(main_df, tmp=temperature)
    planned_output, _ = process.process_plan(main_df, tmp=temperature)
    executed_output, _ = process.process_exec(main_df, tmp=temperature)

    # Build plannedDf and executedDf
    if planned_output is not None and not (isinstance(planned_output, pd.DataFrame) and planned_output.empty):
        if isinstance(planned_output, dict) and "planned_actions" in planned_output:
            planned_actions = planned_output["planned_actions"]
        elif isinstance(planned_output, list):
            planned_actions = planned_output
        else:
            planned_actions = []
        planned_df = pd.DataFrame({"planned_actions": planned_actions}) if planned_actions else pd.DataFrame(columns=["planned_actions"])
    else:
        planned_df = pd.DataFrame(columns=["planned_actions"])

    if executed_output is not None and not (isinstance(executed_output, pd.DataFrame) and executed_output.empty):
        if isinstance(executed_output, dict) and "executed_actions" in executed_output:
            executed_actions = executed_output["executed_actions"]
        elif isinstance(executed_output, list):
            executed_actions = executed_output
        else:
            executed_actions = []
        executed_df = pd.DataFrame({"executed_actions": executed_actions}) if executed_actions else pd.DataFrame(columns=["executed_actions"])
    else:
        executed_df = pd.DataFrame(columns=["executed_actions"])

    # 4) Embeddings: mainDf (parentChunk, childChunk) — only if compute_embeddings=True
    if compute_embeddings:
        emb_type = embedding_model_type or "openai"
        emb_name = embedding_model_name

        def _safe_embed(df: pd.DataFrame, col: str) -> pd.DataFrame:
            try:
                return get_embeddings_for_column(df, col, model_type=emb_type, model_name=emb_name)
            except Exception as e:
                print(f"Warning: Skipping embeddings for column {col!r}: {e}")
                return df

        if "parentChunk" in main_df.columns:
            main_df = _safe_embed(main_df, "parentChunk")
        if "childChunk" in main_df.columns:
            main_df = _safe_embed(main_df, "childChunk")
    else:
        print("Skipping embeddings (compute_embeddings=False).")

    # 5) Analyst theme extraction — always runs
    cluster_themes = process.extract_all_themes(main_df, tmp=temperature)
    print(f"  {len(cluster_themes)} themes extracted.")

    # 6) Overall summary — always runs
    overall_summary = process.extract_overall_summary(cluster_themes, tmp=temperature)

    # 7) Save outputs
    def _df_to_json_path(df: pd.DataFrame, path: Path) -> None:
        records = df.to_dict(orient="records")
        def _serialize(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if isinstance(obj, (list, tuple)):
                return [_serialize(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            return obj
        records = [_serialize(r) for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    main_path = output_dir / f"Main_{run_name}.json"
    plan_path = output_dir / f"Plan_{run_name}.json"
    exec_path = output_dir / f"Exec_{run_name}.json"
    neg_path = output_dir / f"Neg_{run_name}.json"
    pos_path = output_dir / f"Pos_{run_name}.json"
    cluster_path = output_dir / f"Clusters_{run_name}.json"
    summary_path = output_dir / f"OverallSummary_{run_name}.txt"
    cost_path = output_dir / f"cost_{run_name}.json"

    _df_to_json_path(main_df, main_path)
    main_df.to_excel(output_dir / f"Main_{run_name}.xlsx", index=False)
    _df_to_json_path(planned_df, plan_path)
    _df_to_json_path(executed_df, exec_path)
    with open(cluster_path, "w", encoding="utf-8") as f:
        json.dump(cluster_themes, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(overall_summary)

    def _theme_list_to_json(obj: Any, path: Path) -> None:
        if isinstance(obj, list):
            lst = obj
        elif isinstance(obj, dict) and obj:
            lst = list(obj.values())[0]
        elif isinstance(obj, pd.DataFrame) and obj.empty:
            lst = []
        else:
            lst = []
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lst, f, indent=2, ensure_ascii=False)

    _theme_list_to_json(neg_output, neg_path)
    _theme_list_to_json(pos_output, pos_path)

    cost_summary = cost_tracker.get_summary_for_output()
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, indent=2)

    elapsed = time.time() - start_time
    cost_tracker.print_summary()
    print(f"Total pipeline time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return {
        "main_df": main_df,
        "planned_df": planned_df,
        "executed_df": executed_df,
        "overall_summary": overall_summary,
        "cost_tracker": cost_tracker,
        "elapsed_seconds": elapsed,
        "paths": {
            "main_json": str(main_path),
            "plan_json": str(plan_path),
            "exec_json": str(exec_path),
            "neg_json": str(neg_path),
            "pos_json": str(pos_path),
            "cluster_json": str(cluster_path),
            "summary_txt": str(summary_path),
            "cost_json": str(cost_path),
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run earnings call pipeline on a PDF.")
    parser.add_argument("pdf", nargs="?", default=None, help="Path to PDF (default: Concalls/DLF_Jan26.pdf)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("--provider", default="openai", choices=("openai", "huggingface"), help="LLM provider")
    parser.add_argument("--model", default=None, help="LLM model name (default from env)")
    parser.add_argument("--temp", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--embed-type", default=DEFAULT_EMBEDDING_MODEL_TYPE, help="Embedding model type")
    parser.add_argument("--embed-model", default=None, help="Embedding model name")
    parser.add_argument("--embeddings", action="store_true", default=False, help="Compute embeddings for parentChunk and childChunk")
    args = parser.parse_args()

    pdf_path = args.pdf or str(_PROJECT_ROOT / "Concalls" / "DLF_Jan26.pdf")
    if not OPENAI_API_KEY and args.provider == "openai":
        print("Warning: OPENAI_API_KEY not set. Set it in .env for OpenAI.")

    run_pipeline(
        pdf_path=pdf_path,
        output_dir=args.output_dir,
        llm_provider=args.provider,
        llm_model_name=args.model,
        temperature=args.temp,
        embedding_model_type=args.embed_type,
        embedding_model_name=args.embed_model or DEFAULT_EMBEDDING_MODEL_NAME,
        run_name=Path(pdf_path).stem,
        compute_embeddings=args.embeddings,
    )


if __name__ == "__main__":
    main()
