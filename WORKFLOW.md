# Pipeline workflow and file map

High-level flowchart of building blocks and data flow.

```mermaid
flowchart TB
    subgraph inputs["Inputs"]
        PDF[("PDF file\n(Concalls/*.pdf)")]
        ENV[(".env\nAPI keys")]
    end

    subgraph config["Config"]
        CONFIG[("config.py\nOPENAI_API_KEY\nOPENAI_DIRECT_API_KEY\nHUGGINGFACEHUB_API_TOKEN")]
    end

    subgraph load["Document loading"]
        SET_DOCS["document_loader.set_docs()\nPDF → raw docs"]
        SPLIT["document_loader.get_recursive_text_splitter()\nDocs → x_df (origRawChunk, page)"]
    end

    subgraph llm["LLM layer"]
        ADAPTER["llm_providers.get_llm()\nOpenAI (gateway) or HuggingFace"]
        PROMPTS["prompts.PromptCollections\nparent chunks · subchunks\nneg / pos / exec / plan\nall_themes · overall_summary"]
    end

    subgraph cost["Cost tracking"]
        TRACKER["cost_tracker.CostTracker\nPer-step tokens & cost → cost_<name>.json"]
    end

    subgraph process["Processing pipeline"]
        CTX["get_context_summary()\nx_df → doc_df  (parentChunk)"]
        SUB["get_subchunk()\ndoc_df → main_df  (childChunk + sentiment + tag)"]
        RENAME["Rename columns\nsummary_x → childChunk\nsummary_y → parentChunk"]
        NEGPOS["process_neg() / process_pos()\nmain_df → negative / positive themes"]
        PLANEXEC["process_plan() / process_exec()\nmain_df → planned / executed actions"]
        THEMES["extract_all_themes()\nall parentChunks → up to 10 analyst themes\n(single LLM call)"]
        SUMM["extract_overall_summary()\nthemes → 150–180 word narrative paragraph\n(single LLM call)"]
    end

    subgraph embed["Embeddings  (optional: --embeddings flag)"]
        EMBED_MOD["embeddings_module.get_embeddings_for_column()\nvia OPENAI_DIRECT_API_KEY  (bypasses gateway)"]
        MAIN_EMB["main_df: parentChunk_embeddings\n         childChunk_embeddings"]
    end

    subgraph out["Outputs  (output/)"]
        MAIN_OUT[("Main_<name>.json / .xlsx\nparentChunk · childChunk · sentiment · tag")]
        PLANEXEC_OUT[("Plan_<name>.json\nExec_<name>.json")]
        NEGPOS_OUT[("Neg_<name>.json\nPos_<name>.json")]
        CLUSTER_OUT[("Clusters_<name>.json\nup to 10 themes with 4–5 bullet points")]
        SUMMARY_OUT[("OverallSummary_<name>.txt\nanalyst narrative paragraph")]
        COST_OUT[("cost_<name>.json\ntokens & cost per step + total")]
    end

    ENV --> CONFIG
    PDF --> SET_DOCS
    SET_DOCS --> SPLIT
    SPLIT --> CTX
    CONFIG --> ADAPTER
    CONFIG --> EMBED_MOD
    ADAPTER --> CTX
    ADAPTER --> SUB
    ADAPTER --> NEGPOS
    ADAPTER --> PLANEXEC
    ADAPTER --> THEMES
    ADAPTER --> SUMM
    PROMPTS --> CTX
    PROMPTS --> SUB
    PROMPTS --> NEGPOS
    PROMPTS --> PLANEXEC
    PROMPTS --> THEMES
    PROMPTS --> SUMM
    TRACKER --> CTX
    TRACKER --> SUB
    TRACKER --> NEGPOS
    TRACKER --> PLANEXEC
    TRACKER --> THEMES
    TRACKER --> SUMM
    CTX --> SUB
    SUB --> RENAME
    RENAME --> NEGPOS
    RENAME --> PLANEXEC
    RENAME --> THEMES
    RENAME --> MAIN_EMB
    EMBED_MOD --> MAIN_EMB
    THEMES --> SUMM
    MAIN_EMB --> MAIN_OUT
    PLANEXEC --> PLANEXEC_OUT
    NEGPOS --> NEGPOS_OUT
    THEMES --> CLUSTER_OUT
    SUMM --> SUMMARY_OUT
    TRACKER --> COST_OUT
```

## File roles

| File | Role |
|------|------|
| **config.py** | Loads `.env`; exposes `OPENAI_API_KEY`, `OPENAI_DIRECT_API_KEY` (for embeddings), `HUGGINGFACEHUB_API_TOKEN`, default model names. |
| **llm_providers.py** | `get_llm(provider, model_name)` → OpenAI or HuggingFace adapter; same interface for all chains. |
| **cost_tracker.py** | `CostTracker`: records each LLM call (tokens, cost) keyed by step name; `get_summary_for_output()` writes per-step and total to `cost_<name>.json`. |
| **prompts.py** | `PromptCollections`: all prompt strings — parent chunks, subchunks, neg/pos/executed/planned themes, `get_all_themes` (up to 10 analyst themes), `get_overall_summary` (narrative paragraph). |
| **document_loader.py** | `set_docs()`, `get_recursive_text_splitter()`: load PDF/folder/URL and chunk into `x_df`. |
| **process_concall.py** | `ProcessConcall`: context summary → subchunks → neg/pos/exec/plan themes → `extract_all_themes` (prompt-based, single LLM call) → `extract_overall_summary`. Uses LLM adapter + cost tracker throughout. |
| **embeddings_module.py** | `get_embeddings_for_column(df, col, model_type, model_name)`: adds `{col}_embeddings` column. Uses `OPENAI_DIRECT_API_KEY` to bypass the gateway for embedding calls. |
| **run_pipeline.py** | Entrypoint: PDF → full pipeline → theme extraction → overall summary → (optional) embeddings → save all outputs and cost JSON. |

## Data flow (simplified)

1. **PDF** → `set_docs` → **documents** → `get_recursive_text_splitter` → **x_df** (origRawChunk, page).
2. **x_df** → `get_context_summary` (LLM, parallel) → **doc_df** (+ parentChunk).
3. **doc_df** → `get_subchunk` (LLM, parallel) → **main_df** (childChunk, positive, negative, neutral, tag, keywords, parentChunk).
4. **main_df** → `process_neg` / `process_pos` → **Neg_\*.json** / **Pos_\*.json**.
5. **main_df** → `process_plan` / `process_exec` → **Plan_\*.json** / **Exec_\*.json**.
6. **main_df** (all unique parentChunks) → `extract_all_themes` (single LLM call) → **Clusters_\*.json** (up to 10 themes, 4–5 bullet points each).
7. **themes** → `extract_overall_summary` (single LLM call) → **OverallSummary_\*.txt** (150–180 word analyst narrative).
8. *(Optional, `--embeddings`)* **main_df** → `get_embeddings_for_column` (parentChunk, childChunk) → **Main_\*.json** with embedding vectors.
9. All outputs + **cost_\*.json** saved to `output/`.
