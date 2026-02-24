# Pipeline workflow and file map

High-level flowchart of building blocks and data flow.

```mermaid
flowchart TB
    subgraph inputs["Inputs"]
        PDF[("PDF file\n(Concalls/DLF_Jan26.pdf)")]
        ENV[(".env\nAPI keys")]
    end

    subgraph config["Config"]
        CONFIG[("config.py\nLoad OPENAI_API_KEY,\nHUGGINGFACEHUB_API_TOKEN")]
    end

    subgraph load["Document loading"]
        SET_DOCS["document_loader.set_docs()\nPDF → raw docs"]
        SPLIT["document_loader.get_recursive_text_splitter()\nDocs → x_df (Doc, Page)"]
    end

    subgraph llm["LLM layer"]
        ADAPTER["llm_providers.get_llm()\nOpenAI or HuggingFace (Qwen)"]
        PROMPTS["prompts.PromptCollections\nParent, subchunk, neg/pos/exec/plan"]
    end

    subgraph cost["Cost tracking"]
        TRACKER["cost_tracker.CostTracker\nPer-call tokens & cost"]
    end

    subgraph process["Processing pipeline"]
        CTX["process_concall.get_context_summary()\nx_df → doc_df (with summary)"]
        SUB["process_concall.get_subchunk()\ndoc_df → main_df"]
        RENAME["Rename columns\nsummary_x→childChunk, summary_y→parentChunk"]
        PLAN["process_concall.process_plan()\nmain_df → planned_actions"]
        EXEC["process_concall.process_exec()\nmain_df → executed_actions"]
    end

    subgraph embed["Embeddings"]
        EMBED_MOD["embeddings_module.get_embeddings_for_column()"]
        MAIN_EMB["main_df: parentChunk, childChunk"]
        PLAN_EMB["planned_df: planned_actions"]
        EXEC_EMB["executed_df: executed_actions"]
    end

    subgraph out["Outputs"]
        MAIN_OUT[("Main_<name>.pkl / .xlsx")]
        PLAN_OUT[("Plan_<name>.pkl")]
        EXEC_OUT[("Exec_<name>.pkl")]
        COST_OUT[("cost_<name>.json")]
    end

    ENV --> CONFIG
    PDF --> SET_DOCS
    SET_DOCS --> SPLIT
    SPLIT --> CTX
    CONFIG --> ADAPTER
    ADAPTER --> CTX
    ADAPTER --> SUB
    ADAPTER --> PLAN
    ADAPTER --> EXEC
    PROMPTS --> CTX
    PROMPTS --> SUB
    PROMPTS --> PLAN
    PROMPTS --> EXEC
    TRACKER --> CTX
    TRACKER --> SUB
    TRACKER --> PLAN
    TRACKER --> EXEC
    CTX --> SUB
    SUB --> RENAME
    RENAME --> PLAN
    RENAME --> EXEC
    PLAN --> PLAN_EMB
    EXEC --> EXEC_EMB
    RENAME --> MAIN_EMB
    EMBED_MOD --> MAIN_EMB
    EMBED_MOD --> PLAN_EMB
    EMBED_MOD --> EXEC_EMB
    MAIN_EMB --> MAIN_OUT
    PLAN_EMB --> PLAN_OUT
    EXEC_EMB --> EXEC_OUT
    TRACKER --> COST_OUT
```

## File roles

| File | Role |
|------|------|
| **config.py** | Loads `.env`; exposes `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, default model names. |
| **llm_providers.py** | `get_llm(provider, model_name)` → OpenAI or HuggingFace adapter; same interface for chains. |
| **cost_tracker.py** | `CostTracker`: records each LLM call (tokens, cost), summary by step, JSON output. |
| **prompts.py** | `PromptCollections`: all prompt strings (parent chunks, subchunks, neg/pos/executed/planned themes). |
| **document_loader.py** | `set_docs()`, `get_recursive_text_splitter()`: load PDF/folder/URL and chunk. |
| **process_concall.py** | `ProcessConcall`: context summary → subchunks → theme extraction; uses LLM adapter + cost tracker. |
| **embeddings_module.py** | `get_embeddings_for_column(df, column, model_type, model_name)`: add embedding column. |
| **run_pipeline.py** | Entrypoint: PDF → pipeline → embeddings for main/planned/executed → save outputs and cost. |

## Data flow (simplified)

1. **PDF** → `set_docs` → **documents** → `get_recursive_text_splitter` → **x_df** (Doc, Page).
2. **x_df** → `get_context_summary` (with LLM) → **doc_df** (Doc, Page, summary).
3. **doc_df** → `get_subchunk` (with LLM) → **main_df** (summary_x, summary_y, positive, negative, tag, …).
4. Rename → **main_df** (childChunk, parentChunk, …).
5. **main_df** → `process_plan` / `process_exec` → **planned_actions** / **executed_actions**.
6. **main_df**, **planned_df**, **executed_df** → `get_embeddings_for_column` → same DataFrames with `*_embeddings` columns.
7. Save DataFrames + cost summary to **output/**.
