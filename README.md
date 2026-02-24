# Earnings Call Processing Pipeline

Process earnings call PDFs: extract parent/child chunks, sentiment-tagged themes (positive/negative/planned/executed), up to 10 analyst themes with bullet points, and an overall narrative summary. Works with **OpenAI** or **HuggingFace** (e.g. Qwen) via a small LLM abstraction. All LLM costs are tracked per step and written to a JSON file.

## Setup

1. **Clone and install**
   ```bash
   pip install -r requirements.txt
   ```

2. **Secrets** — create a `.env` file:
   ```
   OPENAI_API_KEY=your_gateway_key
   OPENAI_API_BASE=https://ai-gateway.smallcase.com/   # LLM calls go through this gateway
   OPENAI_DIRECT_API_KEY=your_direct_openai_key        # Used for embeddings (gateway does not support embeddings endpoint)
   ```
   Optionally set `HUGGINGFACEHUB_API_TOKEN` for HuggingFace/Qwen.

## Usage

**Run on a PDF (default: `Concalls/DLF_Jan26.pdf`):**
```bash
python run_pipeline.py
# Or:
python run_pipeline.py path/to/your.pdf -o output --provider openai --model gpt-4o-mini
```

**Options:**
- `pdf` – Path to PDF (default: `Concalls/DLF_Jan26.pdf`)
- `-o, --output-dir` – Output directory (default: `output`)
- `--provider` – `openai` (gateway) or `huggingface`
- `--model` – Model ID sent to the gateway/API (e.g. `gpt-4o-mini`). Also set `DEFAULT_OPENAI_MODEL` in `.env` to change the default.
- `--temp` – LLM temperature (default: `0.0`)
- `--embed-type` – Embedding model type: `openai`, `huggingface`, `bge`, `instructor`
- `--embed-model` – Embedding model name (default: `text-embedding-3-large`)
- `--embeddings` – *(optional)* Generate embeddings for `parentChunk` and `childChunk` columns in `Main_*.json`

**Theme extraction and overall summary always run** — no flag needed.

**Outputs (in `output/`):**
- `Main_<run_name>.json` / `.xlsx` – Main DataFrame: parentChunk, childChunk, sentiment scores, tag, keywords
- `Plan_<run_name>.json` – Planned/forward-looking actions
- `Exec_<run_name>.json` – Executed/completed actions
- `Pos_<run_name>.json` – Positive themes and supporting details
- `Neg_<run_name>.json` – Negative themes and supporting details
- `Clusters_<run_name>.json` – Up to 10 analyst themes, each with 4–5 concrete bullet points
- `OverallSummary_<run_name>.txt` – 150–180 word analyst narrative paragraph (headline + anchor figures + management tone + watchpoint)
- `cost_<run_name>.json` – Token usage and cost per step (`context_summary`, `subchunk`, `process_neg`, `process_pos`, `process_plan`, `process_exec`, `extract_themes`, `overall_summary`) plus run total

## Project layout

- `config.py` – Loads `.env`; exposes `OPENAI_API_KEY`, `OPENAI_DIRECT_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, and default model names.
- `llm_providers.py` – LLM abstraction: `get_llm(provider, model_name)` returns an adapter; same prompts/workflow for OpenAI or HuggingFace.
- `cost_tracker.py` – `CostTracker`: records every LLM call (tokens, cost) keyed by step name; `get_summary_for_output()` produces the cost JSON.
- `prompts.py` – `PromptCollections`: all prompt strings — parent chunks, subchunks, neg/pos/executed/planned themes, `get_all_themes` (up to 10 analyst themes), `get_overall_summary` (narrative paragraph).
- `document_loader.py` – `set_docs()`, `get_recursive_text_splitter()` for PDF/folder/URL.
- `process_concall.py` – `ProcessConcall`: context summary → subchunks → neg/pos/exec/plan themes → `extract_all_themes` (prompt-based, single LLM call) → `extract_overall_summary`.
- `embeddings_module.py` – `get_embeddings_for_column(df, column_name, model_type, model_name)`. Uses `OPENAI_DIRECT_API_KEY` with `base_url="https://api.openai.com/v1"` to bypass the gateway for embedding calls.
- `run_pipeline.py` – Entrypoint: load PDF → full pipeline → theme extraction → overall summary → (optional) embeddings → save all outputs and cost JSON.

See `WORKFLOW.md` for a full flowchart of files and data flow.

## Embeddings

Embeddings are **optional** — pass `--embeddings` to generate them. Uses `OPENAI_DIRECT_API_KEY` (not the gateway key) since the Smallcase gateway does not support the embeddings endpoint.

Use `get_embeddings_for_column(df, column_name, model_type=..., model_name=...)` to add `{column_name}_embeddings` to any DataFrame. Supported `model_type`: `openai`, `huggingface` / `sentence_transformers`, `bge`, `instructor`.

With `--embeddings`, the pipeline adds:
- **mainDf**: `parentChunk_embeddings` and `childChunk_embeddings`

## Using the gateway and changing the model

- **Gateway:** LLM/chat calls use `OPENAI_API_BASE` from `.env` (default: `https://ai-gateway.smallcase.com/`).
- **Embeddings:** Bypass the gateway with `OPENAI_DIRECT_API_KEY` pointing directly to `https://api.openai.com/v1`.
- **Changing the model:** Use `--model your-model-id` on the CLI, or set `DEFAULT_OPENAI_MODEL=your-model-id` in `.env`.

## Using a HuggingFace model

1. **Install optional deps:**
   `pip install langchain-huggingface` (HF Inference API) or `pip install transformers torch langchain-community` (local models).

2. **Set token in `.env`:**
   `HUGGINGFACEHUB_API_TOKEN=hf_your_token_here`

3. **Run with HuggingFace:**
   ```bash
   python run_pipeline.py Concalls/DLF_Jan26.pdf -o output_hf --provider huggingface --model Qwen/Qwen2.5-7B-Instruct
   ```
   Or set `DEFAULT_HF_MODEL=Qwen/Qwen2.5-7B-Instruct` in `.env` and use `--provider huggingface` without `--model`.

4. **Local (no API):** For local models (e.g. `Qwen/Qwen2.5-7B-Instruct`), install `transformers` and `torch`; the code uses `HuggingFacePipeline`. No `HUGGINGFACEHUB_API_TOKEN` needed.

See `HUGGINGFACE_USAGE.md` for a step-by-step checklist.
