# Earnings Call Processing Pipeline

Process earnings call PDFs: extract parent/child chunks, themes (positive/negative/planned/executed), and embeddings. Same prompts and workflow work with **OpenAI** or **HuggingFace** (e.g. Qwen) via a small LLM abstraction. All LLM costs are tracked and written to a JSON file.

## Setup

1. **Clone and install**
   ```bash
   pip install -r requirements.txt
   ```

2. **Secrets**
   - use `.env`
   - Set `OPENAI_API_KEY` (your key; works with the default gateway).
   - Default gateway: `OPENAI_API_BASE=https://ai-gateway.smallcase.com/`. Optionally set `HUGGINGFACEHUB_API_TOKEN` for HuggingFace/Qwen.

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
- `--model` – **Model parameter** sent to the gateway/API (e.g. `gpt-4o-mini`). You can change this to any model ID your gateway supports; also set `DEFAULT_OPENAI_MODEL` in `.env` to change the default.
- `--temp` – LLM temperature
- `--embed-type` – Embedding model type: `openai`, `huggingface`, `bge`, `instructor`
- `--embed-model` – Embedding model name

**Outputs (in `output/`):**
- `Main_<run_name>.json` / `.xlsx` – Main DataFrame with parent/child chunks and embeddings
- `Plan_<run_name>.json` – Planned actions and embeddings
- `Exec_<run_name>.json` – Executed actions and embeddings
-  `Pos_<run_name>.json` – Positive themes, details, and embeddings
-  `Neg_<run_name>.json` – Negative themes, details, and embeddings
-  `cost_<run_name>.json` – Cost and token usage per step and total

## Project layout

- `config.py` – Loads `.env` and exposes `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, and default model names.
- `llm_providers.py` – LLM abstraction: `get_llm(provider, model_name)` returns an adapter; same prompts/workflow for OpenAI or HuggingFace (Qwen).
- `cost_tracker.py` – `CostTracker`: records each LLM call (tokens, cost), `print_summary()`, `get_summary_for_output()`.
- `prompts.py` – `PromptCollections`: parent chunks, subchunks, neg/pos/executed/planned theme prompts.
- `document_loader.py` – `set_docs()`, `get_recursive_text_splitter()` for PDF/folder/URL.
- `process_concall.py` – `ProcessConcall`: context summary → subchunks → theme extraction; uses LLM adapter and cost tracker.
- `embeddings_module.py` – `get_embeddings_for_column(df, column_name, model_type=..., model_name=...)` for any column and model.
- `run_pipeline.py` – Entrypoint: load PDF, run pipeline, add embeddings for mainDf (parentChunks, childChunks), plannedDf (planned_actions), executedDf (executed_actions), save outputs and cost.

See `WORKFLOW.md` for a flowchart of files and data flow.

## Embeddings

Use `get_embeddings_for_column(df, column_name, model_type=..., model_name=...)` to add `{column_name}_embeddings` to a DataFrame. Supported `model_type`: `openai`, `huggingface` / `sentence_transformers`, `bge`, `instructor`.

The pipeline already adds:
- **mainDf**: `parentChunks` and `childChunks` embeddings
- **plannedDf**: `planned_actions` embeddings
- **executedDf**: `executed_actions` embeddings

## Using the gateway and changing the model

- **Gateway:** All OpenAI-style calls (chat + embeddings) use `OPENAI_API_BASE` from `.env` (default: `https://ai-gateway.smallcase.com/`). Your `OPENAI_API_KEY` in `.env` is sent with each request.
- **Changing the model:** The `model` parameter sent to the gateway is:
  - From the CLI: `python run_pipeline.py --model your-model-id`
  - From `.env`: set `DEFAULT_OPENAI_MODEL=your-model-id`
  - In code: `get_llm(provider="openai", model_name="your-model-id")` or `ProcessConcall(..., model_name="...")`.

## Using a HuggingFace model

1. **Install optional deps:**  
   `pip install langchain-huggingface` (for HF Inference API) or `pip install transformers torch langchain-community` (for local models like Qwen).

2. **Set token in `.env`:**  
   `HUGGINGFACEHUB_API_TOKEN=hf_your_token_here`  
   (Create a token at https://huggingface.co/settings/tokens.)

3. **Run with HuggingFace:**  
   ```bash
   python run_pipeline.py Concalls/DLF_Jan26.pdf -o output_hf --provider huggingface --model Qwen/Qwen2.5-7B-Instruct
   ```  
   Or set `DEFAULT_HF_MODEL=Qwen/Qwen2.5-7B-Instruct` in `.env` and use `--provider huggingface` without `--model`.

4. **Local (no API):** For a model that runs locally (e.g. `Qwen/Qwen2.5-7B-Instruct`), install `transformers` and `torch`; the code will use `HuggingFacePipeline` if `langchain-huggingface` is not installed. No `HUGGINGFACEHUB_API_TOKEN` needed for local runs.

See `HUGGINGFACE_USAGE.md` for a step-by-step checklist.



