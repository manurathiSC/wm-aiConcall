# Using a HuggingFace model in the pipeline

Follow these steps to run the earnings-call pipeline with a model stored on HuggingFace (e.g. Qwen).

## Option A: HuggingFace Inference API (cloud)

1. **Create a HuggingFace account** at https://huggingface.co/ and log in.

2. **Create an access token** at https://huggingface.co/settings/tokens (read access is enough for inference).

3. **Add to `.env`:**
   ```env
   HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   DEFAULT_HF_MODEL=Qwen/Qwen2.5-7B-Instruct
   ```

4. **Install the HuggingFace LangChain integration:**
   ```bash
   pip install langchain-huggingface
   ```

5. **Run the pipeline with HuggingFace:**
   ```bash
   python run_pipeline.py Concalls/DLF_Jan26.pdf -o output_hf --provider huggingface
   ```
   Or specify the model on the command line:
   ```bash
   python run_pipeline.py Concalls/DLF_Jan26.pdf -o output_hf --provider huggingface --model Qwen/Qwen2.5-7B-Instruct
   ```

6. **Outputs** are written to `output_hf/` (or whatever you pass to `-o`): `Main_*.pkl`, `Plan_*.pkl`, `Exec_*.pkl`, `cost_*.json`.

---

## Option B: Local model (transformers + torch)

Use this when you want to run the model on your machine (no API token needed).

1. **Install dependencies:**
   ```bash
   pip install transformers torch langchain-community
   ```

2. **No token required** for local runs. You can leave `HUGGINGFACEHUB_API_TOKEN` unset.

3. **Run with HuggingFace provider** and the model name (same as on the Hub):
   ```bash
   python run_pipeline.py Concalls/DLF_Jan26.pdf -o output_hf_local --provider huggingface --model Qwen/Qwen2.5-7B-Instruct
   ```

4. The first run will **download the model** from HuggingFace (can be large). Later runs use the cached copy.

5. **Optional:** Set `DEFAULT_HF_MODEL=Qwen/Qwen2.5-7B-Instruct` in `.env` so you can omit `--model`.

---

## Changing the model

- **CLI:** `--model org/model-name` (e.g. `--model Qwen/Qwen2.5-1.5B-Instruct` for a smaller model).
- **Env:** `DEFAULT_HF_MODEL=org/model-name` in `.env`.
- **Code:** `get_llm(provider="huggingface", model_name="org/model-name")`.

Same prompts and workflow are used for both OpenAI (gateway) and HuggingFace; only the backing model changes.

---

## Troubleshooting

- **`ModuleNotFoundError: langchain_core.tracers.langchain_v1`**  
  Your `langchain-core` and `langchain-community` versions may be incompatible. Try:
  ```bash
  pip install --upgrade langchain-core langchain-community
  ```
  or use a fresh venv with `pip install -r requirements.txt` and then `pip install langchain-huggingface`.

- **Embeddings 404 with gateway**  
  If the AI gateway does not expose an embeddings endpoint, use local embeddings instead:
  ```bash
  python run_pipeline.py Concalls/DLF_Jan26.pdf -o output --embed-type sentence_transformers --embed-model sentence-transformers/all-MiniLM-L6-v2
  ```
  (Requires `pip install sentence-transformers`.)
