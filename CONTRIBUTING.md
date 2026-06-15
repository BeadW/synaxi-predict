# Contributing to synaxi-predict

Contributions are welcome — new benchmark importers, model improvements, bug fixes, and documentation all help.

## Setup

```bash
git clone https://github.com/BeadW/synaxi-predict
cd synaxi-predict
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"
git lfs pull          # download trained model and large datasets
```

You'll also need an `ANTHROPIC_API_KEY` in `.env` (copy `.env.example`).

## Running tests

```bash
pytest harness/tests/
```

## Retraining the model

After adding or modifying run data:

```bash
python -m predictor.train
```

The trained artifact is saved to `data/models/predictor_v1.pkl` and tracked via Git LFS.

## Adding a new benchmark dataset

1. Write a script in `scripts/import_<source>.py` that fetches the benchmark and writes JSONL to `data/runs/<source>.jsonl`.

   Required fields per record:
   ```json
   {
     "task_id":           "source/instance_id",
     "strategy":          "model-id",
     "mode":              "multi-turn",
     "task_text":         "description of the task",
     "prompt_tokens_raw": 45000,
     "completion_tokens": 8200,
     "num_turns":         32,
     "total_cost_usd":    0.84,
     "cost_is_estimated": true,
     "passed_criteria":   true
   }
   ```

   `prompt_tokens_raw` = context size at the final API call.  
   `completion_tokens` = total output tokens across **all** turns (not per-turn).

2. Run your importer and verify the stats look sane:
   ```bash
   python scripts/import_<source>.py
   python -m predictor.train
   ```

3. If your benchmark has known repos, add tree-sitter code features. See `scripts/extract_code_features.py` for the SWE-bench approach and `scripts/extract_swe_smith_features.py` for the multi-repo approach.

## Pull request process

1. Fork the repo and create a branch: `git checkout -b feat/my-improvement`
2. Make your changes and add or update tests where relevant
3. Run `pytest` and ensure it passes
4. Open a PR against `main` — describe what dataset or improvement you're adding, and include before/after model metrics if you retrained

## Code style

- Python 3.9+, type hints throughout
- `ruff check .` for linting (`pip install ruff`)
- No comments explaining what code does — only comments explaining non-obvious *why*

## Reporting issues

Open a GitHub issue at https://github.com/BeadW/synaxi-predict/issues. Include:
- What you were trying to do
- What you expected
- What actually happened (error message, prediction output, etc.)
