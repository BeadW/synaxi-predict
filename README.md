# synaxi-predict

Predicts the cost, turn count, and pass rate of a coding task before you run it — so you can pick the right model without wasting tokens on a bad fit.

## What it does

Given a task description (and optionally the current repo), it outputs:

```
  Model                  Est. cost    Turns    Pass
  ──────────────────────────────────────────────────
  single-haiku           $    0.22     8.3    43%
  single-sonnet          $    0.61    26.4    71%  ◀ recommended
```

The predictor is an MLP trained on ~53k agent runs across SWE-bench, SWE-smith, OpenHands, and loong0814 benchmarks, plus real Claude Code runs. It uses TF-IDF on task text + tree-sitter code features from the current repo.

## Quick start

```bash
git clone https://github.com/bradleyjwilliams/synaxi-predict
cd synaxi-predict
pip install -e .
python -m predictor.predict "Fix the failing migration" --models single
```

Or as a Claude Code slash command — copy to your commands directory:

```bash
cp .claude/commands/predict-cost.md ~/.claude/commands/
```

Then in any Claude Code session: `/predict-cost Fix the failing migration`

## Training data

| Source | Records | Notes |
|--------|---------|-------|
| SWE-smith | 25,826 | Synthetic SWE tasks, claude-3-7/3-5-sonnet/gpt-4o |
| loong0814 mini | 9,990 | SWE-bench Verified, 5 models |
| OpenHands SWE-bench Lite | 5,693 | 19 models, real pass/fail |
| loong0814 full | 3,639 | Real API costs from accumulated_cost |
| Claude Code runs | 725 | HumanEval/MBPP runs, upsampled to ~20% of training |

## Adding your own runs

Record actuals after a task to improve future predictions:

```bash
python -m predictor.record_actual <prediction_id> --cost 0.42 --turns 18 --passed true
```

Retrain after accumulating new data:

```bash
python -m predictor.train
```

## Expanding the dataset

Each `scripts/import_*.py` pulls a public benchmark dataset and normalizes it into `data/runs/`. To add a new source, write an importer that produces JSONL with these fields:

```json
{
  "task_id": "source/instance_id",
  "strategy": "model-id",
  "task_text": "description of the task",
  "prompt_tokens_raw": 45000,
  "completion_tokens": 8200,
  "num_turns": 32,
  "total_cost_usd": 0.84,
  "passed_criteria": true,
  "mode": "multi-turn"
}
```

`prompt_tokens_raw` = context at the final API call. `completion_tokens` = total across all turns.

## Code features

Tree-sitter features are pre-computed for SWE-bench Lite/Verified and SWE-smith instances and stored in `data/code_features.json`. To regenerate (requires repo clones):

```bash
python scripts/extract_code_features.py        # SWE-bench
python scripts/extract_swe_smith_features.py   # SWE-smith (clones 129 repos)
```

## Model performance (held-out 20%)

| Target | R² | MAE | Within 2× |
|--------|----|-----|-----------|
| turns | 0.59 | 10.4 turns | 91% |
| completion tokens | 0.32 | 2,936 tok | 75% |
| pass rate (AUC-ROC) | 0.91 | — | 84% acc |

## Repo structure

```
predictor/               Core prediction + training logic
scripts/                 Dataset importers, feature extractors, eval tools
features/code/           Benchmark task definitions (HumanEval, MBPP, etc.)
data/runs/               Training corpus (JSONL, one record per agent run)
data/models/             Trained model artifact (Git LFS)
data/code_features.json  Tree-sitter features per benchmark instance
.claude/commands/        Claude Code slash command
```
