# synaxi-predict

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![CI](https://github.com/BeadW/synaxi-predict/actions/workflows/ci.yml/badge.svg)](https://github.com/BeadW/synaxi-predict/actions)

Predicts the cost, turn count, and pass rate of a Claude Code task before it runs — so you can pick the right model without wasting tokens on a bad fit. Closes the loop by capturing actual results and feeding them back into the model.

## How it works

```
/predict-cost Fix the failing migration
        │
        ▼
  Model              Est. cost   Turns   Pass
  ─────────────────────────────────────────────
  single-haiku       $    0.35    28.1    8%  ◀ recommended
  single-sonnet      $    0.62    18.4   11%
        │
        ▼  [you pick a model]
        │
        ▼
  Subagent runs the task with the chosen model
        │
        ├─ bin/parse-session reads the subagent's session JSONL
        │   → exact turns, token counts, real cost (not estimated)
        │
        ├─ Eval agent checks git diff + test output → passed: true/false
        │
        └─ bin/record-actual logs prediction vs. actuals
           → feeds back into next training run
```

Predictions use an MLP trained on ~53k agent runs (SWE-bench, SWE-smith, OpenHands, loong0814, real Claude Code runs). Input features: TF-IDF on task text + tree-sitter code complexity features from the current repo.

## Install

Inside any Claude Code session:

```
/plugin marketplace add BeadW/synaxi-predict
/plugin install synaxi-predict
```

On the next session start, Claude Code automatically:
- Installs the Python package (`pip install -e`)
- Downloads the model artifact (~190MB) from GitHub Releases into your platform data directory (`~/Library/Application Support/synaxi-predict/` on macOS, `~/.local/share/synaxi-predict/` on Linux)

Updates happen the same way — bump `version` in `.claude-plugin/plugin.json`, release, and the hook re-runs on next session.

Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY` if you plan to run benchmarks.

### Manual install (development)

```bash
git clone https://github.com/BeadW/synaxi-predict ~/synaxi-predict
cd ~/synaxi-predict
git lfs pull        # download trained model (~190MB)
pip install -e .
```

## Usage

### Manual: `/predict-cost`

In any Claude Code session, type:

```
/predict-cost Fix the failing login migration
```

Claude runs the predictor, shows the table, and asks which model you want. After you pick, it dispatches a subagent with that model, then automatically records the actual cost and turns against the prediction.

### Automatic: `agent-dispatch` skill

Once installed, Claude invokes this skill automatically whenever it decides to spawn a subagent — no explicit command needed. The prediction table is computed at skill load time via dynamic injection (tree-sitter code features included), so there's no extra tool call overhead.

### CLI

```bash
# Predict for a task (shows all models)
bin/predict "Add OAuth login" --repo-path /path/to/project

# Predict for Claude Code models only
bin/predict "Add OAuth login" --models single --repo-path .

# List all supported models
bin/predict --list-models

# Show model training date
bin/predict --version

# Parse a subagent session for exact metrics (agentId from Agent tool result)
bin/parse-session <agentId> /path/to/project

# Record actuals manually
bin/record-actual <pred_id> --turns 18 --cost 0.42 --passed true
```

## Closed-loop recording

Every completed task produces a ground-truth record in `data/actuals_live.jsonl`:

```json
{
  "prediction_id": "c7df172d",
  "model": "single-haiku",
  "pred_cost": 0.504,  "actual_cost": 0.243,
  "pred_turns": 86.8,  "actual_turns": 8,
  "passed": true
}
```

After accumulating enough actuals, retrain:

```bash
python -m predictor.train
```

## Training data

| Source | Records | Notes |
|--------|---------|-------|
| SWE-smith | 25,826 | Synthetic SWE tasks, claude-3-7/3-5-sonnet/gpt-4o |
| loong0814 mini | 9,990 | SWE-bench Verified, 5 models |
| OpenHands SWE-bench Lite | 5,693 | 19 models, real pass/fail |
| loong0814 full | 3,639 | Real API costs from accumulated_cost |
| Claude Code runs | 725 | HumanEval/MBPP runs, upsampled ~14× |

## Model performance (held-out 20%)

| Target | R² | MAE | Within 2× |
|--------|----|-----|-----------|
| turns | 0.59 | 10.4 turns | 91% |
| completion tokens | 0.32 | 2,936 tok | 75% |
| pass rate (AUC-ROC) | 0.91 | — | 84% acc |

Turn predictions are calibrated for SWE-bench-style tasks; real Claude Code runs tend to use fewer turns than predicted. This improves as more actuals are recorded via `agent-dispatch`.

## Repo structure

```
bin/                     Executable wrappers (predict, record-actual, parse-session)
predictor/               Core prediction, training, and session-parsing logic
  predict.py             CLI entry point and cost calculation
  train.py               MLP training pipeline
  parse_session.py       Parses Claude Code session JSONL for exact metrics
  record_actual.py       Records actuals against predictions
scripts/                 Dataset importers and eval tools
  import_*.py            Normalise benchmark datasets → data/runs/
  extract_*.py           Compute tree-sitter features for benchmark repos
  eval_holdout.py        R², MAE, within-2× on 20% holdout
  eval_pass_rate.py      AUC-ROC, Brier score, calibration
features/code/           Benchmark task definitions (HumanEval, MBPP, etc.)
data/runs/               Training corpus (JSONL, one record per agent run)
data/models/             Trained model artifact (Git LFS)
data/code_features.json  Tree-sitter features per benchmark instance
.claude-plugin/          Plugin metadata (plugin.json, marketplace.json)
commands/                /predict-cost slash command
skills/                  agent-dispatch skill (auto-invoked on subagent spawn)
```

## Expanding the dataset

Each `scripts/import_*.py` pulls a public benchmark and normalises it into `data/runs/`. New importers should produce JSONL with:

```json
{
  "task_id":           "source/instance_id",
  "strategy":          "model-id",
  "task_text":         "description of the task",
  "prompt_tokens_raw": 45000,
  "completion_tokens": 8200,
  "num_turns":         32,
  "total_cost_usd":    0.84,
  "passed_criteria":   true,
  "mode":              "multi-turn"
}
```

`prompt_tokens_raw` = context size at the final API call. `completion_tokens` = total across all turns.

## Contributing data

If you've accumulated actuals from real tasks, you can share them to help improve the model:

```bash
bin/contribute      # shows uncontributed records, prompts to share via GitHub issue
bin/contribute --all  # non-interactive
```

Requires `gh` CLI to be authenticated. Each contribution is validated against the predictor before being accepted into the training set.

## Contributing code

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome — especially new benchmark importers and actuals data.

## License

[MIT](LICENSE) — use freely, attribution appreciated.
