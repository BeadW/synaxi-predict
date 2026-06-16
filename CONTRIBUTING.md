# Contributing to synaxi-predict

Contributions are welcome — new benchmark importers, actuals data, model improvements, bug fixes, and documentation all help.

## Setup

```bash
git clone https://github.com/BeadW/synaxi-predict
cd synaxi-predict
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"
git lfs pull          # download trained model and large datasets
cp .env.example .env  # add ANTHROPIC_API_KEY if running benchmarks
```

## Running tests

```bash
# Smoke-test the predictor (no API key needed)
python -m predictor.predict "Fix the failing test" --models single --repo-path .

# Lint
ruff check predictor/ scripts/
```

A full test suite is planned. For now CI runs the smoke test and lint on every PR.

## Skills

The repo ships as a Claude Code plugin. Install once inside a Claude Code session:

```
/plugin marketplace add BeadW/synaxi-predict
/plugin install synaxi-predict
```

On the next session start, `scripts/plugin-setup.sh` runs automatically via the `SessionStart` hook. It installs the Python package from `$CLAUDE_PLUGIN_ROOT` and downloads `predictor_v1.pkl` from GitHub Releases into the platform data directory — no manual steps needed.

**`/synaxi-predict`** (`commands/synaxi-predict.md`) — slash command, user-invoked. Shows prediction table, asks for model choice, dispatches subagent, records actuals.

**`synaxi-predict`** skill (`skills/synaxi-predict/SKILL.md`) — `user-invocable: false`, Claude-invoked. Fires automatically whenever Claude would spawn a subagent. Uses dynamic `!` injection to run the prediction at skill load time (tree-sitter features included), then follows the same predict → dispatch → parse → eval → record flow.

**Releasing a new version:** bump `version` in `.claude-plugin/plugin.json`, upload the new model artifact to GitHub Releases, and tag. The hook re-runs on next session start for all users.

## Contributing actuals data

If you've accumulated real task runs, share them to improve the model:

```bash
bin/contribute      # shows table, prompts all/pick/none
bin/contribute --all  # share everything without prompting
```

This files a GitHub issue on the repo with your records. We validate each submission by re-running the predictor against the submitted task text — if `pred_cost`/`pred_turns` don't match within 20%, the record is rejected. A stable anonymous `contributor_id` is generated on first run and stored in `data/contributor_id` (gitignored).

## How actuals are recorded

Every task dispatched via either skill produces:
1. A prediction in `data/predictions_live.jsonl`
2. An actual in `data/actuals_live.jsonl` (written by `bin/record-actual`)

`bin/parse-session <agentId>` extracts exact turn count and cost from the subagent's Claude Code session JSONL — no self-reporting. The agentId comes from the Agent tool result metadata (`agentId: <hex>`).

## Retraining the model

After accumulating new actuals or adding benchmark data:

```bash
python -m predictor.train
```

The artifact is saved to `data/models/predictor_v1.pkl` (Git LFS) and stamped with a `training_date`.

## Adding a new benchmark dataset

1. Write `scripts/import_<source>.py`. It should fetch the benchmark and write JSONL to `data/runs/<source>.jsonl` with these fields:

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

   `prompt_tokens_raw` = context at the final API call.  
   `completion_tokens` = total output tokens across **all** turns (not per-turn).

2. Verify the import and retrain:
   ```bash
   python scripts/import_<source>.py
   python -m predictor.train
   ```

3. If your benchmark has known repos, add tree-sitter features:
   - See `scripts/extract_code_features.py` for the SWE-bench approach (single repo per instance)
   - See `scripts/extract_swe_smith_features.py` for the multi-repo approach (clones 129 repos)

4. Include before/after eval metrics in your PR:
   ```bash
   python scripts/eval_holdout.py
   python scripts/eval_pass_rate.py
   ```

## Pull request process

1. Fork and branch: `git checkout -b feat/my-improvement`
2. Make changes; run `ruff check predictor/ scripts/` and fix any issues
3. If retraining: include before/after metrics from `eval_holdout.py` and `eval_pass_rate.py`
4. Open a PR against `main` — use the PR template

## Code style

- Python 3.9+, type hints throughout
- `ruff check .` for linting
- No comments explaining *what* code does — only non-obvious *why*
- No docstrings beyond a single summary line where needed

## Reporting issues

Open a GitHub issue at https://github.com/BeadW/synaxi-predict/issues. Include what you were trying to do, what you expected, and what actually happened (error message, prediction output, etc.).
