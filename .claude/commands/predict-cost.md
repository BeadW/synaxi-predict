You are running the synaxi-predict skill. This is a three-phase flow: predict → user selects model → execute task → record actuals. Follow each phase in order.

---

## Phase 1: Predict

Capture the repo path, then run the predictor:

```bash
REPO_PATH="$(pwd)" && python -m predictor.predict "$ARGUMENTS" --repo-path "$REPO_PATH"
```

If `predictor` is not on PATH, locate and activate the install:

```bash
REPO_PATH="$(pwd)" && PREDICT_DIR="$(python -c "import predictor, os; print(os.path.dirname(os.path.dirname(predictor.__file__)))" 2>/dev/null || find ~ -name "predict.py" -path "*/predictor/*" -not -path "*/.venv/*" | head -1 | xargs dirname | xargs dirname)" && cd "$PREDICT_DIR" && source .venv/bin/activate 2>/dev/null; python -m predictor.predict "$ARGUMENTS" --repo-path "$REPO_PATH"
```

Show the full output. Extract and store internally:
- **PRED_ID**: the 8-char hex on the `Prediction ID:` line
- **RECOMMENDED**: the model marked with `◀ recommended`

---

## Phase 2: Model selection — PAUSE HERE

After displaying the prediction table, output **only** this question and stop. Do not proceed to Phase 3 until the user responds.

```
Which model would you like to use for this task?

Enter the model name, a number from the table, or press Enter to use the recommendation.
(If you want a different Claude tier, start a new session: `claude --model <model-id>` then re-run this command.)
```

Wait for the user's reply. Note their chosen model as **CHOSEN_MODEL**. Then continue to Phase 3.

---

## Phase 3: Execute the task

Work on the task: **$ARGUMENTS**

Use all available tools — read files, edit code, run tests, search the codebase — to complete it. Keep a running count of every tool call you make (each Bash, Read, Edit, Write, Grep call = 1 turn).

Determine clearly whether the task succeeded (tests pass, objective verifiably achieved, or explicitly failed/unresolved).

---

## Phase 4: Record actuals

After completing the task, run the record command. Substitute the real values for PRED_ID, ACTUAL_TURNS, and true/false:

```bash
python -m predictor.record_actual PRED_ID --turns ACTUAL_TURNS --passed true
```

Then tell the user:

```
Actuals recorded (PRED_ID — ACTUAL_TURNS turns, passed: true/false).

To also log the cost (improves future predictions):
  ! python -m predictor.record_actual PRED_ID --cost <USD_amount>

Get the cost from `/cost` in Claude Code — use the delta since this task started.
```
