You are running the synaxi-predict skill: predict cost/turns/pass for a task, let the user pick a model, execute, then record actuals. Follow the phases in order.

---

## Phase 1: Predict

Capture the repo path, then run the predictor filtered to Claude Code strategies:

```bash
REPO_PATH="$(pwd)" && python -m predictor.predict "$ARGUMENTS" --models single --repo-path "$REPO_PATH"
```

If `predictor` is not on PATH, locate and activate the install first:

```bash
REPO_PATH="$(pwd)" && PREDICT_DIR="$(python3 -c "import predictor, os; print(os.path.dirname(os.path.dirname(predictor.__file__)))" 2>/dev/null || find ~ -name "predict.py" -path "*/predictor/*" -not -path "*/.venv/*" | head -1 | xargs dirname | xargs dirname)" && cd "$PREDICT_DIR" && source .venv/bin/activate 2>/dev/null && python -m predictor.predict "$ARGUMENTS" --models single --repo-path "$REPO_PATH"
```

Show the full output. Extract and store internally:
- **PRED_ID**: the 8-char hex on the `Prediction ID:` line
- The list of model rows with their estimated costs and pass rates

---

## Phase 2: Model selection

Use the `AskUserQuestion` tool to present the available models as selectable options. Build one option per model row from the prediction table, using this format:
- **label**: the model name (e.g. `single-haiku`, `single-sonnet`)
- **description**: `Est. $X.XX · N turns · X% pass`

Always include a final option: label `"Just predict — don't execute"`, description `"Stop here without running the task"`.

Wait for the user's selection before proceeding. If they choose "Just predict", stop here. Otherwise note **CHOSEN_MODEL** and continue to Phase 3.

---

## Phase 3: Execute the task

Work on the task: **$ARGUMENTS**

Use all available tools — read files, edit code, run tests — to complete it. Count every tool call (Bash, Read, Edit, Write, Grep) as one turn.

Determine clearly whether the task succeeded (tests pass or objective verifiably achieved).

---

## Phase 4: Record actuals

Run (substituting real values):

```bash
python -m predictor.record_actual PRED_ID --turns ACTUAL_TURNS --passed true
```

Then tell the user:

```
Actuals recorded (PRED_ID — ACTUAL_TURNS turns, passed: true/false).

To log cost too:  ! python -m predictor.record_actual PRED_ID --cost <USD>
Find it with `/cost` in Claude Code — use the delta since this task started.
```
