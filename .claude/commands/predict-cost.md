You are running the synaxi-predict skill: predict cost/turns/pass for a task, let the user pick a model, execute, then record actuals. Follow the phases in order.

---

## Phase 1: Predict

Find the synaxi-predict wrappers and run the predictor:

```bash
REPO_PATH="$(pwd)"
PREDICT="$(find ~ -maxdepth 4 -name "predict" -path "*/synaxi-predict/bin/*" 2>/dev/null | head -1)"
[ -z "$PREDICT" ] && { echo "ERROR: synaxi-predict not found. Clone it to ~/synaxi-predict and run pip install -e ."; exit 1; }
"$PREDICT" "$ARGUMENTS" --models single --repo-path "$REPO_PATH"
```

Show the full output. Extract and store internally:
- **PRED_ID**: the 8-char hex on the `Prediction ID:` line
- **PREDICT_BIN**: the path to the `predict` wrapper found above (needed for Phase 4's sibling `record-actual`)
- The list of model rows with their estimated costs and pass rates

---

## Phase 2: Model selection

Use the `AskUserQuestion` tool to present the available models as selectable options. Build one option per model row from the prediction table, using this format:
- **label**: the model name (e.g. `single-haiku`, `single-sonnet`)
- **description**: `Est. $X.XX · N turns · X% pass`

Always include a final option: label `"Just predict — don't execute"`, description `"Stop here without running the task"`.

Wait for the user's selection. If they choose "Just predict", stop here. Otherwise note **CHOSEN_MODEL** and continue to Phase 3.

---

## Phase 3: Execute the task

Work on the task: **$ARGUMENTS**

Use all available tools — read files, edit code, run tests — to complete it. Count every tool call (Bash, Read, Edit, Write, Grep) as one turn.

Determine clearly whether the task succeeded (tests pass or objective verifiably achieved).

---

## Phase 4: Record actuals

Use the sibling `record-actual` wrapper next to the `predict` binary found in Phase 1:

```bash
RECORD="$(dirname "$PREDICT_BIN")/record-actual"
"$RECORD" PRED_ID --turns ACTUAL_TURNS --passed true
```

Then tell the user:

```
Actuals recorded (PRED_ID — ACTUAL_TURNS turns, passed: true/false).

To log cost too:  ! <path-to-record-actual> PRED_ID --cost <USD>
Find it with `/cost` in Claude Code — use the delta since this task started.
```
