You are running the synaxi-predict skill. Follow these phases in order.

---

## Phase 1: Predict

Capture the current repo path, then run:

```bash
REPO_PATH="$(pwd)"
PREDICT="$(find ~ -maxdepth 4 -name "predict" -path "*/synaxi-predict/bin/*" 2>/dev/null | head -1)"
[ -z "$PREDICT" ] && { echo "ERROR: synaxi-predict not found. Clone to ~/synaxi-predict and run pip install -e ."; exit 1; }
"$PREDICT" "$ARGUMENTS" --models single --repo-path "$REPO_PATH"
```

Show the full output. Store internally:
- **PRED_ID**: 8-char hex on the `Prediction ID:` line
- **PREDICT_BIN**: the path found for `predict`
- **REPO_PATH**: the captured working directory

---

## Phase 2: Model selection

Use `AskUserQuestion` with one option per model row from the table:
- **label**: model name  
- **description**: `Est. $X.XX · N turns · X% pass`

Include a final option: label `"Just predict — don't execute"`, description `"Stop here"`.

If the user picks "Just predict", stop. Otherwise store **CHOSEN_MODEL** (`haiku` or `sonnet`) and continue.

---

## Phase 3: Execute with a subagent

Dispatch the task to a subagent using the model the user chose:

```
Agent(
  description="Execute: $ARGUMENTS",
  model=CHOSEN_MODEL,
  prompt="""
You are working in the repo at REPO_PATH.
Task: $ARGUMENTS

Complete the task using all available tools. When done, summarise what you changed and whether it succeeded.
"""
)
```

After the agent returns, extract the **AGENT_ID** from its result — it appears as `agentId: <hex-string>`.

---

## Phase 4: Parse exact metrics from the session

Run parse-session using the agent ID and the repo path from Phase 1:

```bash
PARSE_SESSION="$(dirname "$PREDICT_BIN")/parse-session"
"$PARSE_SESSION" AGENT_ID REPO_PATH
```

This returns JSON. Extract and store:
- **ACTUAL_TURNS** from `turns`
- **ACTUAL_COST** from `cost_usd`

---

## Phase 5: Eval — did the task succeed?

Dispatch an eval agent:

```
Agent(
  description="Eval: did the task succeed?",
  prompt="""
Task that was attempted: $ARGUMENTS
Repo: REPO_PATH

Check whether the task succeeded:
1. Run git diff HEAD to see what changed
2. If tests exist (pytest, npm test, etc.), run them and capture output
3. Review the changes against the task requirements

Respond with ONLY valid JSON (no markdown, no explanation):
{"passed": true, "reason": "one sentence"}
  or
{"passed": false, "reason": "one sentence"}
"""
)
```

Parse the JSON from the eval agent's response. Extract **PASSED** (`true`/`false`).

---

## Phase 6: Record actuals

```bash
RECORD="$(dirname "$PREDICT_BIN")/record-actual"
"$RECORD" PRED_ID --turns ACTUAL_TURNS --cost ACTUAL_COST --passed PASSED
```

Then tell the user:

```
Recorded prediction PRED_ID:
  Model:   CHOSEN_MODEL
  Turns:   ACTUAL_TURNS  (predicted: X)
  Cost:    $ACTUAL_COST  (predicted: $X.XX)
  Passed:  PASSED
  Reason:  <eval agent reason>
```
