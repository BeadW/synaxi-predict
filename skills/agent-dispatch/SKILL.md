---
description: >
  Wraps subagent dispatch with cost prediction, model selection, and automatic
  actuals recording. Use this INSTEAD OF calling the Agent tool directly whenever
  delegating a coding task to a subagent. Pass the full task description as args.
when_to_use: >
  Call Skill("agent-dispatch", args="<task description>") any time you would
  use the Agent tool. Covers: implementing features, fixing bugs, writing tests,
  refactoring, research, or any task you would delegate to a subagent.
user-invocable: false
allowed-tools: Bash AskUserQuestion Agent
---

## Cost prediction

```!
python3 -m predictor.predict "$ARGUMENTS" --models single --repo-path "$(pwd)" 2>/dev/null \
  || echo "WARNING: synaxi-predict not installed — restart Claude Code to trigger setup"
```

Parse the block above. Store internally:
- **PRED_ID** — 8-char hex from the `Prediction ID:` line (`none` if prediction failed)
- **PREDICT_BIN** — path to the `predict` binary used (derive siblings from `dirname`)
- **REPO_PATH** — `$(pwd)` used in the prediction command above

Write prediction state so the recording step can find it later:

```bash
mkdir -p /tmp/synaxi
printf '{"pred_id":"%s","repo_path":"%s","predict_bin":"%s"}\n' \
  "PRED_ID" "REPO_PATH" "PREDICT_BIN" \
  > /tmp/synaxi_${CLAUDE_SESSION_ID}.json
```

---

## Model selection

If PRED_ID is `none`, skip to Execute using model `haiku`.

Otherwise use `AskUserQuestion` with one option per model row in the prediction table:
- **label**: model name (e.g. `single-haiku`)
- **description**: `Est. $X.XX · N turns · X% pass`

Include a final option: label `"Just predict — don't execute"`, description `"Stop here"`.

If the user picks "Just predict", stop.

---

## Execute

Dispatch the task agent with the chosen model:

```
Agent(
  model = "haiku" or "sonnet" (from selection above),
  prompt = "Working in REPO_PATH.\n\nTask: $ARGUMENTS\n\nComplete the task using all available tools. Summarise what you changed and whether it succeeded."
)
```

After Agent returns, extract **AGENT_ID** from `agentId: <hex>` in the result metadata.

---

## Parse metrics

```bash
PARSE="$(dirname "PREDICT_BIN")/parse-session"
"$PARSE" AGENT_ID REPO_PATH
```

Extract from the JSON output:
- **ACTUAL_TURNS** from `turns`
- **ACTUAL_COST** from `cost_usd`

---

## Eval

Dispatch a second Agent to assess success independently:

```
Agent(
  prompt = "Task attempted: $ARGUMENTS\nRepo: REPO_PATH\n\n1. Run: git diff HEAD\n2. Run tests if any exist (pytest -q, npm test, etc.)\n3. Assess whether the task requirements are met.\n\nRespond with ONLY valid JSON, no markdown:\n{\"passed\": true, \"reason\": \"one sentence\"}"
)
```

Parse **PASSED** and **REASON** from the JSON.

---

## Record

```bash
RECORD="$(dirname "PREDICT_BIN")/record-actual"
"$RECORD" PRED_ID --turns ACTUAL_TURNS --cost ACTUAL_COST --passed PASSED
```

Report to the user:

```
Recorded PRED_ID:
  Turns:  ACTUAL_TURNS  (predicted: X)
  Cost:   $ACTUAL_COST  (predicted: $X.XX)
  Passed: PASSED — REASON
```
