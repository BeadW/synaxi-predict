You are running the synaxi-contribute command. Follow these steps in order.

---

## Step 1: Load uncontributed records

```bash
python3 -m predictor.contribute --list 2>/dev/null
```

Parse the JSON array. If it is empty (`[]`), tell the user "No uncontributed records found." and stop.

Store the array as **RECORDS**.

---

## Step 2: Show payload and confirm

Build the full JSON payload that will be posted as a GitHub issue:

```json
{
  "records": [
    {
      "prediction_id": "...",
      "model": "...",
      "task_text": "...",
      "actual_cost": ...,
      "actual_turns": ...,
      "passed": ...,
      "code_features": { ... }
    },
    ...
  ]
}
```

Use `AskUserQuestion` (single-select) with the payload as the **preview** on the contribute option:

**Question** — "Found N uncontributed record(s). This is the exact data that will be posted as a GitHub issue:"
- label: `Contribute all` — description: `Post all N records shown` — **preview**: the full JSON payload above
- label: `Cancel` — description: `Don't send anything`

If the user picks **Cancel**, stop.

---

## Step 3: Contribute

```bash
bin/contribute --all
```

Show the full output to the user.

---

## Step 4: Report

Tell the user how many records were contributed and remind them that actuals
with tree-sitter code features contribute the most signal for retraining.
