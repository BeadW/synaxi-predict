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

Build a markdown table for the preview:

```
| ID       | Model         | Cost   | Turns | Passed | Code features |
|----------|---------------|--------|-------|--------|---------------|
| f410a765 | deepseek-chat | $0.008 |    20 | true   | no            |
| 907a58ea | single-sonnet | $0.450 |    15 | null   | no            |
| 9d3e313b | single-haiku  | $0.314 |     4 | true   | no            |
| c12c11e9 | single-haiku  | $0.334 |     6 | true   | yes           |
```

Use `AskUserQuestion` (single-select) with this table as the **preview** on the contribute option:

**Question** — "Found N uncontributed record(s). Review what will be posted, then confirm:"
- label: `Contribute all` — description: `Post all N records` — **preview**: the table above
- label: `Cancel` — description: `Don't send anything`

If the user picks **Cancel**, stop.

---

## Step 3: Contribute

The table above is display-only. The full JSON payload is what actually gets
posted to GitHub — `bin/contribute --all` handles that automatically.

```bash
bin/contribute --all
```

Show the full output to the user.

---

## Step 4: Report

Tell the user how many records were contributed and remind them that actuals
with tree-sitter code features contribute the most signal for retraining.
