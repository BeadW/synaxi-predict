You are running the synaxi-contribute command. Follow these steps in order.

---

## Step 1: Load uncontributed records

```bash
python3 -m predictor.contribute --list 2>/dev/null
```

Parse the JSON array. Records are pre-filtered by `contribute --list` to only
include those with all mandatory fields: `actual_cost`, `actual_turns`,
`passed` (non-null), and `code_features.has_code_features == 1`.

If the array is empty, tell the user:
```
No records ready to contribute yet. Records need actual_cost, actual_turns,
a pass/fail result, and tree-sitter code features before they can be submitted.
```
Then stop.

Store the array as **RECORDS**.

---

## Step 2: Show payload and confirm

Build a markdown table for the preview:

```
| ID       | Model        | Cost   | Turns | Passed |
|----------|--------------|--------|-------|--------|
| c12c11e9 | single-haiku | $0.334 |     6 | true   |
```

Use `AskUserQuestion` (single-select) with this table as the **preview**:

**Question** — "N record(s) ready to contribute. Review what will be posted, then confirm:"
- label: `Contribute` — description: `Post N records` — **preview**: the table above
- label: `Cancel` — description: `Don't send anything`

If the user picks **Cancel**, stop.

---

## Step 3: Contribute

The table above is display-only. `bin/contribute --all` posts the full JSON
for all eligible records to GitHub.

```bash
bin/contribute --all
```

Show the full output to the user.

---

## Step 4: Report

Tell the user how many records were contributed and remind them that actuals
with tree-sitter code features contribute the most signal for retraining.
