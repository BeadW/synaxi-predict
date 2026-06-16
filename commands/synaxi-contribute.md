You are running the synaxi-contribute command. Follow these steps in order.

---

## Step 1: Load uncontributed records

```bash
python3 -m predictor.contribute --list 2>/dev/null
```

Parse the JSON array. If it is empty (`[]`), tell the user "No uncontributed records found." and stop.

Store the array as **RECORDS**. Each record has:
- `prediction_id` — 8-char hex
- `model` — e.g. `single-haiku`
- `actual_cost` — USD
- `actual_turns` — integer
- `passed` — true/false/null
- `task_text` — task description
- `code_features` — tree-sitter snapshot (non-empty means code features were captured)

---

## Step 2: Ask what to contribute

Use `AskUserQuestion` with:

**Question 1** — "Which records do you want to contribute?"
- label: `Contribute all` — description: `Share all N uncontributed records`
- label: `Select individual` — description: `Pick specific records to share`
- label: `Cancel` — description: `Don't contribute anything`

If the user picks **Cancel**, stop.

---

## Step 3a: If "Contribute all"

Store **SELECTED_IDS** = all prediction_ids from RECORDS.

Skip to Step 4.

---

## Step 3b: If "Select individual"

Use a second `AskUserQuestion` with `multiSelect: true`:

**Question** — "Which records do you want to contribute?"

One option per record:
- **label**: `<prediction_id>` (8 chars)
- **description**: `<model> · $<actual_cost> · <actual_turns> turns · passed:<passed> · <first 60 chars of task_text>`

Store **SELECTED_IDS** = the prediction_ids of the chosen options.

If none selected, tell the user "Nothing selected." and stop.

---

## Step 4: Confirm code features

For each record in SELECTED_IDS, check whether its `code_features` dict has `has_code_features: 1`.

If any records are missing code features (empty dict or `has_code_features` != 1), warn the user:

```
⚠️  N record(s) have no tree-sitter code features — they were recorded before
    tree-sitter was installed and will contribute less signal to the model.
    Contribute anyway? (continuing will include them)
```

Use `AskUserQuestion`:
- label: `Contribute anyway` — description: `Include records without code features`
- label: `Skip those` — description: `Only contribute records that have code features`
- label: `Cancel` — description: `Don't contribute anything`

If **Cancel**: stop.
If **Skip those**: filter SELECTED_IDS to only records where `has_code_features == 1`.
If **Contribute anyway**: continue with all SELECTED_IDS.

If all selected records have code features, skip this step entirely.

---

## Step 5: Contribute

```bash
bin/contribute --ids SELECTED_IDS_SPACE_SEPARATED
```

Show the full output to the user.

---

## Step 6: Report

Tell the user how many records were contributed and remind them that more actuals
improve the model's calibration for real Claude Code tasks vs benchmark data.
