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

**Question** — "Which records do you want to contribute?"
- label: `Contribute all` — description: `Share all N uncontributed records`
- label: `Select individual` — description: `Pick specific records to share`
- label: `Cancel` — description: `Don't contribute anything`

If the user picks **Cancel**, stop.

---

## Step 3a: If "Contribute all"

Store **SELECTED_RECORDS** = all records from RECORDS.

Skip to Step 4.

---

## Step 3b: If "Select individual"

Build a preview string for each record — a JSON block showing exactly what will be sent:

```
{
  "prediction_id": "<id>",
  "model": "<model>",
  "task_text": "<full task_text>",
  "actual_cost": <actual_cost>,
  "actual_turns": <actual_turns>,
  "passed": <passed>,
  "code_features": <code_features as JSON>
}
```

Use `AskUserQuestion` with `multiSelect: true`:

**Question** — "Which records do you want to contribute?"

One option per record:
- **label**: `<prediction_id>`
- **description**: `<model> · $<actual_cost> · <actual_turns> turns · passed:<passed> · <first 60 chars of task_text>`
- **preview**: the JSON block built above for this record

Store **SELECTED_RECORDS** = the full record objects for the chosen options.

If none selected, tell the user "Nothing selected." and stop.

---

## Step 4: Confirm code features

For each record in SELECTED_RECORDS, check whether `code_features` has `has_code_features: 1`.

If any records are missing code features (empty dict or `has_code_features` != 1), use `AskUserQuestion`:

**Question** — "N record(s) have no tree-sitter code features (recorded before tree-sitter was installed). They contribute less signal to the model. What would you like to do?"
- label: `Contribute anyway` — description: `Include all selected records`
- label: `Skip those` — description: `Only contribute records that have code features`
- label: `Cancel` — description: `Don't contribute anything`

If **Cancel**: stop.
If **Skip those**: remove records where `has_code_features != 1` from SELECTED_RECORDS.
If **Contribute anyway**: continue unchanged.

If all selected records have code features, skip this step entirely.

---

## Step 5: Show exact payload and confirm

Build the full JSON payload that will be posted — this is exactly what the contribution issue will contain:

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

Use `AskUserQuestion` (single-select) with the payload as the **preview** on the confirm option:

**Question** — "Ready to contribute N record(s). This is the exact data that will be posted as a GitHub issue:"
- label: `Confirm and contribute` — description: `Post the data shown` — **preview**: the full JSON payload above
- label: `Cancel` — description: `Don't send anything`

If the user picks **Cancel**, stop.

---

## Step 6: Contribute

```bash
bin/contribute --ids SELECTED_IDS_SPACE_SEPARATED
```

Show the full output to the user.

---

## Step 7: Report

Tell the user how many records were contributed and remind them that more actuals
(especially with tree-sitter code features) improve the model's calibration for
real Claude Code tasks vs benchmark data.
