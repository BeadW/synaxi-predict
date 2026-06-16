"""
Share actual cost/turns data with the HuggingFace community to improve the predictor.

Usage:
    python -m predictor.contribute             # Interactive mode
    python -m predictor.contribute --all       # Auto-contribute all uncontributed records
    python -m predictor.contribute --help      # Show this help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
ACTUALS_FILE = PROJECT_ROOT / "data" / "actuals_live.jsonl"
PREDICTIONS_FILE = PROJECT_ROOT / "data" / "predictions_live.jsonl"
CONTRIBUTED_FILE = PROJECT_ROOT / "data" / "contributed.json"

HF_DATASET_REPO = "BeadW/synaxi-predict-actuals"
HF_DATASET_FILE = "actuals.jsonl"


def load_contributed_ids() -> set[str]:
    """Load the set of already-contributed prediction IDs."""
    if not CONTRIBUTED_FILE.exists():
        return set()
    try:
        data = json.loads(CONTRIBUTED_FILE.read_text())
        return set(data.get("contributed_ids", []))
    except Exception:
        return set()


def load_predictions() -> dict[str, dict]:
    """Load all predictions indexed by prediction_id."""
    preds = {}
    if not PREDICTIONS_FILE.exists():
        return preds
    for line in PREDICTIONS_FILE.read_text().splitlines():
        if line.strip():
            try:
                p = json.loads(line)
                preds[p["prediction_id"]] = p
            except Exception:
                pass
    return preds


def load_actuals() -> list[dict]:
    """Load all recorded actuals."""
    actuals = []
    if not ACTUALS_FILE.exists():
        return actuals
    for line in ACTUALS_FILE.read_text().splitlines():
        if line.strip():
            try:
                a = json.loads(line)
                actuals.append(a)
            except Exception:
                pass
    return actuals


def get_uncontributed_records() -> list[dict]:
    """Get list of records with actuals that haven't been contributed yet."""
    contributed_ids = load_contributed_ids()
    actuals = load_actuals()
    predictions = load_predictions()

    uncontributed = []
    for actual in actuals:
        pred_id = actual.get("prediction_id")
        if pred_id and pred_id not in contributed_ids:
            # Only include if we have both actual_cost and actual_turns
            if actual.get("actual_cost") is not None and actual.get("actual_turns") is not None:
                pred = predictions.get(pred_id, {})
                uncontributed.append({
                    "prediction_id": pred_id,
                    "model": actual.get("model", ""),
                    "actual_cost": actual.get("actual_cost"),
                    "actual_turns": actual.get("actual_turns"),
                    "passed": actual.get("passed"),
                    "task_text": actual.get("task", ""),
                    "pred_cost": actual.get("pred_cost"),
                    "pred_turns": actual.get("pred_turns"),
                })
    return uncontributed


def print_summary_table(records: list[dict]) -> None:
    """Print a summary table of uncontributed records."""
    if not records:
        print("\n  No uncontributed records found.\n")
        return

    print(f"\n  Found {len(records)} uncontributed record(s):\n")
    print("  " + "─" * 130)
    print(
        f"  {'ID':<10}  {'Model':<25}  {'Pred cost':>10}  "
        f"{'Actual cost':>11}  {'Turns':>6}  {'Pass':>5}  Task"
    )
    print("  " + "─" * 130)

    for rec in records:
        task_snip = rec["task_text"][:50]
        task_snip = task_snip.replace("\n", " ")
        passed_str = "✓" if rec["passed"] else "✗" if rec["passed"] is False else "?"
        print(
            f"  {rec['prediction_id']:<10}  {rec['model']:<25}  "
            f"${rec['pred_cost']:>9.3f}  ${rec['actual_cost']:>10.3f}  "
            f"{rec['actual_turns']:>5}  {passed_str:>5}  {task_snip}"
        )

    print("  " + "─" * 130)
    print()


def check_hf_token() -> str:
    """Check if HF_TOKEN is set. If not, print instructions and exit."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    print("\n  ⚠️  HF_TOKEN not found. To contribute data, you need a HuggingFace API token.\n")
    print("  Steps to get your token:")
    print("    1. Go to https://huggingface.co/settings/tokens")
    print("    2. Create a new token with 'write' access")
    print("    3. Copy the token and set it as an environment variable:")
    print("       export HF_TOKEN='your_token_here'\n")
    print("  Then try again:\n")
    print("    bin/contribute\n")
    sys.exit(0)


def upload_to_huggingface(records: list[dict], token: str) -> bool:
    """Upload records to HuggingFace dataset using huggingface_hub."""
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError:
        print("  Error: huggingface_hub is not installed.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    try:
        jsonl_lines = []
        for rec in records:
            entry = {
                "prediction_id": rec["prediction_id"],
                "model": rec["model"],
                "pred_cost": rec["pred_cost"],
                "actual_cost": rec["actual_cost"],
                "pred_turns": rec["pred_turns"],
                "actual_turns": rec["actual_turns"],
                "passed": rec["passed"],
                "task_text": rec["task_text"],
                "contributed_at": datetime.now(timezone.utc).isoformat(),
            }
            jsonl_lines.append(json.dumps(entry, separators=(",", ":")))

        # Download existing file content so we can append
        file_content = ""
        try:
            file_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=HF_DATASET_FILE,
                repo_type="dataset",
                token=token,
            )
            with open(file_path) as f:
                file_content = f.read()
        except Exception:
            pass

        if file_content and not file_content.endswith("\n"):
            file_content += "\n"
        file_content += "\n".join(jsonl_lines) + "\n"

        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Add {len(records)} contributed actuals record(s)",
            operations=[
                CommitOperationAdd(
                    path_in_repo=HF_DATASET_FILE,
                    path_or_fileobj=file_content.encode("utf-8"),
                )
            ],
            token=token,
            create_pr=False,
        )

        return True
    except Exception as e:
        print(f"  Error uploading to HuggingFace: {e}", file=sys.stderr)
        return False


def save_contributed_ids(new_ids: list[str]) -> None:
    """Append newly contributed IDs to the contributed.json file."""
    contributed_ids = load_contributed_ids()
    contributed_ids.update(new_ids)

    CONTRIBUTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "contributed_ids": sorted(list(contributed_ids)),
        "last_contributed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(CONTRIBUTED_FILE, "w") as f:
        json.dump(data, f, indent=2)


def prompt_user_selection(records: list[dict]) -> list[dict]:
    """Prompt user to select which records to contribute."""
    print("  Select which records to share (all / pick / none):")
    print("    [a] Share all records")
    print("    [p] Pick individual records")
    print("    [n] Don't share anything")
    print()
    response = input("  Your choice [a/p/n]: ").strip().lower()

    if response == "a":
        return records
    elif response == "p":
        selected = []
        for i, rec in enumerate(records, 1):
            task_snip = rec["task_text"][:50].replace("\n", " ")
            response = input(
                f"  [{i}/{len(records)}] {rec['prediction_id']} "
                f"({rec['model']}) - {task_snip}... Share? [y/n]: "
            ).strip().lower()
            if response in ("y", "yes"):
                selected.append(rec)
        return selected
    else:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Share actual cost/turns data to improve the predictor."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Auto-contribute all uncontributed records without prompting"
    )
    args = parser.parse_args()

    uncontributed = get_uncontributed_records()

    if not uncontributed:
        print("\n  No uncontributed records found.\n")
        sys.exit(0)

    print_summary_table(uncontributed)

    if args.all:
        selected = uncontributed
    else:
        selected = prompt_user_selection(uncontributed)

    if not selected:
        print("  No records selected. Exiting.\n")
        sys.exit(0)

    print(f"\n  Uploading {len(selected)} record(s) to HuggingFace...")

    token = check_hf_token()
    success = upload_to_huggingface(selected, token)

    if success:
        pred_ids = [rec["prediction_id"] for rec in selected]
        save_contributed_ids(pred_ids)
        print(f"\n  ✓ Successfully contributed {len(selected)} record(s)!\n")
        print(f"  View the dataset: https://huggingface.co/datasets/{HF_DATASET_REPO}\n")
    else:
        print("\n  ✗ Upload failed. Records were not marked as contributed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
