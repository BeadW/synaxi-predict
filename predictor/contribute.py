"""
Share actual cost/turns data via GitHub issues to improve the predictor.

Usage:
    python -m predictor.contribute             # Interactive mode
    python -m predictor.contribute --all       # Auto-contribute all uncontributed records
    python -m predictor.contribute --help      # Show this help
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONTRIBUTOR_ID_FILE = DATA_DIR / "contributor_id"
ACTUALS_FILE = DATA_DIR / "actuals_live.jsonl"
CONTRIBUTED_FILE = DATA_DIR / "contributed.json"
REPO = "BeadW/synaxi-predict"


def get_or_create_contributor_id() -> str:
    """Load contributor ID from file, or generate a new one if it doesn't exist."""
    if CONTRIBUTOR_ID_FILE.exists():
        return CONTRIBUTOR_ID_FILE.read_text().strip()

    contributor_id = secrets.token_hex(4)  # 8-char hex ID
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONTRIBUTOR_ID_FILE.write_text(contributor_id)
    print(f"  Generated new contributor ID: {contributor_id}\n")
    return contributor_id


def load_contributed_ids() -> set[str]:
    """Load the set of already-contributed prediction IDs."""
    if not CONTRIBUTED_FILE.exists():
        return set()
    try:
        data = json.loads(CONTRIBUTED_FILE.read_text())
        return set(data.get("contributed_ids", []))
    except Exception:
        return set()


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

    uncontributed = []
    for actual in actuals:
        pred_id = actual.get("prediction_id")
        if pred_id and pred_id not in contributed_ids:
            # Only include if we have both actual_cost and actual_turns
            if actual.get("actual_cost") is not None and actual.get("actual_turns") is not None:
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


def check_gh_auth() -> bool:
    """Check if gh is installed and authenticated."""
    try:
        subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def ensure_contribution_label() -> None:
    """Ensure the 'contribution' label exists in the repo."""
    try:
        subprocess.run(
            [
                "gh",
                "label",
                "create",
                "contribution",
                "--repo",
                REPO,
                "--color",
                "0075ca",
                "--force",
            ],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass  # Label may already exist or other error; continue anyway


def create_github_issue(contributor_id: str, records: list[dict]) -> bool:
    """Create a GitHub issue with the contribution data."""
    # Build the issue body
    body_data = {
        "contributor_id": contributor_id,
        "records": [
            {
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
            for rec in records
        ],
    }

    body = json.dumps(body_data, indent=2)

    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                REPO,
                "--title",
                f"synaxi-predict contribution [{contributor_id}]",
                "--body",
                body,
                "--label",
                "contribution",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        print(f"  ✓ Created issue: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to create issue: {e.stderr}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}", file=sys.stderr)
        return False


def save_contributed_ids(new_ids: list[str]) -> None:
    """Append newly contributed IDs to the contributed.json file."""
    contributed_ids = load_contributed_ids()
    contributed_ids.update(new_ids)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
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

    # Check gh installation and authentication
    if not check_gh_auth():
        print("\n  ⚠️  GitHub CLI (gh) is not installed or not authenticated.\n")
        print("  Steps to set up:")
        print("    1. Install gh: https://cli.github.com/")
        print("    2. Authenticate: gh auth login\n")
        print("  Then try again:\n")
        print("    bin/contribute\n")
        sys.exit(0)

    contributor_id = get_or_create_contributor_id()
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

    print(f"\n  Creating GitHub issue with {len(selected)} record(s)...")

    ensure_contribution_label()
    success = create_github_issue(contributor_id, selected)

    if success:
        pred_ids = [rec["prediction_id"] for rec in selected]
        save_contributed_ids(pred_ids)
        print(f"\n  ✓ Successfully contributed {len(selected)} record(s)!\n")
    else:
        print("\n  ✗ Failed to create contribution issue.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
