"""
Server-side script to harvest and validate contribution issues.

Fetches open GitHub issues with the 'contribution' label, validates the
submitted data by re-running predictions, and appends accepted records to
the training dataset.

Usage:
    python scripts/process_contributions.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONTRIBUTIONS_FILE = DATA_DIR / "runs" / "contributions.jsonl"
REPO = "BeadW/synaxi-predict"

# Known models for validation
KNOWN_MODELS = {
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5-20251001",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "gpt-5-2025-08-07",
    "gpt-5.2",
    "o1-mini-2024-09-12",
    "deepseek-chat",
    "deepseek-v2.5",
    "gemini-1.5-pro-002",
    "llama-3.1-405b-instruct",
    "llama-3.1-70b-instruct",
    "llama-3.2-90b-vision-instruct",
    "llama-v3p1-405b-instruct",
    "llama-v3p1-70b-instruct",
    "llama-v3p3-70b-instruct",
    "qwen-2.5-72b-instruct",
    "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "single-haiku",
    "single-sonnet",
}


def fetch_contribution_issues() -> list[dict]:
    """Fetch all open contribution issues from the repository."""
    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--repo",
                REPO,
                "--label",
                "contribution",
                "--state",
                "open",
                "--json",
                "number,title,body",
                "--limit",
                "100",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error fetching issues: {e.stderr}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return []


def validate_record(record: dict) -> tuple[bool, str]:
    """
    Validate a single record.

    Returns (is_valid, error_message)
    """
    # Basic field validation
    task_text = record.get("task_text")
    model = record.get("model")
    actual_cost = record.get("actual_cost")
    actual_turns = record.get("actual_turns")
    passed = record.get("passed")

    if not task_text or not isinstance(task_text, str):
        return False, "Invalid or missing task_text"

    if model not in KNOWN_MODELS:
        return False, f"Unknown model: {model}"

    if not isinstance(actual_cost, (int, float)) or actual_cost < 0.001 or actual_cost > 100:
        return False, f"Invalid actual_cost: must be in [0.001, 100], got {actual_cost}"

    if not isinstance(actual_turns, int) or actual_turns < 1 or actual_turns > 500:
        return False, f"Invalid actual_turns: must be in [1, 500], got {actual_turns}"

    if not isinstance(passed, bool):
        return False, "Invalid passed: must be true or false"

    # Re-run predictor for the model
    try:
        from predictor.predict import predict

        results = predict(task_text)
        pred_result = next((r for r in results if r["model"] == model), None)

        if pred_result is None:
            return False, f"Predictor did not return result for {model}"

        pred_cost = pred_result["est_cost"]
        pred_turns = pred_result["est_turns"]
        submitted_cost = record.get("pred_cost")
        submitted_turns = record.get("pred_turns")

        # Check submitted predictions against current model output.
        # 20% tolerance (not 5%) because the model gets retrained between
        # submission and processing — predictions legitimately drift over time.
        if submitted_cost is not None:
            cost_error = abs(submitted_cost - pred_cost) / max(pred_cost, 0.001)
            if cost_error > 0.20:
                return (
                    False,
                    f"Submitted pred_cost {submitted_cost:.4f} differs >20% from current "
                    f"predictor output {pred_cost:.4f}",
                )

        if submitted_turns is not None:
            turns_error = abs(submitted_turns - pred_turns) / max(pred_turns, 1)
            if turns_error > 0.20:
                return (
                    False,
                    f"Submitted pred_turns {submitted_turns:.1f} differs >20% from current "
                    f"predictor output {pred_turns:.1f}",
                )

    except Exception as e:
        return False, f"Error re-running predictor: {str(e)}"

    return True, ""


def append_to_training_data(
    records: list[dict],
) -> None:
    """Append validated records to the training dataset."""
    CONTRIBUTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CONTRIBUTIONS_FILE, "a") as f:
        for record in records:
            entry = {
                "task_id": f"contribution/{record['prediction_id']}",
                "strategy": record["model"],
                "mode": "multi-turn",
                "task_text": record["task_text"],
                "prompt_tokens_raw": None,
                "completion_tokens": None,
                "num_turns": record["actual_turns"],
                "total_cost_usd": record["actual_cost"],
                "cost_is_estimated": False,
                "passed_criteria": record["passed"],
            }
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")


def close_issue(issue_number: int, comment: str) -> None:
    """Close an issue with a comment."""
    try:
        subprocess.run(
            [
                "gh",
                "issue",
                "comment",
                str(issue_number),
                "--repo",
                REPO,
                "--body",
                comment,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        subprocess.run(
            [
                "gh",
                "issue",
                "close",
                str(issue_number),
                "--repo",
                REPO,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except Exception as e:
        print(f"Warning: Failed to close issue #{issue_number}: {e}", file=sys.stderr)


def comment_on_issue(issue_number: int, comment: str) -> None:
    """Add a comment to an issue."""
    try:
        subprocess.run(
            [
                "gh",
                "issue",
                "comment",
                str(issue_number),
                "--repo",
                REPO,
                "--body",
                comment,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except Exception as e:
        print(f"Warning: Failed to comment on issue #{issue_number}: {e}", file=sys.stderr)


def main() -> None:
    issues = fetch_contribution_issues()
    if not issues:
        print("No contribution issues found.")
        return

    print(f"Processing {len(issues)} contribution issue(s)...\n")

    total_records = 0
    total_accepted = 0
    total_rejected = 0
    contributor_trust = defaultdict(lambda: {"accepted": 0, "rejected": 0})

    for issue in issues:
        issue_number = issue["number"]
        body = issue["body"]

        try:
            body_data = json.loads(body)
        except json.JSONDecodeError:
            print(
                f"Issue #{issue_number}: Failed to parse JSON body. Skipping.",
                file=sys.stderr,
            )
            continue

        contributor_id = body_data.get("contributor_id")
        records = body_data.get("records", [])
        total_records += len(records)

        print(f"Issue #{issue_number} ({contributor_id}): {len(records)} record(s)")

        passed_records = []
        failed_records = []

        for rec in records:
            is_valid, error_msg = validate_record(rec)
            if is_valid:
                passed_records.append(rec)
                contributor_trust[contributor_id]["accepted"] += 1
                total_accepted += 1
            else:
                failed_records.append((rec.get("prediction_id"), error_msg))
                contributor_trust[contributor_id]["rejected"] += 1
                total_rejected += 1

        # Handle results
        if failed_records:
            # Comment with failures
            comment_lines = [f"Validation failed for {len(failed_records)} record(s):\n"]
            for pred_id, error in failed_records:
                comment_lines.append(f"- {pred_id}: {error}")
            comment = "\n".join(comment_lines)
            comment_on_issue(issue_number, comment)
            print(f"  {len(failed_records)} record(s) rejected")
        else:
            # All passed: append to training data and close issue
            append_to_training_data(passed_records)
            close_issue(
                issue_number,
                f"All {len(passed_records)} record(s) validated and added to training data. Thank you!",
            )
            print(f"  All {len(passed_records)} record(s) accepted and added to training data")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Issues processed:     {len(issues)}")
    print(f"Total records:        {total_records}")
    print(f"Records accepted:     {total_accepted}")
    print(f"Records rejected:     {total_rejected}")
    print()
    print("Contributor trust scores:")
    for contrib_id in sorted(contributor_trust.keys()):
        stats = contributor_trust[contrib_id]
        total = stats["accepted"] + stats["rejected"]
        acceptance_rate = (
            f"{stats['accepted'] / total * 100:.0f}%"
            if total > 0
            else "N/A"
        )
        print(
            f"  {contrib_id}: {stats['accepted']}/{total} accepted ({acceptance_rate})"
        )


if __name__ == "__main__":
    main()
