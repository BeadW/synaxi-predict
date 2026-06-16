"""
Record actual cost/turns for a previously logged prediction.

Usage:
    python -m predictor.record_actual <prediction_id> --cost 0.42 --turns 18
    python -m predictor.record_actual --list   # show recent unresolved predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "data" / "predictions_live.jsonl"
ACTUALS_FILE = PROJECT_ROOT / "data" / "actuals_live.jsonl"


def list_pending() -> None:
    if not PREDICTIONS_FILE.exists():
        print("No predictions logged yet.")
        return

    actuals_ids = set()
    if ACTUALS_FILE.exists():
        for line in ACTUALS_FILE.read_text().splitlines():
            if line.strip():
                try:
                    actuals_ids.add(json.loads(line)["prediction_id"])
                except Exception:
                    pass

    preds = []
    for line in PREDICTIONS_FILE.read_text().splitlines():
        if line.strip():
            try:
                p = json.loads(line)
                if p["prediction_id"] not in actuals_ids:
                    preds.append(p)
            except Exception:
                pass

    if not preds:
        print("No unresolved predictions.")
        return

    print(f"\n  {'ID':<12}  {'Model':<45}  {'Pred cost':>10}  {'Pred turns':>10}  Task")
    print("  " + "─" * 100)
    for p in preds[-10:]:
        task_snip = p.get("task", "")[:40]
        rec = p.get("recommendation", {})
        print(f"  {p['prediction_id']:<12}  {rec.get('model','?'):<45}  "
              f"${rec.get('est_cost',0):>9.3f}  {rec.get('est_turns',0):>9.1f}  {task_snip}")
    print()


def record(pred_id: str, actual_cost: float | None, actual_turns: int, passed: bool | None) -> None:
    if not PREDICTIONS_FILE.exists():
        print(f"No predictions file found at {PREDICTIONS_FILE}", file=sys.stderr)
        sys.exit(1)

    pred = None
    for line in PREDICTIONS_FILE.read_text().splitlines():
        if line.strip():
            try:
                p = json.loads(line)
                if p["prediction_id"] == pred_id:
                    pred = p
            except Exception:
                pass

    if pred is None:
        print(f"Prediction ID '{pred_id}' not found.", file=sys.stderr)
        sys.exit(1)

    rec = pred.get("recommendation", {})
    actual: dict = {
        "prediction_id":   pred_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "task":            pred.get("task", ""),
        "model":           rec.get("model", ""),
        "pred_cost":       rec.get("est_cost", 0),
        "pred_turns":      rec.get("est_turns", 0),
        "pred_pass_rate":  rec.get("pass_rate", None),
        "actual_cost":     round(actual_cost, 6) if actual_cost is not None else None,
        "actual_turns":    actual_turns,
        "passed":          passed,
        "turns_error":     actual_turns - rec.get("est_turns", 0),
        "code_features":   pred.get("code_features", {}),
    }
    if actual_cost is not None:
        actual["cost_error"] = round(actual_cost - rec.get("est_cost", 0), 6)

    ACTUALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTUALS_FILE, "a") as f:
        f.write(json.dumps(actual, separators=(",", ":")) + "\n")

    print(f"\n  Recorded: {pred_id}")
    print(f"  Model:        {rec.get('model','')}")
    print(f"  Predicted:    ${rec.get('est_cost',0):.3f} / {rec.get('est_turns',0):.0f} turns")
    if actual_cost is not None:
        print(f"  Actual:       ${actual_cost:.3f} / {actual_turns} turns")
        err_pct = (actual_cost - rec.get('est_cost', 0)) / actual_cost * 100 if actual_cost else 0
        print(f"  Cost error:   {err_pct:+.0f}%")
    else:
        print(f"  Actual turns: {actual_turns}  (cost not recorded — add later with --cost <USD>)")
    if passed is not None:
        print(f"  Passed:       {'yes' if passed else 'no'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_id", nargs="?")
    parser.add_argument("--cost",   type=float, default=None, help="Actual cost in USD (optional)")
    parser.add_argument("--turns",  type=int,   help="Actual number of turns")
    parser.add_argument("--passed", type=lambda x: x.lower() in ("1","true","yes"),
                        default=None, help="Did the task pass? (true/false)")
    parser.add_argument("--list",   action="store_true", help="List pending predictions")
    args = parser.parse_args()

    if args.list or not args.prediction_id:
        list_pending()
        return

    if args.turns is None:
        parser.error("--turns is required")

    record(args.prediction_id, args.cost, args.turns, args.passed)


if __name__ == "__main__":
    main()
