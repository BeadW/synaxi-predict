#!/usr/bin/env python3
"""
Import loong0814 full archives into our JSONL run format.

These archives (claude-sonnet-4.5_new_v2_4runs.tar.gz, gpt_5.2_4runs.tar.gz)
contain per-task output.jsonl files with:
  - metrics.token_usages: per-turn token breakdown (one entry per API call)
  - metrics.accumulated_cost: real total cost from the API (NOT estimated)
  - instruction: full task prompt with embedded issue description

We extract from the last turn's token_usages entry:
  - prompt_tokens_raw = last_usage['prompt_tokens']  (final context size)
  - completion_tokens  = last_usage['completion_tokens']
  - num_turns          = len(token_usages)
  - total_cost_usd     = metrics['accumulated_cost']  (real API cost)

Usage:
    python scripts/import_loong0814_full.py
    python scripts/import_loong0814_full.py --out data/runs/loong0814_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "loong0814_full"

FULL_ARCHIVES = [
    "claude-sonnet-4.5_new_v2_4runs.tar.gz",
    "gpt_5.2_4runs.tar.gz",
]


def _extract_issue_description(instruction: str) -> str:
    """Pull GitHub issue text from the instruction prompt."""
    m = re.search(r"<issue_description>(.*?)</issue_description>", instruction, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: strip file listing headers and return first 2000 chars
    cleaned = re.sub(r"<uploaded_files>.*?</uploaded_files>", "", instruction, flags=re.DOTALL)
    return cleaned.strip()[:2000]


def _normalize_model(raw: str) -> str:
    """Normalize 'openai/gpt-5.2' → 'gpt-5.2', pass others through."""
    return raw.split("/")[-1] if "/" in raw else raw


def process_archive(tar_path: Path) -> list[dict]:
    print(f"\nProcessing {tar_path.name}...", flush=True)
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()  # (instance_id, run_label) dedup

    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith("output.jsonl") and not m.isdir()]
        print(f"  Found {len(members)} output.jsonl file(s)")

        for member in sorted(members, key=lambda m: m.name):
            run_label = Path(member.name).parent.name
            f = tf.extractfile(member)
            if f is None:
                continue
            content = f.read().decode("utf-8", errors="replace")
            run_records = 0
            skipped = 0

            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    skipped += 1
                    continue

                instance_id = d.get("instance_id", "")
                if not instance_id:
                    skipped += 1
                    continue

                key = (instance_id, run_label)
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)

                m = d.get("metrics") or {}
                token_usages = m.get("token_usages", [])
                if not token_usages:
                    skipped += 1
                    continue

                # Model name from the first token usage entry
                model = _normalize_model(token_usages[0].get("model", "unknown"))
                num_turns = len(token_usages)

                # prompt_tokens_raw = last call's context size (proxy for unique context written,
                # used by avg_prompt_by_model in the predictor).
                # completion_tokens  = TOTAL across all turns (from accumulated_token_usage).
                # total_cost_usd     = real API cost from accumulated_cost (not estimated).
                last_usage = token_usages[-1]
                prompt_tokens = int(last_usage.get("prompt_tokens", 0))
                acc_usage = m.get("accumulated_token_usage") or {}
                completion_tokens = int(acc_usage.get("completion_tokens", 0))
                total_cost = float(m.get("accumulated_cost", 0) or 0)

                if num_turns == 0 or prompt_tokens == 0:
                    skipped += 1
                    continue

                instruction = d.get("instruction", "")
                task_text = _extract_issue_description(instruction)

                records.append({
                    "task_id":           f"swebench_verified/{instance_id}",
                    "strategy":          model,
                    "complexity":        None,
                    "category":          "code/swebench_verified",
                    "mode":              "multi-turn",
                    "source":            "loong0814_full",
                    "task_text":         task_text,
                    "prompt_tokens_raw": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_cost_usd":    round(total_cost, 6),
                    "cost_is_estimated": False,
                    "num_turns":         num_turns,
                    "passed_criteria":   None,
                    "run_label":         run_label,
                    "timestamp":         datetime.now(timezone.utc).isoformat(),
                })
                run_records += 1

            print(f"    {run_label}: {run_records} records, {skipped} skipped")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/runs/loong0814_full.jsonl", type=Path)
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for filename in FULL_ARCHIVES:
        tar_path = DATA_DIR / filename
        if not tar_path.exists():
            print(f"Skipping {filename} (not found at {tar_path})")
            continue
        recs = process_archive(tar_path)
        all_records.extend(recs)

    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    from collections import defaultdict
    by_model: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_model[r["strategy"]].append(r)

    print(f"\n{'Model':<55}  {'Records':>7}  {'Avg turns':>9}  {'Avg cost':>9}")
    print("─" * 90)
    for model, recs in sorted(by_model.items()):
        turns = [r["num_turns"] for r in recs]
        costs = [r["total_cost_usd"] for r in recs]
        print(
            f"  {model:<53}  {len(recs):>7}  "
            f"{sum(turns)/len(turns):>8.1f}  "
            f"${sum(costs)/len(costs):>8.3f}"
        )

    print(f"\nWrote {len(all_records)} records → {out_path}")


if __name__ == "__main__":
    main()
