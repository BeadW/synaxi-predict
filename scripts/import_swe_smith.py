#!/usr/bin/env python3
"""
Import SWE-smith trajectories into our JSONL run format.

Source: SWE-bench/SWE-smith-trajectories (parquet files in data/swe_smith/)
Records: ~25,826  Models: claude-3-7-sonnet, claude-3-5-sonnet, gpt-4o

Cost is computed from real token counts (tiktoken cl100k_base) applied to
Bedrock/Anthropic pricing. We tokenize each message to get accurate prompt
and completion token totals, then apply the same cache-aware formula used
in import_loong0814.py.

Usage:
    python scripts/import_swe_smith.py
    python scripts/import_swe_smith.py --out data/runs/swe_smith.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tiktoken

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "swe_smith" / "data"

# Bedrock/API pricing: (input, output, cache_write, cache_read) per token
_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-3-7-sonnet-20250219":  (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-3-5-sonnet-20241022":  (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "gpt-4o-2024-08-06":           (2.50/1e6, 10.00/1e6, 0.0,      1.25/1e6),
}
_DEFAULT_PRICING = (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6)

# Use cl100k_base (GPT-4/Claude proxy) for all models
_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _content_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content or "")


def _extract_task_text(msgs: list[dict]) -> str:
    """Pull PR description from first user message, falling back to full content."""
    for m in msgs:
        if m.get("role") == "user":
            text = _content_text(m.get("content", ""))
            hit = re.search(r"<pr_description>(.*?)</pr_description>", text, re.DOTALL)
            if hit:
                return hit.group(1).strip()
            # Strip uploaded_files preamble, return remainder
            cleaned = re.sub(r"<uploaded_files>.*?</uploaded_files>", "", text, flags=re.DOTALL)
            return cleaned.strip()[:2000]
    return ""


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int, num_turns: int) -> float:
    """
    Estimate total cost.

    prompt_tokens = context at the final API call (proxy for unique context written).
    completion_tokens = TOTAL across all N turns.

    For Claude with prompt caching:
      - Unique context written once: prompt_tokens × P_cw
      - Cache re-read (N-1)/2 times on average: prompt_tokens × (N-1)/2 × P_cr
      - Output (total): completion_tokens × P_out

    For GPT-4o (cached reads, no write cost):
      - New input: prompt_tokens × P_in
      - Cached reads: prompt_tokens × (N-1)/2 × P_cr
      - Output (total): completion_tokens × P_out

    For no-cache models:
      - Average context grows from 0 → prompt_tokens: total input ≈ prompt_tokens × N / 2
      - Output (total): completion_tokens × P_out
    """
    p_in, p_out, p_cw, p_cr = _PRICING.get(model, _DEFAULT_PRICING)
    N = max(num_turns, 1)

    if p_cr > 0:
        cache_read_total = prompt_tokens * (N - 1) / 2
        if p_cw > 0:
            return prompt_tokens * p_cw + cache_read_total * p_cr + completion_tokens * p_out
        else:
            return prompt_tokens * p_in + cache_read_total * p_cr + completion_tokens * p_out
    else:
        return prompt_tokens * N / 2 * p_in + completion_tokens * p_out


def process_parquet(path: Path) -> list[dict]:
    df = pd.read_parquet(path)
    records = []

    for _, row in df.iterrows():
        model = str(row["model"])
        resolved = row.get("resolved")
        instance_id = str(row["instance_id"])

        msgs_raw = row["messages"]
        try:
            msgs = json.loads(msgs_raw) if isinstance(msgs_raw, str) else list(msgs_raw)
        except Exception:
            continue

        # Tokenize each message
        msg_tokens = []
        for m in msgs:
            if not isinstance(m, dict):
                msg_tokens.append((None, 0))
                continue
            role = m.get("role", "")
            text = _content_text(m.get("content", ""))
            msg_tokens.append((role, _count_tokens(text)))

        # num_turns = number of assistant responses
        num_turns = sum(1 for role, _ in msg_tokens if role == "assistant")
        if num_turns == 0:
            continue

        # prompt_tokens_raw = context for the final API call = all tokens before the last assistant response
        last_asst_idx = max(i for i, (role, _) in enumerate(msg_tokens) if role == "assistant")
        prompt_tokens = sum(tok for i, (_, tok) in enumerate(msg_tokens) if i < last_asst_idx)

        # completion_tokens = TOTAL across ALL assistant responses (not just the last one)
        completion_tokens = sum(tok for role, tok in msg_tokens if role == "assistant")

        task_text = _extract_task_text(msgs)
        if not task_text:
            continue

        total_cost = _compute_cost(model, prompt_tokens, completion_tokens, num_turns)

        records.append({
            "task_id":           f"swe_smith/{instance_id}",
            "strategy":          model,
            "complexity":        None,
            "category":          "code/swe_smith",
            "mode":              "multi-turn",
            "source":            "swe_smith",
            "task_text":         task_text,
            "prompt_tokens_raw": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
            "cache_read_tokens": 0,
            "total_cost_usd":    round(total_cost, 6),
            "cost_is_estimated": True,
            "last_call_cost":    0.0,
            "num_turns":         num_turns,
            "tool_call_count":   num_turns,
            "quality_score":     None,
            "passed_criteria":   bool(resolved) if resolved is not None else None,
            "elapsed_s":         0,
            "run_label":         path.stem,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/runs/swe_smith.jsonl", type=Path)
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(DATA_DIR.glob("ticks-*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {DATA_DIR}")
        return

    print(f"Found {len(parquet_files)} parquet files")
    all_records: list[dict] = []

    for i, pf in enumerate(parquet_files):
        print(f"  [{i+1}/{len(parquet_files)}] {pf.name}...", flush=True)
        recs = process_parquet(pf)
        all_records.extend(recs)
        print(f"    → {len(recs)} records  (total: {len(all_records)})", flush=True)

    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    # Summary
    by_model: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_model[r["strategy"]].append(r)

    print(f"\n{'Model':<45}  {'Records':>7}  {'Turns':>7}  {'Cost':>8}  {'Pass%':>6}")
    print("─" * 80)
    for model, recs in sorted(by_model.items()):
        turns = [r["num_turns"] for r in recs]
        costs = [r["total_cost_usd"] for r in recs]
        labeled = [r for r in recs if r["passed_criteria"] is not None]
        pass_pct = f"{sum(r['passed_criteria'] for r in labeled)/len(labeled):.0%}" if labeled else "n/a"
        print(
            f"  {model:<43}  {len(recs):>7}  "
            f"{sum(turns)/len(turns):>6.1f}  "
            f"${sum(costs)/len(costs):>7.3f}  "
            f"{pass_pct:>6}"
        )

    print(f"\nWrote {len(all_records)} records → {out_path}")


if __name__ == "__main__":
    main()
