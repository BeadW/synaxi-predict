#!/usr/bin/env python3
"""
Import loong0814/openhands_trajectories dataset into our JSONL run format.

Each mini archive has one JSON file per task — the FINAL LLM call, with the
full conversation history embedded in the messages array.

Cost is computed accurately using Bedrock/Anthropic pricing + cache token math:

  With prompt caching (Claude models):
    total_cost  = cache_write × P_write + cache_read_total × P_read + output_total × P_out
    cache_write = prompt_tokens_last          (each context token written once)
    cache_read  = prompt_tokens_last × (N-1)/2  (grows linearly across turns)
    output      = completion_tokens_last × N  (roughly constant per turn)

  Without caching (Qwen, GPT-5):
    total_cost  = input_total × P_in + output_total × P_out
    input_total = prompt_tokens_last × N / 2  (linear context growth, avg = last/2)
    output      = completion_tokens_last × N

Usage:
    python scripts/import_loong0814.py
    python scripts/import_loong0814.py --out data/runs/loong0814.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

PROJECT_ROOT = Path(__file__).parent.parent

MINI_FILES = [
    "claude-sonnet-3.7_4runs_mini.tar.gz",
    "claude-sonnet-4_4runs_mini.tar.gz",
    "claude-sonnet-4.5_4runs_mini.tar.gz",
    "gpt-5_4runs_mini.tar.gz",
    "qwen3-coder-480b-a35b-instruct-4runs_mini.tar.gz",
]

# Canonical model IDs to store in records
MODEL_ID_MAP = {
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",
    "claude-sonnet-4":            "claude-sonnet-4-20250514",
    "claude-sonnet-4-5":          "claude-sonnet-4-5-20251001",
    "claude-4-5-sonnet":          "claude-sonnet-4-5-20251001",
    "gpt-5":                      "gpt-5",
}

# Bedrock / Anthropic API pricing per token
# (input, output, cache_write, cache_read)
# For Anthropic models cache_write = $3.75/M, cache_read = $0.30/M (Bedrock on-demand = API rates)
# For OpenAI GPT-5 caching: cached reads cost $5/M (50% of input), cache_write = 0
#   encoded as (input=10, output=40, cache_write=0, cache_read=5)
_PRICING: dict[str, tuple[float, float, float, float]] = {
    # Claude models — Bedrock on-demand pricing
    "claude-3-7-sonnet-20250219":        (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-sonnet-4-20250514":          (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-sonnet-4-5-20250929":        (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    # GPT-5 — OpenAI, cached reads at $5/M (50% discount), no explicit cache_write cost
    "gpt-5-2025-08-07":                  (10.00/1e6, 40.00/1e6, 0.0, 5.00/1e6),
    # Qwen3 Coder — self-hosted / Together.ai estimate, no caching
    "Qwen3-Coder-480B-A35B-Instruct-FP8": (1.50/1e6, 6.00/1e6, 0.0, 0.0),
}
_DEFAULT_PRICING = (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6)


def _compute_total_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int,
    num_turns: int,
) -> float:
    """
    Compute total cost.

    prompt_tokens     = context at the final API call (proxy for unique context written)
    completion_tokens = TOTAL output tokens across ALL N turns (summed from messages history)
    num_turns         = N (used for cache read estimation)
    """
    p_in, p_out, p_cw, p_cr = _PRICING.get(model, _DEFAULT_PRICING)
    N = max(num_turns, 1)

    if p_cr > 0 and cache_read_tokens > 0:
        cache_read_total = prompt_tokens * (N - 1) / 2
        if p_cw > 0:
            return prompt_tokens * p_cw + cache_read_total * p_cr + completion_tokens * p_out
        else:
            return prompt_tokens * p_in + cache_read_total * p_cr + completion_tokens * p_out
    else:
        return prompt_tokens * N / 2 * p_in + completion_tokens * p_out


def _model_from_dir(name: str) -> str:
    """Extract model ID from a run directory name like:
    claude-3-7-sonnet-20250219_maxiter_100_N_v0.31.0-no-hint-juan-inst-t1-run_1
    """
    m = re.match(r"^([^_]+(?:_[^_]+)*?)_maxiter_", name)
    if m:
        raw = m.group(1)
        return MODEL_ID_MAP.get(raw, raw)
    return name


def _extract_issue_description(user_content) -> str:
    """Pull the GitHub issue text from the first user message content."""
    if isinstance(user_content, list):
        text = " ".join(
            p.get("text", "") for p in user_content if isinstance(p, dict) and p.get("type") == "text"
        )
    else:
        text = str(user_content or "")

    m = re.search(r"<issue_description>(.*?)</issue_description>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback: text after the file upload notice
    after_upload = re.sub(r"<uploaded_files>.*?</uploaded_files>", "", text, flags=re.DOTALL)
    return after_upload.strip()[:2000]


def load_swebench_problems() -> dict[str, str]:
    """Load problem_statement keyed by instance_id from SWE-bench Verified."""
    from datasets import load_dataset
    print("Loading SWE-bench Verified problem statements...", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {row["instance_id"]: row["problem_statement"] for row in ds}


def process_archive(tar_path: Path, problems: dict[str, str], extract_dir: Path) -> list[dict]:
    print(f"\nExtracting {tar_path.name}...", flush=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir)

    records = []
    # Find all llm_completions dirs
    for run_dir in sorted(extract_dir.rglob("llm_completions")):
        model_dir = run_dir.parent
        model_id = _model_from_dir(model_dir.name)
        run_label = model_dir.name  # for dedup

        task_dirs = sorted(t for t in run_dir.iterdir() if t.is_dir())
        resolved_count = 0

        for task_dir in task_dirs:
            instance_id = task_dir.name
            json_files = list(task_dir.glob("*.json"))
            if not json_files:
                continue

            data = json.loads(json_files[0].read_text())
            msgs = data.get("messages", [])
            usage = (data.get("response") or {}).get("usage", {})
            last_call_cost = float(data.get("cost") or 0)
            total_tokens_last = int(usage.get("total_tokens", 0))
            # prompt_tokens_raw = final call's context (proxy for unique context written)
            prompt_tokens_last = int(usage.get("prompt_tokens", 0))
            # Anthropic: cache_read_input_tokens; OpenAI: prompt_tokens_details.cached_tokens
            cache_read = int(
                usage.get("cache_read_input_tokens", 0)
                or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            )

            # Count agent turns and sum total completion tokens from conversation history.
            # The messages array is the full conversation for the final API call, so
            # assistant messages in it represent all turns' completions.
            num_turns = 0
            completion_tokens = 0
            for m in msgs:
                if m.get("role") == "assistant":
                    num_turns += 1
                    content = m.get("content", "") or ""
                    if isinstance(content, list):
                        text = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
                    else:
                        text = str(content)
                    completion_tokens += len(_ENC.encode(text, disallowed_special=()))
            if num_turns == 0:
                continue

            # Task text: prefer SWE-bench problem statement
            task_text = problems.get(instance_id, "")
            if not task_text:
                for m in msgs:
                    if m.get("role") == "user":
                        task_text = _extract_issue_description(m.get("content", ""))
                        break

            total_cost = _compute_total_cost(
                model_id, prompt_tokens_last, completion_tokens, cache_read, num_turns
            )

            records.append({
                "task_id":           f"swebench_verified/{instance_id}",
                "strategy":          model_id,
                "complexity":        None,
                "category":          "code/swebench_verified",
                "mode":              "multi-turn",
                "source":            "loong0814",
                "task_text":         task_text,
                "prompt_tokens_raw": prompt_tokens_last,
                "completion_tokens": completion_tokens,
                "total_tokens":      total_tokens_last,
                "cache_read_tokens": cache_read,
                "total_cost_usd":    round(total_cost, 6),
                "cost_is_estimated": True,
                "last_call_cost":    round(last_call_cost, 6),
                "num_turns":         num_turns,
                "tool_call_count":   num_turns,
                "quality_score":     None,
                "passed_criteria":   None,
                "elapsed_s":         0,
                "run_label":         run_label,
                "timestamp":         datetime.now(timezone.utc).isoformat(),
            })

        print(
            f"  {model_id:<55}  {len(task_dirs):>3} tasks  "
            f"{len(records):>4} records so far"
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/runs/loong0814.jsonl", type=Path)
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    problems = load_swebench_problems()
    print(f"Loaded {len(problems)} SWE-bench Verified problem statements")

    from huggingface_hub import hf_hub_download

    all_records: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in MINI_FILES:
            print(f"\nDownloading {filename}...", flush=True)
            try:
                local = hf_hub_download(
                    "loong0814/openhands_trajectories",
                    filename,
                    repo_type="dataset",
                )
            except Exception as e:
                print(f"  Skipped ({e})")
                continue

            extract_dir = Path(tmpdir) / filename.replace(".tar.gz", "")
            extract_dir.mkdir()
            recs = process_archive(Path(local), problems, extract_dir)
            all_records.extend(recs)

    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    # Summary
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
