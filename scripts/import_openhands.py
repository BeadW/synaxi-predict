#!/usr/bin/env python3
"""
Import OpenHands SWE-bench evaluation outputs into our JSONL run format.

Downloads every output.jsonl file from OpenHands/openhands-evaluation-outputs,
tokenizes the history to get accurate prompt/completion token counts, then
calculates cost from token counts using the exact pricing formula.

Usage:
    python scripts/import_openhands.py
    python scripts/import_openhands.py --out data/runs/openhands_swebench.jsonl
    python scripts/import_openhands.py --models claude  # filter to claude only
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import tiktoken

PROJECT_ROOT = Path(__file__).parent.parent

_ENC = tiktoken.get_encoding("cl100k_base")

# Pricing: (input, output, cache_write, cache_read) per token
_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-3-5-haiku-20241022":  (0.80/1e6,  4.00/1e6, 1.00/1e6, 0.08/1e6),
    "claude-3-5-sonnet-20240620": (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-3-5-sonnet-20241022": (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-3-7-sonnet-20250219": (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "gpt-4o-2024-05-13":          (5.00/1e6, 15.00/1e6, 0.0,      2.50/1e6),
    "gpt-4o-2024-08-06":          (2.50/1e6, 10.00/1e6, 0.0,      1.25/1e6),
    "gpt-4o-mini-2024-07-18":     (0.15/1e6,  0.60/1e6, 0.0,      0.075/1e6),
    "o1-mini-2024-09-12":         (3.00/1e6, 12.00/1e6, 0.0,      0.0),
}
_DEFAULT_PRICING = (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6)


def _model_from_path(path: str) -> str:
    m = re.search(r"CodeActAgent/([^/]+)_maxiter_", path)
    if not m:
        return "unknown"
    return m.group(1).replace("@", "-")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(str(text or ""), disallowed_special=()))


def _history_tokens(history: list) -> tuple[int, int, int]:
    """
    Return (prompt_tokens, completion_tokens, num_turns) from a history list.

    Supports two formats:
      - New (dict): each item is {source, message, action, args, ...}
      - Old (list): each item is [action_dict, observation_dict]

    prompt_tokens     = tokens in all history before the final agent response
    completion_tokens = tokens in the final agent response
    num_turns         = number of agent actions
    """
    items = []  # list of (source, tokens)

    for h in history:
        if isinstance(h, dict):
            # New format: single dict per event
            src = h.get("source", "user")
            msg = h.get("message", "") or ""
            args = h.get("args") or {}
            content = args.get("content", "") or args.get("command", "") or args.get("code", "") or ""
            tok = _count_tokens(f"{msg} {content}".strip())
            items.append((src, tok))
        elif isinstance(h, list) and len(h) == 2:
            # Old format: [action_dict, observation_dict]
            action, obs = h[0], h[1]
            if not isinstance(action, dict):
                continue
            src = action.get("source", "user")
            msg = action.get("message", "") or ""
            args = action.get("args") or {}
            action_content = args.get("content", "") or args.get("command", "") or args.get("code", "") or ""
            action_tok = _count_tokens(f"{msg} {action_content}".strip())
            items.append((src, action_tok))
            # Observation goes to context as user/env feedback
            if isinstance(obs, dict):
                obs_content = obs.get("content", "") or obs.get("message", "") or ""
                obs_tok = _count_tokens(obs_content)
                if obs_tok > 0:
                    items.append(("user", obs_tok))

    num_turns = sum(1 for src, _ in items if src == "agent")
    if num_turns == 0:
        return 0, 0, 0

    last_agent_idx = max(i for i, (src, _) in enumerate(items) if src == "agent")
    # prompt_tokens = context at the final agent call (all items before last agent response)
    prompt_tokens = sum(tok for i, (_, tok) in enumerate(items) if i < last_agent_idx)
    # completion_tokens = TOTAL across ALL agent responses (not just the last one)
    completion_tokens = sum(tok for src, tok in items if src == "agent")

    return prompt_tokens, completion_tokens, num_turns


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int, num_turns: int) -> float:
    """
    prompt_tokens = context at final call; completion_tokens = TOTAL across all turns.
    """
    p_in, p_out, p_cw, p_cr = _PRICING.get(model, _DEFAULT_PRICING)
    N = max(num_turns, 1)
    cache_reads = prompt_tokens * (N - 1) / 2
    if p_cw > 0:
        return prompt_tokens * p_cw + cache_reads * p_cr + completion_tokens * p_out
    elif p_cr > 0:
        return prompt_tokens * p_in + cache_reads * p_cr + completion_tokens * p_out
    else:
        return prompt_tokens * N / 2 * p_in + completion_tokens * p_out


def load_jsonl(path: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        "OpenHands/openhands-evaluation-outputs", path, repo_type="dataset"
    )
    rows = []
    with open(local) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def convert(model: str, rows: list[dict]) -> list[dict]:
    records = []
    for r in rows:
        instance_id = r.get("instance_id", "")
        if not instance_id:
            continue

        history  = r.get("history") or []
        rep      = r.get("report") or {}
        resolved = bool(rep.get("resolved", False))

        prompt_tokens, completion_tokens, num_turns = _history_tokens(history)
        if num_turns == 0:
            continue

        cost = _calculate_cost(model, prompt_tokens, completion_tokens, num_turns)

        # Task text: first user message (handle both history formats)
        instruction = r.get("instruction") or ""
        for h in history:
            if isinstance(h, dict) and h.get("source") == "user":
                instruction = h.get("message", "") or instruction
                break
            elif isinstance(h, list) and len(h) == 2 and isinstance(h[0], dict):
                if h[0].get("source") == "user":
                    instruction = h[0].get("message", "") or instruction
                    break

        records.append({
            "task_id":           f"swebench/{instance_id}",
            "strategy":          model,
            "complexity":        None,
            "category":          "code/swebench",
            "mode":              "multi-turn",
            "source":            "openhands",
            "task_text":         instruction,
            "prompt_tokens_raw": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
            "cache_read_tokens": None,
            "total_cost_usd":    round(cost, 6),
            "cost_is_estimated": True,
            "last_call_cost":    0.0,
            "num_turns":         num_turns,
            "tool_call_count":   num_turns,
            "quality_score":     1.0 if resolved else 0.0,
            "passed_criteria":   resolved,
            "elapsed_s":         0,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/runs/openhands_swebench.jsonl", type=Path)
    parser.add_argument("--models", default=None,
                        help="Substring filter on model name (e.g. 'claude')")
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import list_repo_files
    all_files = list(list_repo_files(
        "OpenHands/openhands-evaluation-outputs", repo_type="dataset"
    ))
    jsonl_paths = [
        f for f in all_files
        if f.endswith("output.jsonl") and "SWE-bench_Lite" in f
    ]

    if args.models:
        jsonl_paths = [p for p in jsonl_paths if args.models.lower() in p.lower()]

    print(f"Found {len(jsonl_paths)} output files to import\n")

    all_records: list[dict] = []
    for path in jsonl_paths:
        model = _model_from_path(path)
        print(f"  {model}...", end=" ", flush=True)
        rows = load_jsonl(path)
        recs = convert(model, rows)
        resolved = sum(1 for r in recs if r["passed_criteria"])
        turns = [r["num_turns"] for r in recs if r["num_turns"] > 0]
        costs = [r["total_cost_usd"] for r in recs]
        print(
            f"{len(recs)} tasks  {resolved} resolved ({100*resolved//max(len(recs),1)}%)  "
            f"turns={statistics.mean(turns):.0f}  cost=${statistics.mean(costs):.3f}" if recs else "0 tasks"
        )
        all_records.extend(recs)

    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    print(f"\nWrote {len(all_records)} records → {out_path}")


if __name__ == "__main__":
    main()
