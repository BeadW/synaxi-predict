#!/usr/bin/env python3
"""
Benchmark runner for agentic coding tasks.

Each feature file defines a real multi-turn coding task. The agent gets a
sandbox directory with fixture files, runs Claude Code against it, and we
record the full session cost, turn count, and token usage.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --strategies single-haiku,single-sonnet
    python scripts/run_benchmark.py --feature code/debug
    python scripts/run_benchmark.py --limit 5
    python scripts/run_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from harness.agent import run_task  # noqa: E402
from harness.sandbox import TaskSandbox  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "data" / "runs"

MODEL_FOR_STRATEGY = {
    "single-haiku":  "haiku",
    "single-sonnet": "sonnet",
    "single-opus":   "opus",
}
ALL_STRATEGIES = list(MODEL_FOR_STRATEGY.keys())


COMPLEXITY_RANK = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "Competition": 4}


def _feature_complexity(fp: Path) -> str:
    """Read the Complexity: tag from a feature file."""
    for line in fp.read_text().splitlines()[:6]:
        m = re.match(r"\s*Complexity:\s*(\S+)", line)
        if m:
            return m.group(1)
    return "T1"


def interleave_by_difficulty(features: list) -> list:
    """
    Round-robin across difficulty buckets so as data collection progresses
    we always have a mix of easy/medium/hard rather than all-easy then all-hard.
    Order within each bucket is alphabetical (stable).
    """
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for fp in features:
        tier = _feature_complexity(fp)
        buckets[tier].append(fp)

    tier_order = sorted(buckets, key=lambda t: COMPLEXITY_RANK.get(t, 99))
    result = []
    while any(buckets[t] for t in tier_order):
        for tier in tier_order:
            if buckets[tier]:
                result.append(buckets[tier].pop(0))
    return result


def load_completed() -> set[tuple[str, str]]:
    """Return set of (task_id, strategy) pairs already recorded."""
    done = set()
    for f in RUNS_DIR.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("total_cost_usd", 0) > 0.001:
                        done.add((r["task_id"], r["strategy"]))
                except Exception:
                    pass
    return done


# --------------------------------------------------------------------------- #
# Feature file parsing
# --------------------------------------------------------------------------- #

def _is_multi_turn(content: str) -> bool:
    return "Mode: multi-turn" in content or "project files" in content.lower()


def _parse_project_files(docstring: str) -> dict[str, str]:
    """
    Parse the '=== filename ===' block format into {filename: content}.

    Format:
        === foo.py ===
        def foo(): ...
        === bar.py ===
        import foo
    """
    files: dict[str, str] = {}
    current_name: Optional[str] = None
    current_lines: list[str] = []

    for line in docstring.splitlines():
        m = re.match(r"\s*===\s*(.+?)\s*===\s*$", line)
        if m:
            if current_name is not None:
                files[current_name] = textwrap.dedent("\n".join(current_lines)).strip() + "\n"
            current_name = m.group(1).strip()
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        files[current_name] = textwrap.dedent("\n".join(current_lines)).strip() + "\n"

    return files


def _extract_docstring_block(content: str, after_keyword: str) -> Optional[str]:
    pattern = re.escape(after_keyword) + r'\s*"""\n(.*?)\n\s*"""'
    m = re.search(pattern, content, re.DOTALL)
    return textwrap.dedent(m.group(1)) if m else None


def build_prompt(feature_path: Path) -> tuple[str, dict[str, str]]:
    """
    Returns (prompt_string, files_dict).
    files_dict is non-empty for multi-turn sandbox tasks.
    """
    content = feature_path.read_text()

    if _is_multi_turn(content):
        # Extract fixture files from Background docstring
        raw = _extract_docstring_block(content, "Given the following project files:")
        files = _parse_project_files(raw) if raw else {}

        # Extract the task instruction from the Scenario Given step
        m = re.search(r"Given an agent is tasked with (.+)", content)
        task = m.group(1).strip() if m else "complete the task"

        file_list = ", ".join(files.keys())
        prompt = (
            f"You are working in a Python project. "
            f"The project contains: {file_list}.\n\n"
            f"Task: {task}\n\n"
            f"Use your tools to read the files, make the necessary changes, "
            f"and verify your solution works. Run pytest to confirm all tests pass."
        )
        return prompt, files

    # ── Single-turn fallback (inline code tasks) ───────────────────────────
    code_block = _extract_docstring_block(content, "Given the following broken Python code:")
    if code_block is None:
        code_block = _extract_docstring_block(content, "Given the following Python code:")

    if code_block is not None:
        refactor_m = re.search(r"an agent is tasked with refactoring to (.+)", content)
        if refactor_m:
            return (
                f"Refactor the following Python code to {refactor_m.group(1).strip()}. "
                f"Return only the complete refactored code in a Python code block:\n\n"
                f"```python\n{code_block}\n```"
            ), {}
        return (
            f"Fix the bug in this Python code. "
            f"Return only the corrected code in a Python code block:\n\n"
            f"```python\n{code_block}\n```"
        ), {}

    gen_m = re.search(r"an agent is tasked with writing (.+)", content)
    if gen_m:
        return (
            f"Write {gen_m.group(1).strip()}. "
            f"Return only the implementation as a Python code block."
        ), {}

    feature_m = re.search(r"Feature: (.+)", content)
    return f"Complete: {feature_m.group(1).strip() if feature_m else 'the task'}", {}


def extract_metadata(feature_path: Path) -> dict:
    rel = feature_path.relative_to(PROJECT_ROOT / "features")
    parts = list(rel.parts)
    category = "/".join(parts[:-1])
    filename = parts[-1]
    task_id = f"{category}/{filename.replace('.feature', '')}"
    # Read complexity from the Complexity: tag in the file (authoritative)
    complexity = _feature_complexity(feature_path)
    if complexity not in ("T1", "T2", "T3", "T4", "Competition"):
        # Fallback: infer from filename prefix t1_/t2_/t3_/t4_
        tier_char = filename[1] if filename.startswith("t") and len(filename) > 1 else "1"
        complexity = {
            "1": "T1", "2": "T2", "3": "T3", "4": "T4",
        }.get(tier_char, "T1")
    return {"task_id": task_id, "category": category, "complexity": complexity}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_multi_turn(sandbox: TaskSandbox) -> tuple[float, bool]:
    passed, output = sandbox.run_pytest()
    if passed:
        return 1.0, True
    # Partial credit: count passed vs failed tests
    failed_m = re.search(r"(\d+) failed", output)
    passed_m = re.search(r"(\d+) passed", output)
    n_failed = int(failed_m.group(1)) if failed_m else 1
    n_passed = int(passed_m.group(1)) if passed_m else 0
    total = n_failed + n_passed
    score = round(n_passed / total, 3) if total > 0 else 0.0
    return score, False


def evaluate_single_turn(response: str, feature_path: Path) -> tuple[float, bool]:
    """Evaluate a single-turn response by executing code against acceptance criteria."""
    import subprocess as _sp
    import tempfile as _tmp

    content = feature_path.read_text()

    def _extract_code(text: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    def _run(code: str, script: str) -> bool:
        full = f"{code}\n\n{script}"
        with _tmp.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full)
            path = f.name
        try:
            r = _sp.run([sys.executable, path], capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        finally:
            os.unlink(path)

    code = _extract_code(response)
    criteria = [
        line.strip()[len(prefix):]
        for line in content.splitlines()
        for prefix in ("Then ", "And ")
        if line.strip().startswith(prefix)
        and not any(s in line for s in ("the complexity", "the category", "pytest exits"))
    ]

    if not criteria:
        return 0.5, False

    passed = 0
    for c in criteria:
        m = re.match(r'running (.+?) returns (.+)', c)
        if m:
            script = f"_r = {m.group(1)}\nassert _r == {m.group(2)}, repr(_r)"
            if _run(code, script):
                passed += 1
            continue

        if "code execution script passes:" in c or c == "the code execution script passes:":
            raw = _extract_docstring_block(content, "the code execution script passes:")
            if raw and _run(code, raw):
                passed += 1
            continue

        m = re.match(r'the output contains the text "(.+?)" exactly once', c)
        if m and code.count(m.group(1)) == 1:
            passed += 1
            continue

        m = re.match(r'the output does not contain the text "(.+?)"', c)
        if m and m.group(1) not in code:
            passed += 1
            continue

        # keyword heuristic fallback
        words = [w for w in c.split() if len(w) > 3]
        if words and any(w.lower() in response.lower() for w in words):
            passed += 1

    score = round(passed / len(criteria), 3)
    return score, score == 1.0


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #

def write_record(record: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    runs_file = RUNS_DIR / f"{date_str}.jsonl"
    with open(runs_file, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


# --------------------------------------------------------------------------- #
# Main run loop
# --------------------------------------------------------------------------- #

def run_one(
    feature_path: Path,
    strategy: str,
    dry_run: bool = False,
    completed: set[tuple[str, str]] | None = None,
) -> dict:
    meta = extract_metadata(feature_path)
    if completed and (meta["task_id"], strategy) in completed:
        print(f"  [skip] {meta['task_id']} × {strategy}")
        return {}

    model = MODEL_FOR_STRATEGY[strategy]
    prompt, files = build_prompt(feature_path)
    is_multi = bool(files)

    if dry_run:
        mode = "multi-turn" if is_multi else "single-turn"
        print(f"  [dry-run] {meta['task_id']} × {strategy} ({mode})")
        return {}

    mode_tag = "multi" if is_multi else "single"
    print(f"  {meta['task_id']} × {strategy} [{mode_tag}] ...", end=" ", flush=True)
    t0 = time.time()

    retry_waits = [60, 120, 300]
    for attempt in range(len(retry_waits) + 1):
        try:
            if is_multi:
                with TaskSandbox(files) as sb:
                    result = run_task(prompt, model=model, cwd=sb.path, timeout=300)
                    quality_score, passed = evaluate_multi_turn(sb)
            else:
                result = run_task(prompt, model=model, timeout=120)
                quality_score, passed = evaluate_single_turn(result["response"], feature_path)
            break
        except Exception as e:
            if attempt < len(retry_waits):
                wait = retry_waits[attempt]
                print(f"RETRY({attempt+1}) after {wait}s: {e}")
                time.sleep(wait)
            else:
                elapsed = round(time.time() - t0, 2)
                print(f"ERROR: {e}")
                return {}

    elapsed = round(time.time() - t0, 2)
    status = "PASS" if passed else f"FAIL(q={quality_score:.2f})"
    print(
        f"{status}  turns={result.get('num_turns', 1)}  "
        f"tokens={result['input_tokens']+result['output_tokens']}  "
        f"cost=${result['cost_usd']:.4f}  {elapsed}s"
    )

    record = {
        "task_id":               meta["task_id"],
        "strategy":              strategy,
        "complexity":            meta["complexity"],
        "category":              meta["category"],
        "mode":                  "multi-turn" if is_multi else "single-turn",
        "prompt_tokens_raw":     result["input_tokens"],
        "completion_tokens":     result["output_tokens"],
        "total_tokens":          result["input_tokens"] + result["output_tokens"],
        "total_cost_usd":        result["cost_usd"],
        "num_turns":             result.get("num_turns", 1),
        "tool_call_count":       result.get("tool_call_count", 0),
        "quality_score":         quality_score,
        "passed_criteria":       passed,
        "elapsed_s":             elapsed,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
    }
    write_record(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Synaxi predict benchmark runner")
    parser.add_argument("--strategies", default=",".join(ALL_STRATEGIES))
    parser.add_argument("--feature", default=None, help="Filter on task path substring")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-interleave", action="store_true",
                        help="Disable difficulty interleaving (run in file order)")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in MODEL_FOR_STRATEGY:
            print(f"Unknown strategy: {s}. Available: {', '.join(ALL_STRATEGIES)}")
            sys.exit(1)

    features_dir = PROJECT_ROOT / "features"
    all_features = sorted(features_dir.rglob("*.feature"))
    if args.feature:
        all_features = [f for f in all_features if args.feature in str(f)]
    if args.limit:
        all_features = all_features[: args.limit]

    if not all_features:
        print("No feature files found.")
        sys.exit(1)

    if not args.no_interleave:
        all_features = interleave_by_difficulty(all_features)

    completed = load_completed()
    print(f"Already completed: {len(completed)} (task, strategy) pairs — will skip these")

    total = len(all_features) * len(strategies)
    print(f"Running {len(all_features)} tasks × {len(strategies)} strategies = {total} runs")
    print(f"Strategies: {strategies}")
    print()

    records = []
    for fp in all_features:
        for strategy in strategies:
            r = run_one(fp, strategy, dry_run=args.dry_run, completed=completed)
            if r:
                records.append(r)
                if not args.dry_run:
                    time.sleep(5)

    if not args.dry_run and records:
        passed = sum(1 for r in records if r.get("passed_criteria"))
        total_cost = sum(r.get("total_cost_usd", 0) for r in records)
        avg_turns = sum(r.get("num_turns", 1) for r in records) / len(records)
        print(f"\n{'─'*60}")
        print(f"Completed: {len(records)} runs")
        print(f"Passed:    {passed}/{len(records)}")
        print(f"Avg turns: {avg_turns:.1f}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Records: {RUNS_DIR}")


if __name__ == "__main__":
    main()
