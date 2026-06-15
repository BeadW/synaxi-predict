#!/usr/bin/env python3
"""
Import harder coding benchmarks into the multi-turn .feature format.

Supports:
  APPS       (codeparrot/apps)         -- competition + interview tiers
  LiveCodeBench (livecodebench/code_generation_lite) -- medium + hard tiers

Usage:
    python scripts/import_benchmarks.py --dataset apps --difficulty competition
    python scripts/import_benchmarks.py --dataset apps --difficulty interview --limit 100
    python scripts/import_benchmarks.py --dataset lcb --difficulty hard
    python scripts/import_benchmarks.py --dataset lcb --difficulty medium --limit 100
    python scripts/import_benchmarks.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"
OUT_DIR = FEATURES_DIR / "code" / "generation"


# --------------------------------------------------------------------------- #
# APPS
# --------------------------------------------------------------------------- #

APPS_COMPLEXITY = {
    "introductory": "T1",
    "interview":    "T2",
    "competition":  "T3",
}


def _apps_test_file(inputs: list[str], outputs: list[str], limit: int = 5) -> str:
    """Build a pytest file that runs solution.py as a subprocess per test case."""
    cases = list(zip(inputs, outputs))[:limit]
    cases_repr = json.dumps(cases, ensure_ascii=False, indent=4)
    return f'''\
import subprocess
import sys
import pytest

TEST_CASES = {cases_repr}


def test_apps():
    for i, (inp, expected) in enumerate(TEST_CASES):
        r = subprocess.run(
            [sys.executable, "solution.py"],
            input=inp,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 0, f"Case {{i}}: runtime error\\n{{r.stderr[:400]}}"
        assert r.stdout.strip() == str(expected).strip(), (
            f"Case {{i}}: expected {{expected!r}}, got {{r.stdout!r}}"
        )
'''


def import_apps(difficulty: str = "competition", limit: int | None = None) -> int:
    from datasets import load_dataset

    ds = load_dataset("json",
                      data_files={"test": "hf://datasets/codeparrot/apps/test.jsonl"},
                      split="test")
    rows = [r for r in ds if r["difficulty"] == difficulty]
    if limit:
        rows = rows[:limit]

    complexity = APPS_COMPLEXITY.get(difficulty, "T3")
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for row in rows:
        pid = row.get("problem_id", row.get("id", written))
        slug = f"apps_{difficulty[:4]}{pid:04d}"
        feature_path = out_dir / f"{slug}.feature"
        if feature_path.exists():
            continue

        # Parse test cases
        try:
            io = json.loads(row["input_output"]) if row["input_output"] else {}
        except Exception:
            continue
        inputs  = io.get("inputs", [])
        outputs = io.get("outputs", [])
        if not inputs or not outputs:
            continue

        # Clean up question text — strip excessive whitespace
        question = row["question"].strip().replace('"""', "'''")
        # Keep to a reasonable length
        if len(question) > 2000:
            question = question[:2000] + "\n...(truncated)"

        test_src = _apps_test_file(inputs, outputs, limit=5)
        test_safe = test_src.replace('"""', "'''")

        stub = "import sys\n\n# Read from stdin, write answer to stdout\n"

        feature = f'''\
Feature: APPS/{difficulty}/{pid} — competitive programming problem
  Complexity: {complexity}
  Category: code/generation
  Mode: multi-turn

  Background:
    Given the following project files:
      """
      === problem.txt ===
{textwrap.indent(question, "      ")}

      === solution.py ===
{textwrap.indent(stub, "      ")}
      === test_solution.py ===
{textwrap.indent(test_safe, "      ")}
      """

  Scenario: Agent solves the APPS problem so all tests pass
    Given an agent is tasked with reading problem.txt and implementing solution.py to read from stdin and write the correct answer to stdout so all pytest tests pass
    When the agent completes the task in the sandbox
    Then pytest exits with code 0
'''
        feature_path.write_text(feature)
        written += 1

    return written


# --------------------------------------------------------------------------- #
# LiveCodeBench
# --------------------------------------------------------------------------- #

LCB_COMPLEXITY = {
    "easy":   "T1",
    "medium": "T2",
    "hard":   "T3",
    "Easy":   "T1",
    "Medium": "T2",
    "Hard":   "T3",
}



def _lcb_test_file_stdio(test_cases: list) -> str:
    cases = [(tc["input"], tc["output"]) for tc in test_cases[:5]
             if tc.get("input") and tc.get("output")]
    cases_repr = json.dumps(cases, ensure_ascii=False, indent=4)
    return f'''\
import subprocess
import sys

TEST_CASES = {cases_repr}


def test_lcb():
    for i, (inp, expected) in enumerate(TEST_CASES):
        r = subprocess.run(
            [sys.executable, "solution.py"],
            input=inp,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 0, f"Case {{i}}: runtime error\\n{{r.stderr[:400]}}"
        assert r.stdout.strip() == str(expected).strip(), (
            f"Case {{i}}: expected {{expected!r}}, got {{r.stdout!r}}"
        )
'''


def import_lcb(difficulty: str = "hard", limit: int | None = None) -> int:
    import json as _json
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        "livecodebench/code_generation_lite",
        "test.jsonl",
        repo_type="dataset",
    )

    diff_lower = difficulty.lower()
    rows = []
    with open(local) as f:
        for line in f:
            r = _json.loads(line)
            if r.get("difficulty", "").lower() != diff_lower:
                continue
            # Pre-filter: must have at least one stdin test case
            raw = r.get("public_test_cases", "[]")
            if isinstance(raw, str):
                try:
                    tc = _json.loads(raw)
                except Exception:
                    continue
            else:
                tc = list(raw)
            if any(t.get("testtype") == "stdin" for t in tc):
                rows.append(r)
    if limit:
        rows = rows[:limit]

    complexity = LCB_COMPLEXITY.get(difficulty, "T2")
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for row in rows:
        qid   = str(row.get("question_id", written))
        title = re.sub(r"[^a-z0-9_]", "_", row.get("question_title", f"q{qid}").lower())[:40]
        slug  = f"lcb_{diff_lower[:3]}_{qid[:8].replace('-','_')}_{title}"
        feature_path = out_dir / f"{slug}.feature"
        if feature_path.exists():
            continue

        platform = row.get("platform", "unknown")
        content  = (row.get("question_content") or "").strip()

        # Parse test cases — may be list or JSON string
        raw = row.get("public_test_cases", "[]")
        if isinstance(raw, str):
            try:
                test_cases = _json.loads(raw)
            except Exception:
                test_cases = []
        else:
            test_cases = list(raw)

        # Only handle stdin-style for now (Codeforces / AtCoder)
        # Functional (LeetCode) test cases need a different harness
        stdin_cases = [tc for tc in test_cases if tc.get("testtype") == "stdin"]
        if not stdin_cases:
            continue

        solution_stub = "import sys\n\n# Read from stdin, write answer to stdout\n"
        test_src = _lcb_test_file_stdio(stdin_cases)

        content_safe = content.replace('"""', "'''")[:2000]
        test_safe    = test_src.replace('"""', "'''")
        stub_safe    = solution_stub.replace('"""', "'''")

        feature = f'''\
Feature: LiveCodeBench/{platform}/{difficulty} — {row.get("question_title","problem")}
  Complexity: {complexity}
  Category: code/generation
  Mode: multi-turn

  Background:
    Given the following project files:
      """
      === problem.txt ===
{textwrap.indent(content_safe, "      ")}

      === solution.py ===
{textwrap.indent(stub_safe, "      ")}
      === test_solution.py ===
{textwrap.indent(test_safe, "      ")}
      """

  Scenario: Agent solves the LiveCodeBench problem so all tests pass
    Given an agent is tasked with reading problem.txt and implementing solution.py to read from stdin and write the correct answer to stdout so all pytest tests pass
    When the agent completes the task in the sandbox
    Then pytest exits with code 0
'''
        feature_path.write_text(feature)
        written += 1

    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["apps", "lcb", "all"], default="all")
    parser.add_argument("--difficulty", default=None,
                        help="For apps: introductory/interview/competition. "
                             "For lcb: easy/medium/hard. Default: hard tiers only.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.dataset in ("apps", "all"):
        diffs = [args.difficulty] if args.difficulty else ["competition", "interview"]
        for d in diffs:
            n = import_apps(d, limit=args.limit)
            print(f"APPS {d}: wrote {n} new feature files")

    if args.dataset in ("lcb", "all"):
        diffs = [args.difficulty] if args.difficulty else ["hard", "medium"]
        for d in diffs:
            n = import_lcb(d, limit=args.limit)
            print(f"LiveCodeBench {d}: wrote {n} new feature files")


if __name__ == "__main__":
    main()
