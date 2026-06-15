#!/usr/bin/env python3
"""
Convert HumanEval and MBPP problems into our multi-turn .feature format.

Each problem becomes a sandbox task where:
  - The source stub (function signature + docstring) is in solution.py
  - The pytest test file wraps the canonical check() function
  - The agent must implement the function so all tests pass

Usage:
    python scripts/import_humaneval.py --dataset humaneval --limit 50
    python scripts/import_humaneval.py --dataset mbpp --limit 100
    python scripts/import_humaneval.py --dataset humaneval  # all 164
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"


# --------------------------------------------------------------------------- #
# HumanEval
# --------------------------------------------------------------------------- #

def _humaneval_complexity(task_num: int, prompt: str) -> str:
    """Rough complexity tier based on prompt length and task number."""
    prompt_len = len(prompt)
    if prompt_len < 400:
        return "T1"
    if prompt_len < 800:
        return "T2"
    return "T3"


def _humaneval_test_file(entry_point: str, test_src: str) -> str:
    """
    Convert HumanEval's check(candidate) format into pytest functions.
    """
    # Strip METADATA block
    test_src = re.sub(r"METADATA\s*=\s*\{.*?\}\n+", "", test_src, flags=re.DOTALL)

    # Extract the body of check(candidate)
    m = re.search(r"def check\(candidate\):\n(.*?)(?=\ndef |\Z)", test_src, re.DOTALL)
    body = textwrap.dedent(m.group(1)) if m else "    pass\n"

    # Replace candidate( with entry_point(
    body = body.replace("candidate(", f"{entry_point}(")

    pytest_lines = [
        f"from solution import {entry_point}",
        "",
        "",
        "def test_humaneval():",
    ]
    for line in body.splitlines():
        pytest_lines.append("    " + line if line.strip() else "")

    return "\n".join(pytest_lines) + "\n"


def _indent_solution(prompt: str) -> str:
    """Return the prompt (stub) with a pass body so it runs without error."""
    # The prompt ends with the function signature + docstring, no body
    if not prompt.rstrip().endswith('"""') and not prompt.rstrip().endswith("'''"):
        # Already has a partial body — leave it
        return prompt
    return prompt + "    pass\n"


def import_humaneval(limit: int | None = None) -> int:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    out_dir = FEATURES_DIR / "code" / "generation"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for row in ds:
        task_num = int(row["task_id"].split("/")[1])
        entry_point = row["entry_point"]
        slug = f"he{task_num:03d}_{entry_point}"
        feature_path = out_dir / f"{slug}.feature"

        if feature_path.exists():
            continue  # skip already imported

        complexity = _humaneval_complexity(task_num, row["prompt"])
        stub = _indent_solution(row["prompt"])
        test_src = _humaneval_test_file(entry_point, row["test"])

        # Embed both files in the feature, escaping any triple-quotes
        stub_safe = stub.replace('"""', "'''")
        test_safe = test_src.replace('"""', "'''")

        feature = f'''Feature: HumanEval/{task_num} — implement {entry_point}
  Complexity: {complexity}
  Category: code/generation
  Mode: multi-turn

  Background:
    Given the following project files:
      """
      === solution.py ===
{textwrap.indent(stub_safe, "      ")}
      === test_solution.py ===
{textwrap.indent(test_safe, "      ")}
      """

  Scenario: Agent implements {entry_point} so all tests pass
    Given an agent is tasked with implementing the {entry_point} function in solution.py so all pytest tests pass
    When the agent completes the task in the sandbox
    Then pytest exits with code 0
'''
        feature_path.write_text(feature)
        written += 1

    return written


# --------------------------------------------------------------------------- #
# MBPP
# --------------------------------------------------------------------------- #

def import_mbpp(limit: int | None = None) -> int:
    from datasets import load_dataset
    # MBPP sanitized split is cleaner
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    out_dir = FEATURES_DIR / "code" / "generation"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for row in ds:
        task_id = row.get("task_id", 0)
        code = row.get("code", "")
        text = row.get("text", "")
        test_list = row.get("test_list", [])

        if not code or not test_list:
            continue

        # Extract function name from canonical solution
        m = re.search(r"def (\w+)\(", code)
        if not m:
            continue
        entry_point = m.group(1)
        slug = f"mbpp{task_id:03d}_{entry_point}"
        feature_path = out_dir / f"{slug}.feature"

        if feature_path.exists():
            continue

        # Build stub from just the signature line
        sig_m = re.match(r"(def \w+\([^)]*\)[^:]*:)", code)
        stub = sig_m.group(1) + "\n    pass\n" if sig_m else f"def {entry_point}():\n    pass\n"

        # Convert assert-based tests to pytest
        test_lines = [f"from solution import {entry_point}", "", "", "def test_mbpp():"]
        for t in test_list:
            test_lines.append(f"    {t.strip()}")
        test_src = "\n".join(test_lines) + "\n"

        # Complexity: simple heuristic from code length
        complexity = "T1" if len(code) < 200 else ("T2" if len(code) < 500 else "T3")

        stub_safe = stub.replace('"""', "'''")
        test_safe = test_src.replace('"""', "'''")

        feature = f'''Feature: MBPP/{task_id} — {text[:60].strip()}
  Complexity: {complexity}
  Category: code/generation
  Mode: multi-turn

  Background:
    Given the following project files:
      """
      === solution.py ===
{textwrap.indent(stub_safe, "      ")}
      === test_solution.py ===
{textwrap.indent(test_safe, "      ")}
      """

  Scenario: Agent implements {entry_point} so all pytest tests pass
    Given an agent is tasked with implementing the {entry_point} function in solution.py so all pytest tests pass
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
    parser.add_argument("--dataset", choices=["humaneval", "mbpp", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Max problems per dataset")
    args = parser.parse_args()

    if args.dataset in ("humaneval", "both"):
        n = import_humaneval(args.limit)
        print(f"HumanEval: wrote {n} new feature files")

    if args.dataset in ("mbpp", "both"):
        n = import_mbpp(args.limit)
        print(f"MBPP:      wrote {n} new feature files")


if __name__ == "__main__":
    main()
