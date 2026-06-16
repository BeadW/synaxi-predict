#!/usr/bin/env python3
"""
Extract tree-sitter code features for every SWE-bench instance.

For each instance we know which files were changed (from the patch).
We analyse those files in the cloned repo and produce per-instance
numeric features that complement the task-text TF-IDF features.

Features (all summed across changed .py files):
  loc            total lines of code
  functions      function/method definitions
  classes        class definitions
  branches       if + for + while statements
  try_blocks     try/except blocks
  n_files        number of changed .py files

Aggregate ratios (per file averages):
  avg_loc        loc / n_files
  branch_density branches / max(loc, 1)

Usage:
    python scripts/extract_code_features.py
    python scripts/extract_code_features.py --out data/code_features.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Map repo slug → local clone path
REPO_CLONES: dict[str, Path] = {
    "django/django":               Path("/tmp/django"),
    "sympy/sympy":                 Path("/tmp/sympy-sympy"),
    "sphinx-doc/sphinx":           Path("/tmp/sphinx-doc-sphinx"),
    "matplotlib/matplotlib":       Path("/tmp/matplotlib-matplotlib"),
    "scikit-learn/scikit-learn":   Path("/tmp/scikit-learn-scikit-learn"),
    "pytest-dev/pytest":           Path("/tmp/pytest-dev-pytest"),
    "astropy/astropy":             Path("/tmp/astropy-astropy"),
    "pydata/xarray":               Path("/tmp/pydata-xarray"),
    "pylint-dev/pylint":           Path("/tmp/pylint-dev-pylint"),
    "psf/requests":                Path("/tmp/psf-requests"),
    "mwaskom/seaborn":             Path("/tmp/mwaskom-seaborn"),
    "pallets/flask":               Path("/tmp/pallets-flask"),
}

NULL_FEATURES = {
    "loc": 0, "functions": 0, "classes": 0,
    "branches": 0, "try_blocks": 0, "n_files": 0,
    "avg_loc": 0.0, "branch_density": 0.0,
    "has_code_features": 0,
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "is", "are", "was", "be", "that", "this", "it", "as", "from",
    "by", "not", "but", "when", "if", "fix", "bug", "error", "issue",
    "use", "using", "used", "add", "update", "change", "make", "get",
    "set", "file", "files", "code", "test", "tests", "should", "does",
    "do", "can", "will", "has", "have", "been", "into", "all", "also",
}


def _keywords_from_text(text: str) -> list[str]:
    """Extract meaningful tokens from a task description."""
    tokens = re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _find_relevant_files(repo: Path, keywords: list[str], max_files: int = 30) -> list[Path]:
    """
    Find .py files whose path contains any keyword.
    Falls back to all non-test source files if nothing matches.
    """
    all_py = [
        p for p in repo.rglob("*.py")
        if "__pycache__" not in p.parts
        and ".git" not in p.parts
    ]

    if keywords:
        matched = []
        for p in all_py:
            path_str = p.as_posix().lower()
            if any(kw in path_str for kw in keywords):
                matched.append(p)
        if matched:
            return matched[:max_files]

    # Fallback: source files only (exclude tests/docs/examples), capped
    source = [
        p for p in all_py
        if not any(seg in ("test", "tests", "docs", "examples", "doc") for seg in p.parts)
    ]
    return source[:max_files]


def repo_aggregate_stats(repo: Path, task_description: str = "") -> dict:
    """
    Analyse a live repo with tree-sitter and return code features.

    Uses keyword matching against the task description to find the most
    relevant files before measuring them.  Falls back to top-level source
    files when no keywords match.
    """
    keywords = _keywords_from_text(task_description) if task_description else []
    files    = _find_relevant_files(repo, keywords)

    if not files:
        return {**NULL_FEATURES}

    totals = {"loc": 0, "functions": 0, "classes": 0, "branches": 0, "try_blocks": 0}
    n = 0
    for p in files:
        s = file_stats(p)
        if s:
            for k in totals:
                totals[k] += s[k]
            n += 1

    if n == 0:
        return {**NULL_FEATURES}

    avg_loc = totals["loc"] / n
    branch_density = totals["branches"] / max(totals["loc"], 1)
    return {
        **totals,
        "n_files":        n,
        "avg_loc":        round(avg_loc, 1),
        "branch_density": round(branch_density, 4),
        "has_code_features": 1,
    }


def _parse_changed_files(patch: str) -> list[str]:
    return [ln[6:] for ln in patch.splitlines() if ln.startswith("+++ b/") and ln.endswith(".py")]


def file_stats(path: Path) -> dict | None:
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        raise RuntimeError("pip install tree-sitter tree-sitter-python")

    _PY = Language(tspython.language())
    _PARSER = Parser(_PY)

    try:
        code = path.read_bytes()
    except OSError:
        return None

    tree = _PARSER.parse(code)
    root = tree.root_node

    counts = {
        "function_definition": 0,
        "class_definition": 0,
        "if_statement": 0,
        "for_statement": 0,
        "while_statement": 0,
        "try_statement": 0,
    }

    def walk(node: object) -> None:
        if node.type in counts:
            counts[node.type] += 1
        for child in node.children:
            walk(child)

    walk(root)
    loc = len(code.decode("utf-8", errors="replace").splitlines())
    return {
        "loc": loc,
        "functions": counts["function_definition"],
        "classes": counts["class_definition"],
        "branches": counts["if_statement"] + counts["for_statement"] + counts["while_statement"],
        "try_blocks": counts["try_statement"],
    }


def instance_features(repo: str, patch: str) -> dict:
    clone = REPO_CLONES.get(repo)
    if clone is None or not clone.exists():
        return {**NULL_FEATURES}

    changed = _parse_changed_files(patch)
    if not changed:
        return {**NULL_FEATURES}

    totals = {"loc": 0, "functions": 0, "classes": 0, "branches": 0, "try_blocks": 0}
    n_files = 0
    for fpath in changed:
        stats = file_stats(clone / fpath)
        if stats:
            for k in totals:
                totals[k] += stats[k]
            n_files += 1

    if n_files == 0:
        return {**NULL_FEATURES}

    avg_loc = totals["loc"] / n_files
    branch_density = totals["branches"] / max(totals["loc"], 1)

    return {
        **totals,
        "n_files": n_files,
        "avg_loc": round(avg_loc, 1),
        "branch_density": round(branch_density, 4),
        "has_code_features": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/code_features.json", type=Path)
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out

    from datasets import load_dataset

    features: dict[str, dict] = {}

    for benchmark, split in [
        ("princeton-nlp/SWE-bench_Lite", "test"),
        ("princeton-nlp/SWE-bench_Verified", "test"),
    ]:
        print(f"Processing {benchmark}...")
        ds = load_dataset(benchmark, split=split)
        for row in ds:
            iid  = row["instance_id"]
            repo = row["repo"]
            patch = row["patch"]
            if iid not in features:
                features[iid] = instance_features(repo, patch)

        covered = sum(1 for f in features.values() if f["has_code_features"])
        print(f"  {covered}/{len(features)} instances with code features so far")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(features, indent=2))
    print(f"\nWrote {len(features)} entries → {out_path}")

    # Quick summary
    import numpy as np
    has = [f for f in features.values() if f["has_code_features"]]
    print(f"Coverage: {len(has)}/{len(features)} ({100*len(has)/len(features):.0f}%)")
    for col in ["loc", "functions", "branches"]:
        vals = np.array([f[col] for f in has])
        print(f"  {col}: mean={vals.mean():.0f}  p90={np.percentile(vals,90):.0f}  max={vals.max():.0f}")


if __name__ == "__main__":
    main()
