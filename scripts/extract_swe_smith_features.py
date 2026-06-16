#!/usr/bin/env python3
"""
Clone all swe_smith repos and extract tree-sitter code features.

For each unique instance_id in the swe_smith parquet files:
  - Parse the patch to get changed .py file paths
  - Run tree-sitter on those files in the cloned repo
  - Store features in data/code_features.json (merged with existing entries)

Also adds near-zero features for CC/HumanEval/MBPP tasks (single-file,
low-complexity repos) so the model sees explicit signal vs complex repos.

Usage:
    python scripts/extract_swe_smith_features.py
    python scripts/extract_swe_smith_features.py --clone-dir /tmp/swe_smith_repos --jobs 8
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
CLONE_BASE = Path("/tmp/swe_smith_repos")

NULL_FEATURES = {
    "loc": 0, "functions": 0, "classes": 0,
    "branches": 0, "try_blocks": 0, "n_files": 0,
    "avg_loc": 0.0, "branch_density": 0.0, "has_code_features": 0,
}


def file_stats(path: Path) -> dict | None:
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        sys.exit("pip install tree-sitter tree-sitter-python")

    PY = Language(tspython.language())
    parser = Parser(PY)
    try:
        code = path.read_bytes()
    except OSError:
        return None

    tree = parser.parse(code)
    counts = {t: 0 for t in ("function_definition", "class_definition",
                              "if_statement", "for_statement",
                              "while_statement", "try_statement")}

    def walk(node):
        if node.type in counts:
            counts[node.type] += 1
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    loc = len(code.decode("utf-8", errors="replace").splitlines())
    return {
        "loc": loc,
        "functions": counts["function_definition"],
        "classes": counts["class_definition"],
        "branches": counts["if_statement"] + counts["for_statement"] + counts["while_statement"],
        "try_blocks": counts["try_statement"],
    }


def clone_repo(owner_repo: str, clone_base: Path) -> Path | None:
    owner, repo = owner_repo.split("/", 1)
    dest = clone_base / f"{owner}__{repo}"
    if dest.exists():
        return dest
    url = f"https://github.com/{owner_repo}.git"
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--quiet", url, str(dest)],
        capture_output=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  FAILED clone {owner_repo}: {result.stderr.decode()[:100]}", flush=True)
        return None
    return dest


def parse_changed_py_files(patch: str) -> list[str]:
    return [ln[6:] for ln in (patch or "").splitlines()
            if ln.startswith("+++ b/") and ln.endswith(".py")]


def _resolve_file(fp: str, pkg_to_clone: dict[str, Path]) -> Path | None:
    """Resolve a patch file path to an absolute path in a cloned repo."""
    parts = fp.split("/")
    top = parts[0]

    # Direct match: top-level package dir or single-file module (e.g. bottle.py)
    clone = pkg_to_clone.get(top)
    if clone:
        candidate = clone / fp
        if candidate.exists():
            return candidate

    # src/ layout: strip leading 'src/' and try again
    if top == "src" and len(parts) > 1:
        inner = "/".join(parts[1:])
        inner_top = parts[1]
        clone = pkg_to_clone.get(inner_top)
        if clone:
            # file might be at clone/src/pkg/... or clone/pkg/...
            for prefix in [clone / fp, clone / inner]:
                if prefix.exists():
                    return prefix

    return None


def instance_features(patch: str, pkg_to_clone: dict[str, Path]) -> dict:
    """
    pkg_to_clone: maps top-level directory/module name → clone path.
    Handles both regular package layouts and src/ layouts.
    """
    changed = parse_changed_py_files(patch)
    if not changed:
        return {**NULL_FEATURES}

    totals = {"loc": 0, "functions": 0, "classes": 0, "branches": 0, "try_blocks": 0}
    n = 0
    for fp in changed:
        resolved = _resolve_file(fp, pkg_to_clone)
        if not resolved:
            continue
        s = file_stats(resolved)
        if s:
            for k in totals:
                totals[k] += s[k]
            n += 1

    if n == 0:
        return {**NULL_FEATURES}

    return {
        **totals,
        "n_files": n,
        "avg_loc": round(totals["loc"] / n, 1),
        "branch_density": round(totals["branches"] / max(totals["loc"], 1), 4),
        "has_code_features": 1,
    }


def add_cc_features(features: dict) -> int:
    """
    Add features for CC/HumanEval/MBPP tasks, keyed by stem (filename without
    extension) to match what build_dataset extracts via task_id.split('/')[-1].
    """
    added = 0
    for feat_file in sorted((PROJECT_ROOT / "features").rglob("*.feature")):
        # Key must match build_dataset: task_id.split('/')[-1].replace('.feature','')
        instance_id = feat_file.stem
        if instance_id in features:
            continue
        stats = file_stats(feat_file)
        if stats:
            features[instance_id] = {
                **stats,
                "n_files": 1,
                "avg_loc": float(stats["loc"]),
                "branch_density": round(stats["branches"] / max(stats["loc"], 1), 4),
                "has_code_features": 1,
            }
        else:
            features[instance_id] = {**NULL_FEATURES}
        added += 1
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-dir", default=str(CLONE_BASE), type=Path)
    parser.add_argument("--jobs", default=8, type=int, help="parallel clone workers")
    parser.add_argument("--out", default="data/code_features.json", type=Path)
    args = parser.parse_args()

    clone_base: Path = PROJECT_ROOT / args.clone_dir if not args.clone_dir.is_absolute() else args.clone_dir
    out_path: Path = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    clone_base.mkdir(parents=True, exist_ok=True)

    # Load existing features
    features: dict = {}
    if out_path.exists():
        features = json.loads(out_path.read_text())
    print(f"Loaded {len(features)} existing entries from {out_path}", flush=True)

    # Load swe_smith parquets
    parquet_files = sorted((PROJECT_ROOT / "data/swe_smith/data").glob("ticks-*.parquet"))
    print(f"Loading {len(parquet_files)} parquet files...", flush=True)
    dfs = [pd.read_parquet(f, columns=["instance_id", "patch"]) for f in parquet_files]
    df = pd.concat(dfs).drop_duplicates("instance_id")
    print(f"  {len(df)} unique instances", flush=True)

    # Extract unique repos
    repos: dict[str, str] = {}  # owner__repo → owner/repo
    for iid in df["instance_id"]:
        m = re.match(r"^([^.]+)__([^.]+)\.", iid)
        if m:
            key = f"{m.group(1)}__{m.group(2)}"
            repos[key] = f"{m.group(1)}/{m.group(2)}"

    print(f"\nCloning {len(repos)} repos to {clone_base} (jobs={args.jobs})...", flush=True)
    clones: dict[str, Path] = {}  # owner__repo → clone path

    def _clone(item):
        key, owner_repo = item
        path = clone_repo(owner_repo, clone_base)
        return key, path

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_clone, item): item[0] for item in repos.items()}
        done = 0
        for fut in as_completed(futs):
            key, path = fut.result()
            done += 1
            if path:
                clones[key] = path
            if done % 20 == 0 or done == len(repos):
                print(f"  {done}/{len(repos)} repos cloned ({len(clones)} ok)", flush=True)

    print(f"\nCloned {len(clones)}/{len(repos)} repos", flush=True)

    # Build top-level package/module → clone path map
    pkg_to_clone: dict[str, Path] = {}
    for clone in clones.values():
        for child in clone.iterdir():
            if child.name.startswith("."):
                continue
            if child.is_dir():
                pkg_to_clone[child.name] = clone
            elif child.suffix == ".py":
                # single-file modules like bottle.py → key is "bottle.py"
                pkg_to_clone[child.name] = clone
    print(f"Package map: {len(pkg_to_clone)} top-level entries across {len(clones)} repos\n", flush=True)

    # Extract features for each instance
    new_count = skip_count = fail_count = 0
    patch_by_iid = dict(zip(df["instance_id"], df["patch"]))

    for i, (iid, patch) in enumerate(patch_by_iid.items()):
        if iid in features and features[iid].get("has_code_features"):
            skip_count += 1
            continue

        feat = instance_features(patch or "", pkg_to_clone)
        features[iid] = feat
        if feat["has_code_features"]:
            new_count += 1
        else:
            fail_count += 1

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(patch_by_iid)}  new={new_count}  skip={skip_count}  fail={fail_count}", flush=True)

    print(f"\nSWE-smith done: new={new_count}  skipped={skip_count}  failed/null={fail_count}")

    # Add CC/HumanEval/MBPP features
    cc_added = add_cc_features(features)
    print(f"CC/HumanEval/MBPP features added: {cc_added}")

    # Write merged output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(features, separators=(",", ":")))

    total = len(features)
    has = sum(1 for f in features.values() if f.get("has_code_features"))
    print(f"\nWrote {total} entries → {out_path}  ({has} with features, {total-has} null)")


if __name__ == "__main__":
    main()
