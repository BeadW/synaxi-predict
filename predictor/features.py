"""
Feature extraction for the token/cost predictor.

All features must be computable from the task prompt text alone at inference
time — before the task is run. File-based features are used during training
only as supplemental signals where available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

STRATEGY_IDS = {"single-haiku": 0, "single-sonnet": 1, "single-opus": 2}

# Keywords that signal algorithmic/implementation complexity.
# Grouped roughly by tier — presence of tier-N terms adds to complexity score.
_TIER4_TERMS = {
    "b-tree", "btree", "b_tree",
    "lsm", "lsm-tree", "sstable", "memtable", "compaction",
    "regex engine", "nfa", "automaton",
    "pratt parser", "precedence climbing", "recursive descent",
    "consistent hashing", "virtual node", "hash ring",
    "event sourcing", "cqrs", "optimistic concurrency",
    "skip list", "skiplist",
    "dag scheduler", "topological sort", "strongly connected",
    "concurrent", "thread-safe", "lock-free", "atomic",
}

_TIER3_TERMS = {
    "dijkstra", "bellman-ford", "shortest path", "minimum spanning",
    "async", "asyncio", "coroutine", "rate limiter", "backoff",
    "parser", "lexer", "tokenizer", "ast",
    "cache eviction", "lru", "lfu", "ttl",
    "bloom filter", "trie", "segment tree", "fenwick",
    "concurrency", "race condition", "deadlock", "mutex",
    "binary search", "merge sort", "quicksort",
    "graph", "bfs", "dfs", "cycle detection",
}

_TIER2_TERMS = {
    "recursion", "recursive", "base case",
    "linked list", "stack", "queue", "deque",
    "regex", "regular expression", "pattern matching",
    "hash table", "dict", "map", "set",
    "sorting", "searching",
    "fizzbuzz", "palindrome", "anagram", "permutation",
    "decorator", "context manager", "generator",
    "class", "inheritance", "polymorphism",
}

_DEBUG_VERBS    = {"fix", "debug", "bug", "broken", "repair", "correct", "patch", "failing", "error"}
_GENERATE_VERBS = {"write", "implement", "create", "generate", "build", "develop", "design"}
_REFACTOR_VERBS = {"refactor", "extract", "rename", "simplify", "restructure", "split", "convert"}

# Complexity tier indicators found in feature-file headers
_COMPLEXITY_TIER = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "Competition": 4}


def _complexity_score(text: str) -> int:
    """
    Estimate task complexity 0-4 from prompt text via keyword matching.
    Higher = harder.
    """
    lower = text.lower()
    if any(t in lower for t in _TIER4_TERMS):
        return 4
    if any(t in lower for t in _TIER3_TERMS):
        return 3
    if any(t in lower for t in _TIER2_TERMS):
        return 2
    return 1


def extract_features(
    prompt: str,
    strategy: str,
    files: Dict[str, str] | None = None,
) -> Dict[str, float]:
    """
    Extract a flat feature dict from a task prompt.

    Parameters
    ----------
    prompt   : The full agent prompt string.
    strategy : Strategy ID e.g. "single-sonnet".
    files    : Optional sandbox files (present during training, absent at inference).

    Returns
    -------
    dict of numeric features (all float-compatible).
    """
    files = files or {}
    lower = prompt.lower()

    # ── Prompt complexity ──────────────────────────────────────────────────
    prompt_chars  = len(prompt)
    prompt_words  = len(prompt.split())
    prompt_lines  = prompt.count("\n") + 1
    prompt_tokens = prompt_chars // 4   # rough BPE estimate

    complexity_score = _complexity_score(prompt)

    # Requirement count heuristic: sentences / clauses containing "must", "should",
    # "implement", "support", commas in task descriptions
    requirements = len(re.findall(
        r"\b(must|should|implement|support|handle|ensure|return|raise|pass)\b",
        lower,
    ))

    # Task type
    is_debug    = int(any(v in lower for v in _DEBUG_VERBS))
    is_generate = int(any(v in lower for v in _GENERATE_VERBS))
    is_refactor = int(any(v in lower for v in _REFACTOR_VERBS))

    # ── Sandbox file features (training-time enrichment) ──────────────────
    file_count       = len(files)
    total_src_lines  = 0
    total_src_chars  = 0
    test_fn_count    = 0
    src_file_count   = 0

    all_src = ""
    for fname, content in files.items():
        lines = content.splitlines()
        total_src_lines += len(lines)
        total_src_chars += len(content)
        name = Path(fname).name
        if re.match(r"test_.*\.py$", name):
            test_fn_count += content.count("def test_")
        elif fname.endswith(".py"):
            src_file_count += 1
            all_src += content + "\n"

    fn_count     = all_src.count("def ")
    class_count  = all_src.count("class ")
    import_count = all_src.count("import ")
    nested_lines = sum(1 for ln in all_src.splitlines() if ln.startswith("        "))

    # If we have files, upgrade complexity_score from file evidence
    if test_fn_count > 15:
        complexity_score = max(complexity_score, 3)
    if test_fn_count > 25:
        complexity_score = max(complexity_score, 4)

    # ── Strategy ──────────────────────────────────────────────────────────
    strategy_id = STRATEGY_IDS.get(strategy, 1)

    return {
        # Prompt-only features (available at inference time)
        "prompt_chars":       float(prompt_chars),
        "prompt_words":       float(prompt_words),
        "prompt_lines":       float(prompt_lines),
        "prompt_tokens":      float(prompt_tokens),
        "complexity_score":   float(complexity_score),
        "requirements":       float(requirements),
        "is_debug":           float(is_debug),
        "is_generate":        float(is_generate),
        "is_refactor":        float(is_refactor),
        "strategy_id":        float(strategy_id),
        # File-based features (0 at inference time, populated during training)
        "file_count":         float(file_count),
        "src_file_count":     float(src_file_count),
        "test_fn_count":      float(test_fn_count),
        "total_src_lines":    float(total_src_lines),
        "total_src_chars":    float(total_src_chars),
        "fn_count":           float(fn_count),
        "class_count":        float(class_count),
        "import_count":       float(import_count),
        "nested_lines":       float(nested_lines),
    }
