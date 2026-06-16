"""
Predict cost, turns, and pass rate for a coding task across all known models.

Cost is NOT predicted by the model — it is calculated from predicted token
counts (prompt_tokens, completion_tokens, num_turns) using the exact pricing
formula for each model. This keeps the model focused on usage patterns while
cost stays accurate as prices change.

Usage:
    python -m predictor.predict "Fix the failing Django migration"
    python -m predictor.predict "Implement a B-tree" --models claude-3-5-haiku-20241022
    echo "task description" | python -m predictor.predict -
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# MLP forward passes on sparse TF-IDF inputs can produce intermediate overflow
# in sklearn's matmul — weights are still valid (checked at load time).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "predictor_v1.pkl"
PREDICTIONS_FILE = PROJECT_ROOT / "data" / "predictions_live.jsonl"

# Pricing per token: (input, output, cache_write, cache_read)
_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-3-5-haiku-20241022":           (0.80/1e6,  4.00/1e6, 1.00/1e6, 0.08/1e6),
    "claude-3-5-sonnet-20240620":          (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-3-5-sonnet-20241022":          (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-3-7-sonnet-20250219":          (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-sonnet-4-20250514":            (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-sonnet-4-5-20250929":          (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "claude-sonnet-4-5-20251001":          (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
    "gpt-4o-2024-05-13":                   (5.00/1e6, 15.00/1e6, 0.0,      2.50/1e6),
    "gpt-4o-2024-08-06":                   (2.50/1e6, 10.00/1e6, 0.0,      1.25/1e6),
    "gpt-4o-mini-2024-07-18":              (0.15/1e6,  0.60/1e6, 0.0,      0.075/1e6),
    "gpt-5-2025-08-07":                    (10.00/1e6, 40.00/1e6, 0.0,     5.00/1e6),
    "gpt-5.2":                             (10.00/1e6, 40.00/1e6, 0.0,     5.00/1e6),
    "o1-mini-2024-09-12":                  (3.00/1e6, 12.00/1e6, 0.0,      0.0),
    "deepseek-chat":                       (0.27/1e6,  1.10/1e6, 0.0,      0.07/1e6),
    "deepseek-v2.5":                       (0.27/1e6,  1.10/1e6, 0.0,      0.07/1e6),
    "gemini-1.5-pro-002":                  (1.25/1e6,  5.00/1e6, 0.0,      0.0),
    "llama-3.1-405b-instruct":             (0.80/1e6,  0.80/1e6, 0.0,      0.0),
    "llama-3.1-70b-instruct":              (0.12/1e6,  0.12/1e6, 0.0,      0.0),
    "llama-3.2-90b-vision-instruct":       (0.12/1e6,  0.12/1e6, 0.0,      0.0),
    "llama-v3p1-405b-instruct":            (0.80/1e6,  0.80/1e6, 0.0,      0.0),
    "llama-v3p1-70b-instruct":             (0.12/1e6,  0.12/1e6, 0.0,      0.0),
    "llama-v3p3-70b-instruct":             (0.12/1e6,  0.12/1e6, 0.0,      0.0),
    "qwen-2.5-72b-instruct":               (0.40/1e6,  0.40/1e6, 0.0,      0.0),
    "Qwen3-Coder-480B-A35B-Instruct-FP8":  (1.50/1e6,  6.00/1e6, 0.0,      0.0),
    "single-haiku":                        (0.80/1e6,  4.00/1e6, 1.00/1e6, 0.08/1e6),
    "single-sonnet":                       (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6),
}
_DEFAULT_PRICING = (3.00/1e6, 15.00/1e6, 3.75/1e6, 0.30/1e6)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int, num_turns: int) -> float:
    """
    Calculate total cost from token counts.

    prompt_tokens   = context at the final API call (proxy for unique context written)
    completion_tokens = TOTAL output tokens across ALL turns (not per-turn)
    num_turns        = number of API calls (used for cache read estimation)

    Anthropic (cache_write > 0):
      cost = prompt × P_cw + prompt × (N-1)/2 × P_cr + total_completion × P_out

    OpenAI cached (cache_write = 0, cache_read > 0):
      cost = prompt × P_in + prompt × (N-1)/2 × P_cr + total_completion × P_out

    No cache:
      cost = prompt × N/2 × P_in + total_completion × P_out
    """
    p_in, p_out, p_cw, p_cr = _PRICING.get(model, _DEFAULT_PRICING)
    N = max(int(num_turns), 1)
    pt = max(int(prompt_tokens), 0)
    ct = max(int(completion_tokens), 0)

    cache_reads = pt * (N - 1) / 2
    if p_cw > 0:
        return pt * p_cw + cache_reads * p_cr + ct * p_out
    elif p_cr > 0:
        return pt * p_in + cache_reads * p_cr + ct * p_out
    else:
        return pt * N / 2 * p_in + ct * p_out


def load_artifact(path: Path = DEFAULT_MODEL_PATH) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _detect_git_root() -> Optional[Path]:
    import subprocess
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=Path.cwd()
        )
        if r.returncode == 0:
            return Path(r.stdout.strip())
    except Exception:
        pass
    return None


def _make_feature_row(artifact: Dict, text: str, repo_path: Optional[Path]) -> np.ndarray:
    """Build a single dense feature row: L2-normalized TF-IDF + MaxAbsScaled code."""
    from scipy.sparse import hstack, csr_matrix

    tfidf      = artifact["tfidf"]
    l2_norm    = artifact.get("l2_norm")
    code_scaler = artifact.get("code_scaler")
    cols       = artifact.get("code_cols", [])

    X_text = tfidf.transform([text])
    if l2_norm:
        X_text = l2_norm.transform(X_text)

    raw_code = np.zeros((1, len(cols)), dtype=np.float32)
    if cols and repo_path and repo_path.exists():
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.extract_code_features import repo_aggregate_stats
            stats = repo_aggregate_stats(repo_path, task_description=text)
            for i, col in enumerate(cols):
                raw_code[0, i] = float(stats.get(col, 0))
        except Exception:
            pass

    if code_scaler and cols:
        X_code = csr_matrix(code_scaler.transform(raw_code))
        return hstack([X_text, X_code], format="csr").toarray()
    return X_text.toarray()


def predict(
    task_text: str,
    models: Optional[List[str]] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    repo_path: Optional[Path] = None,
) -> List[Dict]:
    artifact  = load_artifact(model_path)
    mdls      = artifact["models"]
    available = artifact.get("model_list", [])
    avg_prompt_by_model = artifact.get("avg_prompt_by_model", {})
    global_avg_prompt   = artifact.get("global_avg_prompt", 50000)

    if models:
        target = []
        for m in models:
            target.extend(a for a in available if m.lower() in a.lower())
        predict_for = list(dict.fromkeys(target)) or available
    else:
        predict_for = available

    if repo_path is None:
        repo_path = _detect_git_root()

    results = []
    for model_id in predict_for:
        text = f"modelname {model_id.replace('-', ' ')} endmodelname {task_text}"
        X = _make_feature_row(artifact, text, repo_path)

        log_xform   = artifact.get("log_transform", False)
        clip_bounds = artifact.get("clip_bounds", {})
        if log_xform:
            t_lo, t_hi = clip_bounds.get("turns_log",      (0.0, 6.0))
            c_lo, c_hi = clip_bounds.get("completion_log", (0.0, 10.0))
            turns    = max(1.0, float(np.expm1(np.clip(mdls["turns"].predict(X)[0],      t_lo, t_hi))))
            comp_tok = max(1.0, float(np.expm1(np.clip(mdls["completion"].predict(X)[0], c_lo, c_hi))))
        else:
            turns    = max(1.0, float(mdls["turns"].predict(X)[0]))
            comp_tok = max(1.0, float(mdls["completion"].predict(X)[0]))
        # Use per-model average context size (not predicted — essentially unpredictable)
        prompt_tok = avg_prompt_by_model.get(model_id, global_avg_prompt)
        cost      = calculate_cost(model_id, prompt_tok, comp_tok, turns)
        pass_rate = float(np.clip(mdls["pass_rate"].predict(X)[0], 0.0, 1.0)) if "pass_rate" in mdls else 0.5

        results.append({
            "model":          model_id,
            "est_cost":       round(cost, 4),
            "est_turns":      round(turns, 1),
            "est_prompt_tok": int(round(prompt_tok)),
            "est_comp_tok":   int(round(comp_tok)),
            "pass_rate":      round(pass_rate, 3),
        })

    results.sort(key=lambda r: r["est_cost"])
    return results


def recommend(results: List[Dict]) -> tuple[str, str]:
    if not results:
        return "unknown", "no models available"

    cheapest = results[0]

    for r in results[1:]:
        pass_gap     = r["pass_rate"] - cheapest["pass_rate"]
        cost_premium = (r["est_cost"] - cheapest["est_cost"]) / max(cheapest["est_cost"], 0.001)
        if pass_gap >= 0.10 and cost_premium < 3.0:
            cheapest = r

    reason_parts = [f"est. cost ${cheapest['est_cost']:.3f}"]
    reason_parts.append(f"{cheapest['est_turns']:.0f} turns")
    reason_parts.append(f"{cheapest['pass_rate']:.0%} pass rate")
    return cheapest["model"], "  ·  ".join(reason_parts)


def _log_prediction(task_text: str, results: List[Dict], rec_model: str) -> str:
    import uuid
    from datetime import datetime, timezone
    pred_id = uuid.uuid4().hex[:8]
    rec = next((r for r in results if r["model"] == rec_model), results[0] if results else {})
    entry = {
        "prediction_id": pred_id,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "task":          task_text[:500],
        "recommendation": rec,
        "all_results":   results,
    }
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_FILE, "a") as f:
        f.write(__import__("json").dumps(entry, separators=(",", ":")) + "\n")
    return pred_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="-")
    parser.add_argument("--models", default=None)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), type=Path)
    parser.add_argument("--repo-path", default=None, type=Path)
    args = parser.parse_args()

    if args.task == "-":
        task_text = sys.stdin.read().strip()
    else:
        task_text = args.task.strip()

    if not task_text:
        parser.print_help()
        sys.exit(1)

    filter_models = [m.strip() for m in args.models.split(",")] if args.models else None
    results = predict(task_text, models=filter_models, model_path=args.model_path,
                      repo_path=args.repo_path)
    rec_model, rec_reason = recommend(results)
    pred_id = _log_prediction(task_text, results, rec_model)

    artifact = load_artifact(args.model_path)
    repo_used = args.repo_path or _detect_git_root()

    print()
    print(f"  Task: {task_text[:100]}{'...' if len(task_text) > 100 else ''}")
    if repo_used:
        print(f"  Repo: {repo_used}")
    print(f"  (predictor trained on {artifact['n_train']} runs)")
    print()
    print(f"  {'Model':<55}  {'Est. cost':>10}  {'Turns':>7}  {'Pass':>6}")
    print("  " + "─" * 82)
    for r in results:
        marker = "◀ recommended" if r["model"] == rec_model else ""
        print(
            f"  {r['model']:<55}  "
            f"${r['est_cost']:>9.3f}  "
            f"{r['est_turns']:>6.1f}  "
            f"{r['pass_rate']:>5.0%}  {marker}"
        )
    print()
    print(f"  Recommendation: {rec_model}")
    print(f"  {rec_reason}")
    print()
    print(f"  Prediction ID: {pred_id}")
    print(f"  To record actuals after your task:")
    print(f"    ! python -m predictor.record_actual {pred_id} --turns <N> [--cost <USD>] [--passed true/false]")
    print()


if __name__ == "__main__":
    main()
