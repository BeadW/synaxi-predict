"""
Train the cost/turn predictor on benchmark run data.

Targets:
  - num_turns          (predicted from task text + model + code features)
  - completion_tokens  (per-response token count; predicted)
  - prompt_tokens      (final context size; NOT predicted — use per-model average)
  - passed_criteria    (pass rate, where labeled)

Cost is calculated at inference time from predicted tokens × pricing formula.
Model: MLP on L2-normalized TF-IDF + MaxAbsScaled code features.

Usage:
    python -m predictor.train
    python -m predictor.train --runs data/runs/ --out data/models/
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import hstack, csr_matrix

PROJECT_ROOT = Path(__file__).parent.parent

CODE_FEATURE_COLS = [
    "loc", "functions", "classes", "branches",
    "try_blocks", "n_files", "avg_loc", "branch_density", "has_code_features",
]


def load_code_features(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_runs(runs_dir: Path) -> List[Dict]:
    records = []
    cc_records = []
    for f in sorted(runs_dir.glob("*.jsonl")):
        is_cc = f.stem.startswith("2026")  # our own Claude Code runs (dated files)
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("mode") == "multi-turn" and r.get("num_turns", 0) > 0:
                        records.append(r)
                        if is_cc:
                            cc_records.append(r)
                except Exception:
                    pass

    # Upsample our own CC runs to prevent SWE-bench data from dominating.
    # Target: CC records should represent ~20% of total training set.
    if cc_records:
        total = len(records)
        target_cc = max(len(cc_records), total // 5)  # 20% of total
        repeat = max(1, target_cc // len(cc_records))
        if repeat > 1:
            records.extend(cc_records * (repeat - 1))

    return records


def _has_valid_tokens(r: Dict) -> bool:
    return r.get("prompt_tokens_raw", 0) > 0 and r.get("completion_tokens", 0) > 0


def get_task_text(record: Dict) -> str:
    if record.get("task_text"):
        return record["task_text"]
    task_id = record.get("task_id", "")
    feature_path = PROJECT_ROOT / "features" / (task_id + ".feature")
    if feature_path.exists():
        try:
            from scripts.run_benchmark import build_prompt
            prompt, _ = build_prompt(feature_path)
            return prompt
        except Exception:
            pass
    return task_id


def build_dataset(records: List[Dict], code_features: Dict) -> Tuple:
    # Token prediction subset (needs valid prompt + completion tokens)
    tok_texts, tok_model_names, tok_code_rows = [], [], []
    y_turns, y_completion = [], []
    # Separate per-model prompt token averages (for cost calculation at inference)
    model_prompt_tokens: dict[str, list] = defaultdict(list)
    # Pass rate subset (all records with label)
    pass_texts, pass_model_names, pass_code_rows = [], [], []
    y_pass: List[float] = []

    for r in records:
        task_text = get_task_text(r)
        if not task_text:
            continue
        model = r.get("strategy", "unknown")
        text = f"modelname {model.replace('-', ' ')} endmodelname {task_text}"

        raw_id = r.get("task_id", "")
        instance_id = raw_id.split("/")[-1].replace(".feature", "")
        cf = code_features.get(instance_id, {})
        code_row = [float(cf.get(col, 0)) for col in CODE_FEATURE_COLS]

        if _has_valid_tokens(r):
            tok_texts.append(text)
            tok_model_names.append(model)
            tok_code_rows.append(code_row)
            y_turns.append(float(r["num_turns"]))
            y_completion.append(float(r["completion_tokens"]))
            model_prompt_tokens[model].append(float(r["prompt_tokens_raw"]))

        pc = r.get("passed_criteria")
        if pc is not None:
            pass_texts.append(text)
            pass_model_names.append(model)
            pass_code_rows.append(code_row)
            y_pass.append(float(bool(pc)))

    # Per-model average prompt_tokens (for cost calculation at inference)
    avg_prompt_by_model = {m: float(np.mean(v)) for m, v in model_prompt_tokens.items()}
    global_avg_prompt = float(np.mean([v for vals in model_prompt_tokens.values() for v in vals]))

    return (
        tok_texts,
        list(set(tok_model_names)),
        np.array(tok_code_rows, dtype=np.float32),
        np.array(y_turns),
        np.array(y_completion),
        avg_prompt_by_model,
        global_avg_prompt,
        pass_texts,
        np.array(pass_code_rows, dtype=np.float32),
        np.array(y_pass),
    )


def train(runs_dir: Path, models_dir: Path, verbose: bool = True) -> Dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import MaxAbsScaler, Normalizer
    from sklearn.neural_network import MLPRegressor

    records = load_runs(runs_dir)
    if not records:
        raise ValueError(f"No valid run records in {runs_dir}")

    code_features = load_code_features(PROJECT_ROOT / "data" / "code_features.json")
    (tok_texts, model_names, tok_code_arr,
     y_turns, y_completion,
     avg_prompt_by_model, global_avg_prompt,
     pass_texts, pass_code_arr, y_pass) = build_dataset(records, code_features)

    n_with_code = int(tok_code_arr[:, -1].sum())
    if verbose:
        print(f"Token training samples: {len(tok_texts)}")
        print(f"  models: {len(model_names)}")
        print(f"  code features: {n_with_code}/{len(tok_texts)} ({100*n_with_code//max(len(tok_texts),1)}%)")
        print(f"  turns:             {y_turns.min():.0f}–{y_turns.max():.0f}  (mean {y_turns.mean():.1f})")
        print(f"  completion_tokens: {y_completion.min():.0f}–{y_completion.max():.0f}  (mean {y_completion.mean():.0f})")
        print(f"  prompt tokens per model (averages):")
        for m, avg in sorted(avg_prompt_by_model.items()):
            print(f"    {m:<55}  {avg:,.0f} tok")
        print(f"Pass rate samples:  {len(pass_texts)} ({y_pass.mean():.1%} pass)")
        print()

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)
    all_texts = tok_texts + pass_texts
    tfidf.fit(all_texts)

    l2_norm = Normalizer(norm="l2")
    code_scaler = MaxAbsScaler()

    def _make_X(texts, code_arr):
        X_text = l2_norm.transform(tfidf.transform(texts))
        X_code = csr_matrix(code_scaler.transform(code_arr))
        return hstack([X_text, X_code], format="csr").toarray()

    code_scaler.fit(np.vstack([tok_code_arr, pass_code_arr]))
    l2_norm.fit(tfidf.transform(all_texts))

    X_tok = _make_X(tok_texts, tok_code_arr)
    X_pass = _make_X(pass_texts, pass_code_arr)

    if verbose:
        print(f"  Token feature matrix: {X_tok.shape[0]} × {X_tok.shape[1]}")
        print(f"  Pass  feature matrix: {X_pass.shape[0]} × {X_pass.shape[1]}")

    mlp_params = dict(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        alpha=0.0001,
        learning_rate_init=0.001,
        n_iter_no_change=20,
    )

    fitted_models = {}
    targets = [
        ("turns",      X_tok,  np.log1p(y_turns)),
        ("completion", X_tok,  np.log1p(y_completion)),
    ]
    if len(y_pass) > 0:
        targets.append(("pass_rate", X_pass, y_pass))

    for name, X_t, y_t_log in targets:
        mdl = MLPRegressor(**mlp_params)
        mdl.fit(X_t, y_t_log)
        # MAE in original units for interpretability
        y_pred_orig = np.expm1(mdl.predict(X_t))
        if name == "turns":
            y_orig = y_turns
        elif name == "completion":
            y_orig = y_completion
        else:
            y_orig = y_t_log  # pass_rate stored directly (not log-transformed)
            y_pred_orig = mdl.predict(X_t)
        mae = np.mean(np.abs(y_pred_orig - y_orig))
        fitted_models[name] = mdl
        if verbose:
            unit = "tok" if name == "completion" else ""
            print(f"  {name:12s} train MAE: {mae:.1f}{unit}  (n={len(y_t_log)})")

    # Clip bounds: prevent extrapolation beyond p1/p99 training range
    clip_bounds = {
        "turns_log":      (float(np.log1p(np.percentile(y_turns, 1))),
                           float(np.log1p(np.percentile(y_turns, 99)))),
        "completion_log": (float(np.log1p(np.percentile(y_completion, 1))),
                           float(np.log1p(np.percentile(y_completion, 99)))),
    }
    if verbose:
        for k, (lo, hi) in clip_bounds.items():
            print(f"  clip {k}: [{np.expm1(lo):.0f}, {np.expm1(hi):.0f}]")

    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "tfidf":               tfidf,
        "l2_norm":             l2_norm,
        "code_scaler":         code_scaler,
        "code_cols":           CODE_FEATURE_COLS,
        "models":              fitted_models,
        "model_list":          sorted(model_names),
        "avg_prompt_by_model": avg_prompt_by_model,
        "global_avg_prompt":   global_avg_prompt,
        "n_train":             len(tok_texts),
        "log_transform":       True,  # turns and completion are log1p-transformed
        "clip_bounds":         clip_bounds,
    }
    out_path = models_dir / "predictor_v1.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)

    if verbose:
        print(f"\nSaved to {out_path}")

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="data/runs", type=Path)
    parser.add_argument("--out",  default="data/models", type=Path)
    args = parser.parse_args()
    runs_dir   = PROJECT_ROOT / args.runs if not args.runs.is_absolute() else args.runs
    models_dir = PROJECT_ROOT / args.out  if not args.out.is_absolute()  else args.out
    train(runs_dir, models_dir)


if __name__ == "__main__":
    main()
