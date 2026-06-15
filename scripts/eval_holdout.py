#!/usr/bin/env python3
"""Evaluate predictor on a held-out 20% test split."""
from __future__ import annotations
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from predictor.train import load_runs, build_dataset

PROJECT_ROOT = Path(__file__).parent.parent


def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - np.mean(y)) ** 2)


def mae(y, yh):
    return np.mean(np.abs(y - yh))


def mape(y, yh):
    return np.mean(np.abs((y - yh) / np.maximum(y, 1)))


def within_2x(y, yh):
    ratio = yh / np.maximum(y, 1)
    return np.mean((ratio >= 0.5) & (ratio <= 2.0))


def main():
    from sklearn.neural_network import MLPRegressor

    records = load_runs(PROJECT_ROOT / "data/runs")
    code_features = {}
    cf_path = PROJECT_ROOT / "data" / "code_features.json"
    if cf_path.exists():
        code_features = json.load(open(cf_path))

    (tok_texts, model_names, tok_code_arr,
     y_turns, y_completion,
     avg_prompt, global_avg,
     pass_texts, pass_code_arr, y_pass) = build_dataset(records, code_features)

    art = pickle.load(open(PROJECT_ROOT / "data/models/predictor_v1.pkl", "rb"))
    tfidf, l2, cs = art["tfidf"], art["l2_norm"], art["code_scaler"]
    cb = art["clip_bounds"]

    def make_X(texts, code_arr):
        Xt = l2.transform(tfidf.transform(texts))
        Xc = csr_matrix(cs.transform(code_arr))
        return hstack([Xt, Xc], format="csr").toarray()

    rng = np.random.RandomState(42)
    n = len(tok_texts)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    print(f"Train: {len(tr_idx)}  Test: {len(te_idx)}", flush=True)
    X_all = make_X(tok_texts, tok_code_arr)
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_turns_te = y_turns[te_idx]
    y_comp_te = y_completion[te_idx]

    kw = dict(
        hidden_layer_sizes=(512, 256, 128), max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.1, alpha=1e-4,
        learning_rate_init=1e-3, n_iter_no_change=20,
    )

    print("Training turns model...", flush=True)
    m_t = MLPRegressor(**kw)
    m_t.fit(X_tr, np.log1p(y_turns[tr_idx]))

    print("Training completion model...", flush=True)
    m_c = MLPRegressor(**kw)
    m_c.fit(X_tr, np.log1p(y_completion[tr_idx]))

    pt = np.expm1(np.clip(m_t.predict(X_te), *cb["turns_log"]))
    pc = np.expm1(np.clip(m_c.predict(X_te), *cb["completion_log"]))

    print()
    print(f"{'Metric':<22} {'turns':>10} {'completion':>12}")
    print("-" * 47)
    print(f"{'R²':<22} {r2(y_turns_te, pt):>10.3f} {r2(y_comp_te, pc):>12.3f}")
    print(f"{'MAE':<22} {mae(y_turns_te, pt):>9.1f}t {mae(y_comp_te, pc):>9.0f}tok")
    print(f"{'MAPE':<22} {mape(y_turns_te, pt):>10.1%} {mape(y_comp_te, pc):>12.1%}")
    print(f"{'Within 2x of actual':<22} {within_2x(y_turns_te, pt):>10.1%} {within_2x(y_comp_te, pc):>12.1%}")
    print()
    print(f"Test set means: turns={y_turns_te.mean():.1f}  completion={y_comp_te.mean():.0f}tok")
    print(f"Clip ranges:    turns=[{np.expm1(cb['turns_log'][0]):.0f},{np.expm1(cb['turns_log'][1]):.0f}]  "
          f"completion=[{np.expm1(cb['completion_log'][0]):.0f},{np.expm1(cb['completion_log'][1]):.0f}]")


if __name__ == "__main__":
    main()
