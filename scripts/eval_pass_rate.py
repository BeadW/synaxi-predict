#!/usr/bin/env python3
"""Evaluate pass-rate predictor: AUC, Brier score, calibration."""
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


def main():
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    records = load_runs(PROJECT_ROOT / "data/runs")
    cf_path = PROJECT_ROOT / "data" / "code_features.json"
    code_features = json.load(open(cf_path)) if cf_path.exists() else {}

    (tok_texts, model_names, tok_code_arr, y_turns, y_comp, avg_p, gavg,
     pass_texts, pass_code_arr, y_pass) = build_dataset(records, code_features)

    art = pickle.load(open(PROJECT_ROOT / "data/models/predictor_v1.pkl", "rb"))
    tfidf, l2, cs = art["tfidf"], art["l2_norm"], art["code_scaler"]

    def make_X(texts, code_arr):
        Xt = l2.transform(tfidf.transform(texts))
        Xc = csr_matrix(cs.transform(code_arr))
        return hstack([Xt, Xc], format="csr").toarray()

    rng = np.random.RandomState(42)
    n = len(pass_texts)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    print(f"Pass/fail dataset: {n} records  ({y_pass.mean():.1%} pass rate overall)")
    print(f"Train: {len(tr_idx)}  Test: {len(te_idx)}\n", flush=True)

    X_all = make_X(pass_texts, pass_code_arr)
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_pass[tr_idx], y_pass[te_idx]

    kw = dict(
        hidden_layer_sizes=(512, 256, 128), max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.1, alpha=1e-4,
        learning_rate_init=1e-3, n_iter_no_change=20,
    )
    print("Training pass_rate model...", flush=True)
    m = MLPRegressor(**kw)
    m.fit(X_tr, y_tr)

    raw_pred = m.predict(X_te)
    pred = np.clip(raw_pred, 0.0, 1.0)
    binary_pred = (pred >= 0.5).astype(float)

    auc = roc_auc_score(y_te, pred)
    brier = brier_score_loss(y_te, pred)
    ll = log_loss(y_te, np.clip(pred, 1e-6, 1 - 1e-6))
    acc = np.mean(binary_pred == y_te)
    baseline_acc = max(y_te.mean(), 1 - y_te.mean())  # majority class

    print()
    print(f"  AUC-ROC:              {auc:.3f}   (0.5=random, 1.0=perfect)")
    print(f"  Brier score:          {brier:.3f}   (0=perfect, 0.25=random for 50/50)")
    print(f"  Log loss:             {ll:.3f}")
    print(f"  Accuracy @ 0.5 thr:   {acc:.1%}   (majority-class baseline: {baseline_acc:.1%})")
    print()

    # Calibration: bucket predicted probs and compare to actual pass rate
    print("Calibration (predicted probability vs actual pass rate):")
    print(f"  {'Pred bucket':<15}  {'Count':>6}  {'Pred mean':>10}  {'Actual pass':>11}")
    print("  " + "-" * 50)
    buckets = np.linspace(0, 1, 11)
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (pred >= lo) & (pred < hi)
        if mask.sum() > 0:
            print(f"  {lo:.1f}–{hi:.1f}          {mask.sum():>6}  {pred[mask].mean():>10.2f}  {y_te[mask].mean():>11.2f}")

    # Per-source breakdown — need to reload raw records to get source labels
    print()
    print("Per-source breakdown (using full dataset, not split):")
    all_pred = np.clip(m.predict(X_all), 0.0, 1.0)

    # Rebuild source labels in same order as build_dataset produces them
    import sys
    from collections import defaultdict
    src_results = defaultdict(list)
    for f in sorted(PROJECT_ROOT.glob("data/runs/*.jsonl")):
        is_cc = f.stem.startswith("2026")
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    pc = r.get("passed_criteria")
                    if pc is None:
                        continue
                    src = "cc" if is_cc else r.get("source", "unknown")
                    src_results[src].append(float(bool(pc)))
                except Exception:
                    pass
    for src, vals in sorted(src_results.items()):
        print(f"  {src:<20}  {len(vals):>6} records  actual_pass={np.mean(vals):.1%}")


if __name__ == "__main__":
    main()
