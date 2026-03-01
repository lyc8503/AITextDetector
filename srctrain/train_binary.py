"""
Two-stage TF-IDF + LinearSVC training for web deployment.

Stage 1: Full TF-IDF + SVC → export full model, rank features by |idf · coef|
Stage 2: Restricted TF-IDF (top-K) + SVC → export pruned model

Models are trained sequentially; scipy/numpy BLAS uses all available cores
automatically for sparse matrix ops (set OMP_NUM_THREADS to limit if needed).
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from loader import load_dataset, to_col, gen_folders
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')


def _train_one(model_name, label_idx, train_set, test_set, C, top_k):
    """Train both stages for a single binary model. Runs in a worker process."""
    # Limit BLAS/OpenMP threads inside each worker to avoid oversubscription
    os.environ.setdefault('OMP_NUM_THREADS', '2')

    x_train, y_train = to_col(train_set, only_model=model_name)
    x_train = np.array(x_train); y_train = np.array(y_train)
    x_test, y_test = to_col(test_set, only_model=model_name)
    x_test = np.array(x_test); y_test = np.array(y_test)
    print(f"[{model_name}] Train: {len(x_train):,}  Test: {len(x_test):,}")
    mask_tr = (y_train == 0) | (y_train == label_idx)
    X_tr = x_train[mask_tr]; y_tr = (y_train[mask_tr] == label_idx).astype(int)
    mask_te = (y_test == 0) | (y_test == label_idx)
    X_te = x_test[mask_te]; y_te = (y_test[mask_te] == label_idx).astype(int)

    # Stage 1: full features -> full model export + ranking
    print(f"\n[{model_name}] Stage 1: full TF-IDF + SVC ...", flush=True)
    tfidf_full = TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                 min_df=3, sublinear_tf=True)
    X_tr_full = tfidf_full.fit_transform(X_tr)
    svc_full = LinearSVC(C=C, max_iter=5000)
    svc_full.fit(X_tr_full, y_tr)
    y_pred_s1 = svc_full.predict(tfidf_full.transform(X_te))
    acc_s1 = accuracy_score(y_te, y_pred_s1)
    f1_s1 = f1_score(y_te, y_pred_s1)
    tn1, fp1, fn1, tp1 = confusion_matrix(y_te, y_pred_s1).ravel()
    n_full = X_tr_full.shape[1]
    print(f"  [{model_name}] {n_full:,} features -> acc={acc_s1:.4f}  f1={f1_s1:.4f}  "
          f"[tn={tn1} fp={fp1} fn={fn1} tp={tp1}]")

    # Export full model
    joblib.dump(tfidf_full, os.path.join(OUTPUT_DIR, f'tfidf_full_{model_name}.joblib'))
    joblib.dump(svc_full,   os.path.join(OUTPUT_DIR, f'model_full_{model_name}.joblib'))
    print(f"  [{model_name}] Saved full model ({n_full:,} features)")

    # Select top-K
    importance = np.abs(tfidf_full.idf_ * svc_full.coef_[0])
    k = min(top_k, n_full)
    top_idx = np.argpartition(importance, -k)[-k:]
    rev_vocab = {idx: g for g, idx in tfidf_full.vocabulary_.items()}
    restricted_vocab = {rev_vocab[i]: j for j, i in enumerate(top_idx)}
    del tfidf_full, svc_full, X_tr_full

    # Stage 2: restricted features -> pruned model export
    print(f"[{model_name}] Stage 2: top-{k:,} TF-IDF + SVC ...", flush=True)
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                            sublinear_tf=True, vocabulary=restricted_vocab)
    X_tr_vec = tfidf.fit_transform(X_tr)
    X_te_vec = tfidf.transform(X_te)
    svc = LinearSVC(C=C, max_iter=5000)
    svc.fit(X_tr_vec, y_tr)

    y_pred = svc.predict(X_te_vec)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

    print(f"  [{model_name}] acc={acc:.4f}  f1={f1:.4f}  "
          f"[tn={tn} fp={fp} fn={fn} tp={tp}]  delta={acc - acc_s1:+.4f}")

    joblib.dump(tfidf, os.path.join(OUTPUT_DIR, f'tfidf_{model_name}.joblib'))
    joblib.dump(svc,   os.path.join(OUTPUT_DIR, f'model_{model_name}.joblib'))
    print(f"  [{model_name}] Saved pruned model (top-{k:,} features)")

    return (model_name, acc_s1, f1_s1, acc, f1)


def train_binary_models(C=1.0, top_k=20_000):
    print("Loading dataset...")
    train_set, test_set = load_dataset()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_workers = len(gen_folders)
    print(f"Spawning {n_workers} parallel workers (one per model)...")
    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for model_name, label_idx in gen_folders.items():
            fut = pool.submit(_train_one, model_name, label_idx,
                              train_set, test_set, C, top_k)
            futures[fut] = model_name

    # Collect results in original model order
    results = {}
    for fut in as_completed(futures):
        results[futures[fut]] = fut.result()
    summary = [results[m] for m in gen_folders]

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY  top-{top_k:,}  C={C}")
    print(f"{'='*60}")
    print(f"  {'model':<14} {'s1 acc':>8} {'s2 acc':>8} {'Δacc':>8} {'s1 f1':>8} {'s2 f1':>8} {'Δf1':>8}")
    for name, a1, f1_1, a2, f1_2 in summary:
        print(f"  {name:<14} {a1:>8.4f} {a2:>8.4f} {a2-a1:>+8.4f} {f1_1:>8.4f} {f1_2:>8.4f} {f1_2-f1_1:>+8.4f}")
    accs = [a for _, _, _, a, _ in summary]
    f1s = [f for _, _, _, _, f in summary]
    print(f"  {'AVG':<14} {'':>8} {np.mean(accs):>8.4f} {'':>8} {'':>8} {np.mean(f1s):>8.4f}")
    print(f"  MIN acc: {min(accs):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--topk', type=int, default=20_000)
    args = parser.parse_args()
    train_binary_models(C=args.C, top_k=args.topk)
