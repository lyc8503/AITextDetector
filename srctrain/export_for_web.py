"""
Export two-stage TF-IDF + LinearSVC models to models.json for browser inference.

Since models are trained on restricted vocabulary (see train_binary.py),
ALL features are exported — no pruning needed → zero accuracy gap.

Each feature: gram → [combined, idf]
  combined = idf · coef  (precomputed)
  idf needed for L2 normalisation in browser

Browser inference:
    tf = 1 + ln(count)
    score = Σ tf_i · combined_i / sqrt(Σ (tf_i · idf_i)²) + intercept

Usage:  python srctrain/export_for_web.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from loader import gen_folders

import joblib
import numpy as np
import json

MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', 'model')
SRCWEB_DIR  = os.path.join(os.path.dirname(__file__), '..', 'srcweb')
OUTPUT_PATH = os.path.join(SRCWEB_DIR, 'models.json')


def export():
    all_data = {}

    for model_name in gen_folders.keys():
        print(f"Exporting {model_name} …", end="  ", flush=True)

        tfidf_path = os.path.join(MODEL_DIR, f'tfidf_{model_name}.joblib')
        model_path = os.path.join(MODEL_DIR, f'model_{model_name}.joblib')

        if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
            print("SKIP (joblib files not found)")
            continue

        tfidf = joblib.load(tfidf_path)
        model = joblib.load(model_path)

        vocab     = tfidf.vocabulary_
        idf       = tfidf.idf_
        coef      = model.coef_[0]
        intercept = float(model.intercept_[0])
        combined  = idf * coef

        rev_vocab = {idx: gram for gram, idx in vocab.items()}

        # Export ALL features (model was trained on restricted vocab)
        # Only skip features with zero coefficient
        weights = {}
        for idx in range(len(combined)):
            if coef[idx] == 0.0:
                continue
            gram = rev_vocab[idx]
            weights[gram] = [
                round(float(combined[idx]), 5),
                round(float(idf[idx]), 5),
            ]

        all_data[model_name] = {
            'weights':   weights,
            'intercept': round(intercept, 6),
        }
        print(f"{len(weights):,} features (of {len(vocab):,} total)")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, separators=(',', ':'))

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nSaved → {OUTPUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    export()
