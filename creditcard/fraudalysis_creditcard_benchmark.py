#!/usr/bin/env python3
"""
Fraudalysis — Credit Card Fraud Detection Benchmark
CPU-accelerated analysis of 284,807 anonymised European credit card transactions
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Methodology is deliberately identical to the synthetic-financial and metaverse
benchmarks (Random Forest, 100 trees, random_state=42, CPU-only) so all three
results are directly comparable.
"""

import pandas as pd
import numpy as np
import json
import time
import os

print("=" * 60)
print("  FRAUDALYSIS — CREDIT CARD FRAUD BENCHMARK")
print("  CPU-Based Analysis (scikit-learn, no GPU used)")
print("=" * 60)

# ── GPU detection ──
# Probed, not asserted, so the banner and the published JSON can never claim
# hardware that wasn't actually used. Kaggle exposes a T4 when a GPU
# accelerator is enabled, but these estimators are CPU-only either way.
gpu_present = False
gpu_name = "none (CPU-only runtime)"
try:
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if out.returncode == 0 and out.stdout.strip():
        gpu_name = out.stdout.strip().splitlines()[0].strip()
        gpu_present = True
except Exception:
    pass
print(f"  GPU present: {gpu_present} ({gpu_name})")
print("  Note: estimators below are scikit-learn CPU models; GPU not used.")

# ── Load Data ──
print("\nLoading dataset...")
start = time.time()

csv_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.endswith('.csv'):
            csv_path = os.path.join(root, f)
            break
    if csv_path:
        break

if not csv_path or not os.path.exists(csv_path):
    print("❌ CSV not found in Kaggle input")
    exit(1)

# NOTE: the unit sits outside the braces on purpose. An invisible non-breaking
# space before "GB" in the sibling benchmark script landed inside the format
# spec as ".2f\xa0GB" and raised ValueError, killing the run before any
# benchmarking happened. Assigning the value to a variable first makes that
# class of invisible-character bug impossible.
df = pd.read_csv(csv_path)
load_time = time.time() - start
size_gb = df.memory_usage(deep=True).sum() / 1024**3
print(f"✅ Loaded {len(df):,} rows ({size_gb:.2f} GB) in {load_time:.2f}s")

# ── 1. Transaction Overview ──
print(f"\n{'-'*50}")
print("  1. TRANSACTION OVERVIEW")
print(f"{'-'*50}")

# V1..V28 are PCA components of the original (confidential) features, already
# mean-centred. Time is in seconds since the first transaction in the window.
v_cols = [c for c in df.columns if c.startswith("V")]
print(f"Total transactions: {len(df):,}")
print(f"Anonymised PCA features: {len(v_cols)} (V1-V{len(v_cols)})")
if "Time" in df.columns:
    print(f"Time window: {df['Time'].min():,.0f}s to {df['Time'].max():,.0f}s "
          f"({(df['Time'].max() - df['Time'].min()) / 3600:.1f} hours)")

# ── 2. Fraud Detection ──
print(f"\n{'-'*50}")
print("  2. FRAUD DETECTION RESULTS")
print(f"{'-'*50}")

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]
fraud_rate = len(fraud) / len(df) * 100

print(f"Fraud cases detected: {len(fraud):,} ({fraud_rate:.4f}%)")
print(f"Legitimate transactions: {len(legit):,}")
print(f"\nClass ratio (legit:fraud) = {len(legit) / max(len(fraud), 1):.0f}:1")

# ── 3. Amount Analysis ──
print(f"\n{'-'*50}")
print("  3. AMOUNT ANALYSIS")
print(f"{'-'*50}")

# Worth calling out in the report: mean fraud amount is HIGHER than legitimate
# but the MEDIAN is much LOWER. Card-present fraud clusters in many small
# transactions; the mean is pulled up by a few large ones. A mean-only summary
# would misrepresent the pattern.
print("\nFraud Amounts:")
for label, fn in (("Min", "min"), ("Max", "max"), ("Mean", "mean"), ("Median", "median")):
    print(f"  {label:<7s}: {getattr(fraud['Amount'], fn)():>12.2f}")

print("\nLegitimate Amounts:")
for label, fn in (("Min", "min"), ("Max", "max"), ("Mean", "mean"), ("Median", "median")):
    print(f"  {label:<7s}: {getattr(legit['Amount'], fn)():>12.2f}")

print(f"\nMedian fraud amount is {(fraud['Amount'].median() / legit['Amount'].median()):.2f}x "
      f"the legitimate median, while the mean is {(fraud['Amount'].mean() / legit['Amount'].mean()):.2f}x.")

# ── 4. ML Model ──
print(f"\n{'-'*50}")
print("  4. MACHINE LEARNING MODEL")
print(f"{'-'*50}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

features = v_cols + ['Amount']
X = df[features]
y = df['Class']

# Handle imbalance — oversample fraud to 3:1 against legitimate. Same ratio and
# random_state as the sibling benchmarks so the ROC-AUC stays comparable.
fraud_idx = y[y == 1].index
legit_idx = y[y == 0].sample(n=len(fraud_idx) * 3, random_state=42).index
sample_idx = fraud_idx.union(legit_idx)

X_sample = X.loc[sample_idx]
y_sample = y.loc[sample_idx]

print(f"Training on {len(X_sample):,} samples ({len(fraud_idx):,} fraud, {len(legit_idx):,} legitimate)")

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)

start = time.time()
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
train_time = time.time() - start
print(f"Model trained in {train_time:.2f}s")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True)

print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
print(f"Confusion Matrix (rows=true, cols=pred):")
print(cm)

tn, fp, fn_, tp = cm.ravel()

importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nTop 10 Feature Importances:")
print(importances.head(10).to_string(index=False))

# ── 5. Summary Results ──
print(f"\n{'='*60}")
print("  BENCHMARK RESULTS SUMMARY")
print(f"{'='*60}")

results = {
    'dataset': 'Credit Card Fraud Detection',
    'source': 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
    'total_transactions': int(len(df)),
    'fraud_cases': int(len(fraud)),
    'fraud_rate_percent': round(float(fraud_rate), 4),
    'amount_stats': {
        'fraud_mean': round(float(fraud['Amount'].mean()), 2),
        'fraud_median': round(float(fraud['Amount'].median()), 2),
        'fraud_max': round(float(fraud['Amount'].max()), 2),
        'legit_mean': round(float(legit['Amount'].mean()), 2),
        'legit_median': round(float(legit['Amount'].median()), 2),
        'legit_max': round(float(legit['Amount'].max()), 2)
    },
    'ml_mode': 'Random Forest (100 trees)',
    'roc_auc_score': round(float(roc_auc), 4),
    'precision_fraud': round(float(report['Fraud']['precision']), 4),
    'recall_fraud': round(float(report['Fraud']['recall']), 4),
    'confusion_matrix': [[int(tn), int(fp)], [int(fn_), int(tp)]],
    'processing_time_seconds': round(load_time + train_time, 2),
    # Probed, not asserted, so this can never claim hardware that wasn't used.
    'gpu_accelerated': False,  # scikit-learn estimators here are CPU-only
    'gpu_present': gpu_present,
    'gpu_name': gpu_name,
    'hardware_note': 'Kaggle notebook runtime; CPU-based scikit-learn estimators (no GPU acceleration used)',
    'top_features': importances.head(10).to_dict('records')
}

# Save results
with open('/kaggle/working/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print(f"\n✅ Results saved to /kaggle/working/results.json")
print(f"✅ Notebook complete")