#!/usr/bin/env python3
"""
Fraudalysis — Synthetic Financial Fraud Detection Benchmark
CPU-accelerated analysis of 6.3M mobile money transactions (scikit-learn Random Forest)
Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
"""

import pandas as pd
import numpy as np
import json
import time
import os
from collections import Counter

print("=" * 60)
print("  FRAUDALYSIS — SYNTHETIC FINANCIAL BENCHMARK")
print("  CPU-Based Analysis (scikit-learn, no GPU used)")
print("=" * 60)

# ── GPU detection ──
# Detect rather than assume, so the banner and the published JSON can never
# claim hardware that wasn't actually used. Kaggle exposes a T4 when a GPU
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

# Find the CSV in Kaggle input
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

# A non-breaking space (\\xa0) had crept in before "GB", so it landed INSIDE the
# format spec as ".2f\xa0GB" and Python raised
#   ValueError: Invalid format specifier '.2f GB' for object of type 'float'
# -- killing the run here, before any benchmarking happened. Moving the unit
# out of the braces entirely means no invisible character can break it again.
df = pd.read_csv(csv_path)
load_time = time.time() - start
size_gb = df.memory_usage(deep=True).sum() / 1024**3
print(f"✅ Loaded {len(df):,} rows ({size_gb:.2f} GB) in {load_time:.2f}s")
print(f"Columns: {list(df.columns)}")

# ── 1. Transaction Overview ──
print(f"\n{'─'*50}")
print("  1. TRANSACTION OVERVIEW")
print(f"{'─'*50}")
print(f"Total transactions: {len(df):,}")
print(f"Time steps: {df['step'].min()} to {df['step'].max()} ({(df['step'].max() - df['step'].min())} hours)")

type_counts = df['type'].value_counts()
print(f"\nTransaction Types:")
for t, c in type_counts.items():
    print(f"  {t:>12s}: {c:>8,} ({c/len(df)*100:5.2f}%)")

# ── 2. Fraud Detection ──
print(f"\n{'─'*50}")
print("  2. FRAUD DETECTION RESULTS")
print(f"{'─'*50}")

fraud = df[df['isFraud'] == 1]
legit = df[df['isFraud'] == 0]
flagged = df[df['isFlaggedFraud'] == 1]

print(f"Fraud cases detected: {len(fraud):,} ({len(fraud)/len(df)*100:.4f}%)")
print(f"Flagged fraud: {len(flagged):,}")
print(f"Legitimate transactions: {len(legit):,}")

print(f"\nFraud by Transaction Type:")
fraud_by_type = df[df['isFraud'] == 1].groupby('type').size().sort_values(ascending=False)
for t, c in fraud_by_type.items():
    total_type = len(df[df['type'] == t])
    print(f"  {t:>12s}: {c:>6,} / {total_type:,} ({c/total_type*100:.2f}%)")

# ── 3. Amount Anaysis ──
print(f"\n{'─'*50}")
print("  3. AMOUNT ANALYSIS")
print(f"{'─'*50}")

fraud_amts = fraud['amount']
legit_amts = legit['amount']

print("\nFraud Amounts:")
print(f"  Min:     {fraud_amts.min():>12.2f}")
print(f"  Max:     {fraud_amts.max():>12.2f}")
print(f"  Mean:    {fraud_amts.mean():>12.2f}")
print(f"  Median:  {fraud_amts.median():>12.2f}")

print("\nLegitimate Amounts:")
print(f"  Min:     {legit_amts.min():>12.2f}")
print(f"  Max:     {legit_amts.max():>12.2f}")
print(f"  Mean:    {legit_amts.mean():>12.2f}")
print(f"  Median:  {legit_amts.median():>12.2f}")

# ── 4. ML Model ──
print(f"\n{'─'*50}")
print("  4. MACHINE LEARNING MODEL")
print(f"{'─'*50}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Prepare features
df_ml = df.copy()
df_ml['type_encoded'] = pd.factorize(df_ml['type'])[0]

features = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'type_encoded']
X = df_ml[features]
y = df_ml['isFraud']

# Handle imbalance — sample
fraud_idx = y[y == 1].index
legit_idx = y[y == 0].sample(n=len(fraud_idx) * 2, random_state=42).index
sample_idx = fraud_idx.union(legit_idx)

X_sample = X.loc[sample_idx]
y_sample = y.loc[sample_idx]

print(f"Training on {len(X_sample):,} samples ({len(fraud_idx):,} fraud, {len(legit_idx):,} legitimate)")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)

# Train
start = time.time()
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
train_time = time.time() - start
print(f"Model trained in {train_time:.2f}s")

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Feature importance
importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nFeature Importances:")
print(importances.to_string(index=False))

# ── 5. Summary Results ──
print(f"\n{'='*60}")
print("  BENCHMARK RESULTS SUMMARY")
print(f"{'='*60}")

results = {
    'dataset': 'Synthetic Financial Datasets For Fraud Detection',
    'source': 'https://www.kaggle.com/datasets/ealaxi/paysim1',
    'total_tansactions': int(len(df)),
    'fraud_cases': int(len(fraud)),
    'fraud_rate_percent': round(float(len(fraud) / len(df) * 100), 4),
    'flagged_fraud': int(len(flagged)),
    'transaction_types': {k: int(v) for k, v in type_counts.items()},
    'ml_mode': 'Random Forest (100 trees)',
    'roc_auc_score': round(float(roc_auc), 4),
    'processing_time_seconds': round(load_time + train_time, 2),
    # Probed, not asserted. This was a hardcoded False with a comment, which
    # meant nobody ever checked whether a GPU was actually present -- and the
    # script banner claimed GPU acceleration regardless. Recording the real
    # detection keeps the published numbers and the claim honest.
    'gpu_accelerated': False,  # scikit-learn estimators here are CPU-only
    'gpu_present': gpu_present,
    'gpu_name': gpu_name,
    'hardware_note': 'Kaggle notebook runtime; CPU-based scikit-learn estimators (no GPU acceleration used)',
    'top_features': importances.to_dict('records')
}

# Save results
with open('/kaggle/working/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print(f"\n✅ Results saved to /kaggle/working/results.json")
print(f"✅ Notebook complete")