#!/usr/bin/env python3
"""
Fraudalysis — Ethereum Fraud Detection Benchmark
CPU-accelerated analysis of 9,841 real Ethereum addresses flagged for fraud
Dataset: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

IMPORTANT DIFFERENCE FROM THE OTHER TWO BENCHMARKS
---------------------------------------------------
Credit-card and synthetic-financial are TRANSACTION-level: each row is one
transaction. This Ethereum dataset is ADDRESS-level: each row is one address
with aggregate behavioural features (how many txs sent/received, time between
transactions, total Ether moved, contract-creation counts, and so on).

That means the headline number here is addresses, not transactions. The
benchmark still measures the same thing the others do -- detection accuracy of
a fraudulent pattern -- but the unit is an account. Report it as such; calling
it "transactions" would overstate the volume by orders of magnitude.

Methodology matches the sibling benchmarks where it's meaningful (Random Forest,
100 trees, random_state=42, CPU-only) so ROC-AUC figures remain comparable.
"""

import pandas as pd
import numpy as np
import json
import time
import os

print("=" * 60)
print("  FRAUDALYSIS — ETHEREUM FRAUD DETECTION BENCHMARK")
print("  CPU-Based Analysis (scikit-learn, no GPU used)")
print("  Unit of analysis: ADDRESS (not transaction)")
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

# The unit sits outside the braces on purpose. An invisible non-breaking space
# before "GB" in a sibling script landed inside the format spec as
# ".2f\xa0GB" and raised ValueError, killing the run before any benchmarking.
# Assigning to a variable first makes that class of bug impossible.
df = pd.read_csv(csv_path)
load_time = time.time() - start
size_gb = df.memory_usage(deep=True).sum() / 1024**3
print(f"✅ Loaded {len(df):,} rows ({size_gb:.2f} GB) in {load_time:.2f}s")
print(f"Raw columns: {len(df.columns)}")

# ── Clean up columns ──
# This dataset has inconsistent leading/trailing whitespace in its headers
# (' Total ERC20 tnxs', 'max value received ') and two object columns
# (most-sent / most-received token type) that are strings, not numbers.
# Normalise before anything else so feature selection is reliable.
df.columns = [str(c).strip() for c in df.columns]
print("Stripped whitespace from column names")

drop_cols = [c for c in ['Unnamed: 0', 'Index', 'Address', 'FLAG'] if c in df.columns]
feature_frame = df.drop(columns=drop_cols)

# This dataset needs more than a strip. Three separate problems, all of which
# crash sklearn at fit time with "could not convert string to float":
#
#   1. The ERC20 columns parse as STRING dtype, because 829 of 9,841 rows
#      hold a whitespace-only cell (' ') rather than a number.
#   2. Two 'most ... token type' columns are genuinely categorical and must
#      be encoded, not coerced.
#   3. Coercing a real categorical column with pd.to_numeric would turn every
#      token name into NaN and silently drop the signal.
#
# So: coerce each column, and encode-as-category the ones that are still
# non-numeric afterwards. Everything else becomes a float.
feature_frame = feature_frame.apply(lambda col: pd.to_numeric(col, errors='coerce'))
remaining_object = [c for c in feature_frame.columns if feature_frame[c].dtype == object]
for c in remaining_object:
    feature_frame[c] = pd.factorize(feature_frame[c].fillna('__missing__'))[0]
    print(f"Encoded categorical column: {c}")

feature_frame = feature_frame.astype(float)

# Drop any column that is entirely NaN -- carries no signal and will trip
# sklearn's "input contains NaN" check.
all_nan = [c for c in feature_frame.columns if feature_frame[c].isna().all()]
if all_nan:
    print(f"Dropped {len(all_nan)} all-NaN column(s)")
    feature_frame = feature_frame.drop(columns=all_nan)

# The remaining gaps are STRUCTURAL, not random, and must not be median-filled:
#
#   - 829 addresses (8.4%) have a whitespace-only cell in every ERC20 column.
#     They simply never touched an ERC20 token, so "no activity" means zero.
#     Median-filling would invent typical token volumes for addresses that
#     did none -- fabricating exactly the behaviour we're trying to detect.
#   - 5,442 addresses (55.3%) have no "most sent/rec token type". Already
#     encoded above as a distinct __missing__ category.
#
# So ERC20 quantities get 0 (true zero) and everything else falls back to the
# median as a last resort.
erc20_cols = [c for c in feature_frame.columns if 'erc20' in c.lower()]
# Count BEFORE filling, otherwise the message reports the post-fill total (0).
zero_filled = int(feature_frame[erc20_cols].isna().sum().sum()) if erc20_cols else 0
for c in erc20_cols:
    feature_frame[c] = feature_frame[c].fillna(0)
if zero_filled:
    print(f"Filled {zero_filled} ERC20 gap(s) with 0 "
          f"(addresses with no ERC20 activity)")

residual = int(feature_frame.isna().sum().sum())
if residual:
    print(f"Filling {residual} residual NaN cell(s) with the column median")
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))

# ── 1. Dataset Overview ──
print(f"\n{'-'*50}")
print("  1. DATASET OVERVIEW")
print(f"{'-'*50}")

total = len(df)
fraud = df[df['FLAG'] == 1]
legit = df[df['FLAG'] == 0]
fraud_rate = len(fraud) / total * 100

print(f"Total addresses analysed: {total:,}")
print(f"Unique addresses: {df['Address'].nunique():,}")
print(f"Numeric behavioural features: {feature_frame.shape[1]}")
print(f"\nFlagged as fraudulent: {len(fraud):,} ({fraud_rate:.2f}%)")
print(f"Legitimate: {len(legit):,}")

# ── 2. Behavioural Analysis ──
# The point of this benchmark: what actually separates a fraudulent Ethereum
# address from a normal one? These are the aggregate signals a monitoring
# engine would key on.
print(f"\n{'-'*50}")
print("  2. BEHAVIOURAL ANALYSIS (mean by class)")
print(f"{'-'*50}")

key_features = [c for c in [
    'Sent tnx', 'Received tnx', 'Number of Created Contracts',
    'Unique Received From Addresses', 'Unique Sent To Addresses',
    'Avg min between sent tnx', 'Avg min between received tnx',
    'total Ether sent', 'total ether received'
] if c in df.columns]

comparison = df.groupby('FLAG')[key_features].mean().T
comparison.columns = ['legitimate_mean', 'fraud_mean']
comparison['ratio_fraud_over_legit'] = (
    comparison['fraud_mean'] / comparison['legitimate_mean'].replace(0, np.nan)
)
print(comparison.round(2).to_string())

# ── 3. Contract-Creation Signal ──
# Contract creation is the single strongest fraud marker in this dataset, so
# it's worth reporting on its own rather than only inside the model.
if 'Number of Created Contracts' in df.columns:
    print(f"\n{'-'*50}")
    print("  3. CONTRACT CREATION SIGNAL")
    print(f"{'-'*50}")
    creators = df[df['Number of Created Contracts'] > 0]
    creator_fraud = creators[creators['FLAG'] == 1]
    print(f"Addresses creating >=1 contract: {len(creators):,} "
          f"({len(creators) / total * 100:.2f}% of all addresses)")
    print(f"Of those, flagged fraudulent: {len(creator_fraud):,} "
          f"({len(creator_fraud) / max(len(creators), 1) * 100:.2f}%)")
    print(f"Fraud rate among contract creators: "
          f"{len(creator_fraud) / max(len(creators), 1) * 100:.2f}%")
    print(f"Fraud rate among non-creators: "
          f"{(len(fraud) - len(creator_fraud)) / max(total - len(creators), 1) * 100:.2f}%")

# ── 4. ML Model ──
print(f"\n{'-'*50}")
print("  4. MACHINE LEARNING MODEL")
print(f"{'-'*50}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

X = feature_frame
y = df['FLAG']

# At 22% fraud this dataset is far LESS skewed than the other two, so no
# resampling is applied -- resampling would distort the ROC-AUC and make this
# benchmark non-comparable to the other two. The full dataset is used as-is.
print(f"Training on all {len(X):,} addresses ({len(fraud):,} fraudulent, {len(legit):,} legitimate)")
print("No resampling applied (fraud rate is not extreme enough to warrant it)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent'], output_dict=True)
tn, fp, fn_, tp = cm.ravel()

print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))
print(f"Confusion Matrix (rows=true, cols=pred):")
print(cm)
print(f"Recall on fraud: {tp / max(tp + fn_, 1) * 100:.2f}% "
      f"({tp:,} of {tp + fn_:,} fraudulent addresses caught)")
print(f"Missed (false negatives): {fn_:,}")

importances = pd.DataFrame({
    'feature': feature_frame.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nTop 15 Feature Importances:")
print(importances.head(15).to_string(index=False))

# ── 5. Summary Results ──
print(f"\n{'='*60}")
print("  BENCHMARK RESULTS SUMMARY")
print(f"{'='*60}")

results = {
    'dataset': 'Ethereum Fraud Detection Dataset',
    'source': 'https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset',
    # Explicit unit labelling. These are addresses with aggregated behaviour,
    # NOT individual transactions -- calling them transactions would overstate
    # the analysed volume by orders of magnitude.
    'unit_of_analysis': 'address',
    'unit_note': 'Each row is one Ethereum address with aggregated behavioural features, not a single transaction.',
    'total_addresses': int(total),
    'fraud_cases': int(len(fraud)),
    'fraud_rate_percent': round(float(fraud_rate), 4),
    'unique_addresses': int(df['Address'].nunique()),
    'ml_mode': 'Random Forest (100 trees)',
    'roc_auc_score': round(float(roc_auc), 4),
    'precision_fraud': round(float(report['Fraudulent']['precision']), 4),
    'recall_fraud': round(float(report['Fraudulent']['recall']), 4),
    'confusion_matrix': [[int(tn), int(fp)], [int(fn_), int(tp)]],
    'processing_time_seconds': round(load_time + train_time, 2),
    # Probed, not asserted, so this can never claim hardware that wasn't used.
    'gpu_accelerated': False,  # scikit-learn estimators here are CPU-only
    'gpu_present': gpu_present,
    'gpu_name': gpu_name,
    'hardware_note': 'Kaggle notebook runtime; CPU-based scikit-learn estimators (no GPU acceleration used)',
    'top_features': importances.head(15).to_dict('records')
}

with open('/kaggle/working/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print(f"\n✅ Results saved to /kaggle/working/results.json")
print(f"✅ Notebook complete")