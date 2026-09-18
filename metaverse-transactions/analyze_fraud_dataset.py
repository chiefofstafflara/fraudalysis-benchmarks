#!/usr/bin/env python3
"""Fraudalysis - Metaverse Financial Transactions Dataset Analysis"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime

CSV_PATH = '/home/ubuntu/metaverse_transactions.csv'

def load_data():
    rows = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['amount'] = float(row['amount'])
            row['risk_score'] = float(row['risk_score'])
            try:
                row['hour_of_day'] = int(row['hour_of_day'])
            except:
                row['hour_of_day'] = 0
            rows.append(row)
    return rows

def analyze(rows):
    total = len(rows)
    print(f"{'='*60}")
    print(f"  FRAUDALYSIS — DATASET ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"\n📊  Total transactions: {total:,}")
    
    # 1. Transaction types
    types = Counter(r['transaction_type'] for r in rows)
    print(f"\n{'─'*50}")
    print("  TRANSACTION TYPE BREAKDOWN")
    print(f"{'─'*50}")
    for t, count in types.most_common():
        pct = count / total * 100
        print(f"  {t:>15s}: {count:>6,} ({pct:5.2f}%)")
    
    # 2. Anomaly (fraud) distribution
    anomalies = Counter(r['anomaly'] for r in rows)
    print(f"\n{'─'*50}")
    print("  ANOMALY (FRAUD) DISTRIBUTION")
    print(f"{'─'*50}")
    for a, count in anomalies.most_common():
        pct = count / total * 100
        print(f"  {a:>15s}: {count:>6,} ({pct:5.2f}%)")
    
    fraud_rows = [r for r in rows if r['anomaly'] == 'high_risk']
    legit_rows = [r for r in rows if r['anomaly'] == 'low_risk']
    
    # 3. Fraud by transaction type
    print(f"\n{'─'*50}")
    print("  FRAUD BY TRANSACTION TYPE")
    print(f"{'─'*50}")
    fraud_by_type = Counter(r['transaction_type'] for r in fraud_rows)
    for t, count in fraud_by_type.most_common():
        total_of_type = types[t]
        pct = count / total_of_type * 100 if total_of_type else 0
        print(f"  {t:>15s}: {count:>5,} / {total_of_type:>6,} ({pct:5.2f}% of this type)")
    
    # 4. Amount analysis
    amounts_all = [r['amount'] for r in rows]
    amounts_fraud = [r['amount'] for r in fraud_rows]
    
    def amt_stats(amounts, label):
        if not amounts:
            return
        sorted_amt = sorted(amounts)
        n = len(sorted_amt)
        print(f"\n  {label} Amounts:")
        print(f"    Min:       {sorted_amt[0]:>12.2f}")
        print(f"    Max:       {sorted_amt[-1]:>12.2f}")
        print(f"    Mean:      {sum(sorted_amt)/n:>12.2f}")
        print(f"    Median:    {sorted_amt[n//2]:>12.2f}")
        print(f"    Top 1%:    {sorted_amt[-int(n*0.01)]:>12.2f}")
    
    amt_stats(amounts_all, "All")
    amt_stats(amounts_fraud, "Fraud (high_risk)")
    
    # 5. Risk score distribution for fraud vs legit
    print(f"\n{'─'*50}")
    print("  RISK SCORE ANALYSIS")
    print(f"{'─'*50}")
    risk_fraud = [r['risk_score'] for r in fraud_rows]
    risk_legit = [r['risk_score'] for r in legit_rows]
    print(f"  Fraud avg risk score:     {sum(risk_fraud)/len(risk_fraud):.2f}" if risk_fraud else "  No fraud data")
    print(f"  Legit avg risk score:     {sum(risk_legit)/len(risk_legit):.2f}" if risk_legit else "  No legit data")
    
    # 6. Fraud by hour of day
    print(f"\n{'─'*50}")
    print("  FRAUD BY HOUR OF DAY (top 5 risky hours)")
    print(f"{'─'*50}")
    hour_fraud = Counter(r['hour_of_day'] for r in fraud_rows)
    hour_all = Counter(r['hour_of_day'] for r in rows)
    hour_risk = {}
    for h in range(24):
        f = hour_fraud.get(h, 0)
        a = hour_all.get(h, 0)
        if a > 0:
            hour_risk[h] = f / a * 100
    for h, risk in sorted(hour_risk.items(), key=lambda x: -x[1])[:5]:
        f = hour_fraud.get(h, 0)
        a = hour_all.get(h, 0)
        print(f"  Hour {h:02d}:00 — {risk:5.2f}% fraud rate ({f}/{a} txns)")
    
    # 7. Fraud by purchase pattern
    print(f"\n{'─'*50}")
    print("  FRAUD BY PURCHASE PATTERN")
    print(f"{'─'*50}")
    patterns = Counter(r['purchase_pattern'] for r in rows)
    fraud_patterns = Counter(r['purchase_pattern'] for r in fraud_rows)
    for p, count in patterns.most_common():
        f_count = fraud_patterns.get(p, 0)
        pct = f_count / count * 100 if count else 0
        print(f"  {p:>15s}: {f_count:>5,}/{count:>5,} fraud ({pct:5.2f}%)")
    
    # 8. Fraud by age group
    print(f"\n{'─'*50}")
    print("  FRAUD BY AGE GROUP")
    print(f"{'─'*50}")
    ages = Counter(r['age_group'] for r in rows)
    fraud_ages = Counter(r['age_group'] for r in fraud_rows)
    for a, count in ages.most_common():
        f_count = fraud_ages.get(a, 0)
        pct = f_count / count * 100 if count else 0
        print(f"  {a:>15s}: {f_count:>5,}/{count:>5,} fraud ({pct:5.2f}%)")
    
    # 9. Top sending addresses involved in fraud
    print(f"\n{'─'*50}")
    print("  TOP 10 FRAUD SENDING ADDRESSES")
    print(f"{'─'*50}")
    fraud_senders = Counter(r['sending_address'] for r in fraud_rows)
    for addr, count in fraud_senders.most_common(10):
        print(f"  {addr[:20]}...  {count} fraud txns")
    
    # 10. Summary
    print(f"\n{'='*60}")
    print("  KEY INSIGHTS FOR FRAUDALYSIS")
    print(f"{'='*60}")
    
    # Find most fraudulent type
    worst_type = max(fraud_by_type.items(), key=lambda x: x[1]/types[x[0]]*100 if types[x[0]] else 0)
    print(f"\n  🔴 Most fraudulent txn type: '{worst_type[0]}' "
          f"({fraud_by_type[worst_type[0]]/types[worst_type[0]]*100:.1f}% fraud rate)")
    
    # Find highest fraud hour
    worst_hour = max(hour_risk.items(), key=lambda x: x[1])
    print(f"  🕐 Peak fraud hour: {worst_hour[0]:02d}:00 ({worst_hour[1]:.1f}% fraud rate)")
    
    # Fraud amount pattern
    if amounts_fraud and amounts_all:
        fraud_avg = sum(amounts_fraud)/len(amounts_fraud)
        all_avg = sum(amounts_all)/len(amounts_all)
        print(f"  💰 Avg fraud amount: {fraud_avg:.2f} vs avg all: {all_avg:.2f} "
              f"({fraud_avg/all_avg*100:.1f}% of average)")
    
    print(f"  📊 Overall fraud rate: {len(fraud_rows)/total*100:.3f}%")
    
    # Recommendation
    print(f"\n  🎯 RECOMMENDATION:")
    print(f"     This dataset is highly relevant for Fraudalysis testing.")
    print(f"     It contains blockchain addresses, multiple fraud types,")
    print(f"     and risk scoring — directly applicable patterns.")
    
if __name__ == '__main__':
    import sys
    print("Loading data...")
    data = load_data()
    analyze(data)