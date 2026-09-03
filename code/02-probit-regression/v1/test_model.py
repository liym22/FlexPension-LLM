#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate both Probit stages on the held-out sample.

Model 1 predicts participation. Model 2 predicts the pension channel for
cases predicted to participate. The script reports precision, recall, and F1.
"""

import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)

print("=" * 80)
print("Step 4: Probit model evaluation")
print("=" * 80)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
output_dir = PROJECT_ROOT / "data" / "probit" / "v1_results"

# Load test data.
reg = pd.read_csv(PROCESSED_DIR / 'regression_data.csv')
reg['hhid']  = reg['hhid'].astype(str)
reg['pline'] = reg['pline'].astype(str)

sample_30 = pd.read_csv(PROCESSED_DIR / 'sample_30.csv', dtype={'hhid': str, 'pline': str})

test_keys = set(zip(sample_30['hhid'], sample_30['pline'].astype(str)))
is_test   = reg.apply(lambda r: (r['hhid'], r['pline']) in test_keys, axis=1)
test_df   = reg[is_test].copy()
print(f"Held-out set: {len(test_df)} records")

# Load fitted models.
with open(output_dir / 'model1.pkl', 'rb') as f:
    m1 = pickle.load(f)
with open(output_dir / 'model2.pkl', 'rb') as f:
    m2 = pickle.load(f)

res1, cols1, age2_1 = m1['result'], m1['cols'], m1['add_age2']
res2, cols2, age2_2 = m2['result'], m2['cols'], m2['add_age2']

print(f"Model 1 predictors: {len(cols1)}")
print(f"Model 2 predictors: {len(cols2)}")


def prepare_X(df_in, cols, add_age2):
    """Prepare the design matrix in the fitted model's column order."""
    df = df_in.copy()
    # Convert model inputs to float.
    for c in df.columns:
        if hasattr(df[c], 'dtype') and str(df[c].dtype).startswith('Int'):
            df[c] = df[c].astype(float)

    if add_age2 and 'age' in df.columns:
        age_mean = df['age'].mean()    # The original run used this split-specific mean.
        df['age_c']    = df['age'] - age_mean
        df['age_c_sq'] = df['age_c'] ** 2
        df = df.drop(columns=['age'], errors='ignore')

    # Fill absent indicator columns with zero.
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    X = df[cols].copy().fillna(0).values
    return sm.add_constant(X, has_constant='add')


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: predict the enrollment action.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 1] Predict participation with Model 1...")

X_test1 = prepare_X(test_df, cols1, age2_1)
prob1    = res1.predict(X_test1)
pred1    = (prob1 >= 0.5).astype(int)

test_df = test_df.copy()
test_df['pred_decision_prob'] = prob1
test_df['pred_decision']      = pred1
test_df['true_decision']      = test_df['decision'].astype(float).fillna(-1).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: predict scheme type for predicted participants.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2] Predict pension channel for predicted participants...")

test_df['pred_type_prob'] = np.nan
test_df['pred_type']      = pd.NA
test_df['true_type']      = test_df['type'].astype(float)

insured_mask = test_df['pred_decision'] == 1
insured_sub  = test_df[insured_mask].copy()
print(f"  Predicted participants: {insured_mask.sum()}")

if len(insured_sub) > 0:
    X_test2 = prepare_X(insured_sub, cols2, age2_2)
    prob2    = res2.predict(X_test2)
    pred2    = (prob2 >= 0.5).astype(int)

    test_df.loc[insured_mask, 'pred_type_prob'] = prob2
    test_df.loc[insured_mask, 'pred_type']      = pred2


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: compute evaluation metrics.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] Calculate evaluation metrics...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Step 4 evaluation results")
report_lines.append("=" * 80)

# Evaluate the enrollment-action model.
valid1 = test_df[test_df['true_decision'].isin([0, 1])]
y_true1 = valid1['true_decision'].values
y_pred1 = valid1['pred_decision'].values

report_lines.append("\n-- Model 1: participation prediction --")
report_lines.append(f"Valid evaluation records: {len(valid1)}")
report_lines.append("\nConfusion matrix (rows=true, columns=predicted):")
cm1 = confusion_matrix(y_true1, y_pred1, labels=[0, 1])
report_lines.append("                 Non-participant  Participant")
report_lines.append(f"True non-participant {cm1[0,0]:8d} {cm1[0,1]:12d}")
report_lines.append(f"True participant     {cm1[1,0]:8d} {cm1[1,1]:12d}")

cr1 = classification_report(y_true1, y_pred1, target_names=['Non-participant (0)', 'Participant (1)'])
report_lines.append(f"\nClassification report:\n{cr1}")

p1 = precision_score(y_true1, y_pred1, zero_division=0)
r1 = recall_score(y_true1, y_pred1, zero_division=0)
f1 = f1_score(y_true1, y_pred1, zero_division=0)
report_lines.append(f"Precision (participant): {p1:.4f}")
report_lines.append(f"Recall    (participant): {r1:.4f}")
report_lines.append(f"F1-score  (participant): {f1:.4f}")

# Evaluate the scheme-type model.
report_lines.append("\n-- Model 2: pension-channel prediction --")
valid2 = test_df[insured_mask & test_df['true_type'].notna() &
                 test_df['pred_type'].notna()]
valid2 = valid2[valid2['true_type'].isin([0.0, 1.0])]

if len(valid2) > 0:
    y_true2 = valid2['true_type'].astype(int).values
    y_pred2 = valid2['pred_type'].astype(int).values

    report_lines.append(f"Valid evaluation records: {len(valid2)}")
    cm2 = confusion_matrix(y_true2, y_pred2, labels=[0, 1])
    report_lines.append("\nConfusion matrix (rows=true, columns=predicted):")
    report_lines.append("                    Resident  Employee")
    report_lines.append(f"True resident       {cm2[0,0]:8d} {cm2[0,1]:9d}")
    report_lines.append(f"True employee       {cm2[1,0]:8d} {cm2[1,1]:9d}")

    # Explicit labels keep small single-class subsets evaluable.
    cr2 = classification_report(y_true2, y_pred2, labels=[0, 1], zero_division=0)
    report_lines.append(f"\nClassification report:\n{cr2}")

    p2 = precision_score(y_true2, y_pred2, zero_division=0)
    r2 = recall_score(y_true2, y_pred2, zero_division=0)
    f2_score = f1_score(y_true2, y_pred2, zero_division=0)
    report_lines.append(f"Precision (employee): {p2:.4f}")
    report_lines.append(f"Recall    (employee): {r2:.4f}")
    report_lines.append(f"F1-score  (employee): {f2_score:.4f}")
else:
    report_lines.append("  No valid records to evaluate.")
    f2_score = 0.0  # Define the composite component when no type cases are evaluable.

# End-to-end hierarchical evaluation.
report_lines.append("\n-- End-to-end evaluation (decision + type) --")
# Three-class labels: non-participant = 0, employee = 1, resident = 2.
def combined_label(row):
    if row['pred_decision'] == 0:
        return 'pred_uninsured'
    elif row['pred_type'] == 1:
        return 'pred_employee'
    else:
        return 'pred_resident'

def true_combined_label(row):
    if row['true_decision'] == 0:
        return 'true_uninsured'
    elif row['true_type'] == 1.0:
        return 'true_employee'
    elif row['true_type'] == 0.0:
        return 'true_resident'
    else:
        return 'unknown'

valid_all = test_df[test_df['true_decision'].isin([0, 1])].copy()
valid_all['pred_label'] = valid_all.apply(combined_label, axis=1)
valid_all['true_label'] = valid_all.apply(true_combined_label, axis=1)

both_uninsured   = ((valid_all['pred_label'] == 'pred_uninsured') &
                    (valid_all['true_label'] == 'true_uninsured')).sum()
both_employee    = ((valid_all['pred_label'] == 'pred_employee')  &
                    (valid_all['true_label'] == 'true_employee')).sum()
both_resident    = ((valid_all['pred_label'] == 'pred_resident')  &
                    (valid_all['true_label'] == 'true_resident')).sum()
total            = len(valid_all)
correct          = both_uninsured + both_employee + both_resident

report_lines.append(f"Total records:              {total}")
report_lines.append(f"Fully correct predictions:  {correct} ({correct/total*100:.1f}%)")
report_lines.append(f"  Correct non-participant:  {both_uninsured}")
report_lines.append(f"  Correct employee:         {both_employee}")
report_lines.append(f"  Correct resident:         {both_resident}")

# Weighted composite metric.
comprehensive_f1 = 0.6 * f1 + 0.4 * f2_score
report_lines.append("\n-- Composite benchmark metrics --")
report_lines.append(f"Stage 1 F1-score (weight 0.6): {f1:.4f}")
report_lines.append(f"Stage 2 F1-score (weight 0.4): {f2_score:.4f}")
report_lines.append(f"Composite F1-score:           {comprehensive_f1:.4f}")
# ============================================================

report_text = "\n".join(report_lines)
print(report_text)

result_path = output_dir / 'test_results.txt'
with open(result_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"\nSaved evaluation results: {result_path}")

# Save detailed predictions.
detail_path = output_dir / 'test_predictions.csv'
cols_to_save = ['hhid', 'pline', 'true_decision', 'pred_decision',
                'pred_decision_prob', 'true_type', 'pred_type', 'pred_type_prob']
test_df[[c for c in cols_to_save if c in test_df.columns]].to_csv(
    detail_path, index=False, encoding='utf-8-sig')
print(f"Saved detailed predictions: {detail_path}")

print("\n" + "=" * 80)
print("Step 4 complete.")
print("=" * 80)
