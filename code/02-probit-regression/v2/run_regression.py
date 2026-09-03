#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit version 2 of the two-stage Probit specifications.

Relative to version 1, this version replaces 31 hukou-province indicators
with regional indicators and province-level policy variables. The 30-case
screening sample is excluded from both stages.
"""

import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from pathlib import Path
from scipy import stats

print("=" * 80)
print("Step 3 (v2): Probit regression with regional and policy variables")
print("=" * 80)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
output_dir = PROJECT_ROOT / "data" / "probit" / "v2_results"
output_dir.mkdir(parents=True, exist_ok=True)

# Load regression data.
reg = pd.read_csv(PROCESSED_DIR / 'regression_data.csv')
print(f"Loaded regression_data.csv: {len(reg)} records, {len(reg.columns)} columns")

sample_30 = pd.read_csv(PROCESSED_DIR / 'sample_30.csv', dtype={'hhid': str, 'pline': str})
reg['hhid']  = reg['hhid'].astype(str)
reg['pline'] = reg['pline'].astype(str)

# Exclude the 30-case baseline screening sample.
test_keys  = set(zip(sample_30['hhid'], sample_30['pline'].astype(str)))
is_test    = reg.apply(lambda r: (r['hhid'], r['pline']) in test_keys, axis=1)
train_df   = reg[~is_test].copy()
print(f"Training set: {len(train_df)} records ({is_test.sum()} held-out records excluded)")

# Define predictors using region and policy variables instead of province indicators.
BASE_VARS = [
    'age',
    'gender', 'education', 'health',
    # Hukou-type indicators.
    'hukou_agri', 'hukou_uni', 'hukou_else',
    # Province indicators are omitted.
    # Migration.
    'floating',
    # Employment type, with helpers/farmers as reference.
    'job_linshigong', 'job_guzhu', 'job_ziying', 'job_ziyouzhiye',
    # Industry, with agriculture as reference.
    'industry_mining', 'industry_constr', 'industry_utility', 'industry_retail',
    'industry_transport', 'industry_hotel', 'industry_it', 'industry_finance',
    'industry_realestate', 'industry_education', 'industry_service',
    'industry_gov', 'industry_other', 'industry_unknown',
    # Occupation, with managers as reference.
    'occu_professional', 'occu_clerk', 'occu_delivery', 'occu_service',
    'occu_production', 'occu_other', 'occu_unknown',
    # Employer type, with public institutions as reference.
    'employer_soe', 'employer_individual', 'employer_private',
    'employer_foreign', 'employer_other', 'employer_unknown',
    # Income and assets.
    'ln_ind_income', 'ln_hh_income', 'ln_hh_consump', 'ln_hh_asset', 'ln_hh_liabi',
    # Household composition.
    'hh_num', 'hh_child', 'hh_old',
    # Enrollment history.
    'contribution_years', 'has_contribution_history',
    # Household enrollment.
    'hh_pay_num', 'hh_receive_num', 'hh_pension',
    # Behavioral variables.
    'risk_preference', 'econ_expectation',
    # Rural residence.
    'rural',
    # Region indicators, with eastern China as reference.
    'region_central', 'region_west', 'region_northeast',
    # Province-level policy variables.
    'ln_avg_wage', 'zhigong_burden', 'jumin_burden', 'ln_pension',
]
# Retain predictors present in the transformed data.
BASE_VARS = [c for c in BASE_VARS if c in reg.columns]
print(f"Predictor count: {len(BASE_VARS)}")


def drop_perfect_collinear(X):
    """Remove perfectly collinear and zero-variance columns."""
    removed = []
    while True:
        zero_var = X.columns[X.std() == 0].tolist()
        if zero_var:
            X = X.drop(columns=zero_var)
            removed.extend(zero_var)
            continue
        break
    if removed:
        print(f"  Removed zero-variance columns: {removed}")
    return X


def run_probit(df_train, y_col, x_cols, label):
    """Fit a Probit model and test the quadratic age specification."""
    print(f"\n{'─'*60}")
    print(f"  {label}")

    # Remove rows with missing outcomes.
    df_clean = df_train[x_cols + [y_col]].dropna(subset=[y_col]).copy()

    # Convert nullable numeric columns to float.
    for c in df_clean.columns:
        if hasattr(df_clean[c], 'dtype') and str(df_clean[c].dtype).startswith('Int'):
            df_clean[c] = df_clean[c].astype(float)

    df_clean = df_clean.dropna()

    y = df_clean[y_col].values
    X_df = df_clean[x_cols].copy()
    X_df = drop_perfect_collinear(X_df)
    used_cols = X_df.columns.tolist()
    X = sm.add_constant(X_df, has_constant='add')  # Preserve names in model summaries.

    print(f"  Samples: {len(y)}; predictors: {len(used_cols)}")

    # Initial fit without the squared-age term.
    try:
        mdl = sm.Probit(y, X)
        res = mdl.fit(method='bfgs', maxiter=500, disp=False)
    except Exception as e:
        print(f"  BFGS failed; trying Newton: {e}")
        try:
            res = sm.Probit(y, X).fit(method='newton', maxiter=500, disp=False)
        except Exception as e2:
            print(f"  Newton also failed: {e2}")
            return None, used_cols, False

    # Test for a U-shaped age effect.
    add_age2 = False
    if 'age' in used_cols:
        resid = y - res.predict()
        age_vals = X_df['age'].values
        age_c    = age_vals - age_vals.mean()
        A        = np.column_stack([np.ones(len(age_c)), age_c, age_c**2])
        ols_res  = np.linalg.lstsq(A, resid, rcond=None)[0]
        resid_hat = A @ ols_res
        ss_res   = ((resid - resid_hat)**2).sum()
        dof      = len(resid) - 3
        se2_coef = ss_res / dof * np.linalg.inv(A.T @ A)[2, 2]
        t_stat   = ols_res[2] / (se2_coef**0.5) if se2_coef > 0 else 0
        p_val    = 2 * (1 - stats.t.cdf(abs(t_stat), df=dof))
        shape = 'U-shaped' if ols_res[2] > 0 else 'inverted U-shaped'
        print(f"  Quadratic-age test: coefficient={ols_res[2]:.4f}, t={t_stat:.2f}, p={p_val:.4f} -> {shape}")

        if p_val < 0.05:
            add_age2 = True
            print(f"  Age residuals are {shape}; refitting with centered age squared")
            X_df['age_c']    = age_c
            X_df['age_c_sq'] = age_c**2
            X_df = X_df.drop(columns=['age'])
            X_df = drop_perfect_collinear(X_df)
            used_cols = X_df.columns.tolist()
            X = sm.add_constant(X_df, has_constant='add')  # Preserve predictor names.
            try:
                res = sm.Probit(y, X).fit(method='bfgs', maxiter=500, disp=False)
            except Exception:
                res = sm.Probit(y, X).fit(method='newton', maxiter=500, disp=False)
        else:
            print(f"  Quadratic age is not significant (p={p_val:.4f}); retaining linear age")

    print(f"  Converged: {res.mle_retvals.get('converged', True)}")
    print(f"  Pseudo R2: {res.prsquared:.4f}  AIC: {res.aic:.2f}  BIC: {res.bic:.2f}")
    return res, used_cols, add_age2


# ════════════════════════════════════════════════════════════════════════════
# Model 1: predict the enrollment action.
# ════════════════════════════════════════════════════════════════════════════
print("\n[Model 1] Predict participation")

res1, cols1, age2_1 = run_probit(train_df, 'decision', BASE_VARS, 'Model 1: participation')

if res1 is not None:
    summary1_path = output_dir / 'model1_summary.txt'
    with open(summary1_path, 'w', encoding='utf-8') as f:
        f.write(str(res1.summary2()))
        f.write(f"\n\nPredictors used:\n{cols1}")
        f.write(f"\nQuadratic age included: {age2_1}")
        try:
            ame1 = res1.get_margeff()
            f.write("\n\n" + "=" * 60)
            f.write("\nAverage Marginal Effects\n")
            f.write(str(ame1.summary()))
        except Exception as e:
            f.write(f"\n\nAME calculation failed: {e}")
    print(f"Saved Model 1 summary: {summary1_path}")

    pkl1_path = output_dir / 'model1.pkl'
    with open(pkl1_path, 'wb') as f:
        pickle.dump({'result': res1, 'cols': cols1, 'add_age2': age2_1}, f)
    print(f"Saved Model 1: {pkl1_path}")

# ════════════════════════════════════════════════════════════════════════════
# Model 2: predict scheme type among participants.
# ════════════════════════════════════════════════════════════════════════════
print("\n[Model 2] Predict pension channel among participants")

train_insured = train_df[train_df['decision'] == 1].copy()
print(f"Participant training records: {len(train_insured)}")

res2, cols2, age2_2 = run_probit(train_insured, 'type', BASE_VARS, 'Model 2: pension channel')

if res2 is not None:
    summary2_path = output_dir / 'model2_summary.txt'
    with open(summary2_path, 'w', encoding='utf-8') as f:
        f.write(str(res2.summary2()))
        f.write(f"\n\nPredictors used:\n{cols2}")
        f.write(f"\nQuadratic age included: {age2_2}")
        try:
            ame2 = res2.get_margeff()
            f.write("\n\n" + "=" * 60)
            f.write("\nAverage Marginal Effects\n")
            f.write(str(ame2.summary()))
        except Exception as e:
            f.write(f"\n\nAME calculation failed: {e}")
    print(f"Saved Model 2 summary: {summary2_path}")

    pkl2_path = output_dir / 'model2.pkl'
    with open(pkl2_path, 'wb') as f:
        pickle.dump({'result': res2, 'cols': cols2, 'add_age2': age2_2}, f)
    print(f"Saved Model 2: {pkl2_path}")

print("\n" + "=" * 80)
print("Step 3 (v2) complete.")
print("=" * 80)
