#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Filter and summarize the eligible CHFS 2019 flexible-worker sample."""

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chfs2019"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CHFS 2019 flexible-worker sample analysis")
print("=" * 80)

# 1. Load source files.
print("\n[Step 1] Loading data files...")
try:
    # Individual table.
    ind_data, ind_meta = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019_ind_202112.dta'))
    print(f"Loaded individual table: {ind_data.shape[0]} rows, {ind_data.shape[1]} columns")
    
    # Master table with derived variables.
    master_data, master_meta = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019_master_202112.dta'))
    print(f"Loaded master table: {master_data.shape[0]} rows, {master_data.shape[1]} columns")
    
except Exception as e:
    print(f"Failed to load data files: {e}")
    exit(1)

# 2. Preprocess and filter the sample.
print("\n[Step 2] Applying sample filters...")

# 2.1 Calculate age.
ind_data['age'] = 2019 - ind_data['a2005']

# 2.2 Apply flexible-worker inclusion criteria.
# Criterion 1: working age, male [16, 60), female [16, 55).
age_condition = (
    ((ind_data['a2003'] == 1) & (ind_data['age'] >= 16) & (ind_data['age'] < 60)) |  # Male.
    ((ind_data['a2003'] == 2) & (ind_data['age'] >= 16) & (ind_data['age'] < 55))    # Female.
)

# Criterion 2: currently working.
work_condition = (ind_data['a3132b'] == 1)

# Criterion 3: flexible-employment code in {2, 3, 4, 5, 6}.
# Codes denote temporary, employer, self-employed, family helper, and freelance work.
flexible_work_condition = ind_data['a3132d'].isin([2, 3, 4, 5, 6])

# Combine all inclusion criteria.
final_condition = age_condition & work_condition & flexible_work_condition

# Select eligible respondents.
sampled_ind = ind_data[final_condition].copy()

print("\nFilter counts:")
print(f"  - Eligible age: {age_condition.sum()}")
print(f"  - Employed: {work_condition.sum()}")
print(f"  - Flexible work: {flexible_work_condition.sum()}")
print(f"  - All conditions: {final_condition.sum()}")

# 3. Extract respondent identifiers.
print("\n[Step 3] Extracting sample identifiers...")
sample_ids = sampled_ind[['hhid', 'pline']].copy()
print(f"Extracted {len(sample_ids)} sample records")

# 4. Join household-level fields from the master table.
print("\n[Step 4] Merging master-table data...")
sampled_with_master = sampled_ind.merge(
    master_data[['hhid', 'rural', 'region', 'total_income']], 
    on='hhid', 
    how='left'
)
print(f"Merge complete: {len(sampled_with_master)} records")

# 5. Generate sample statistics.
print("\n" + "=" * 80)
print("Statistical analysis report")
print("=" * 80)

# 5.1 Sample size.
print(f"\nEligible sample size: {len(sampled_ind)}")
print(f"  - Households: {sampled_ind['hhid'].nunique()}")

# 5.2 Gender distribution.
print("\nGender distribution:")
gender_dist = sampled_ind['a2003'].value_counts().sort_index()
gender_labels = {1: 'male', 2: 'female'}
for gender, count in gender_dist.items():
    label = gender_labels.get(gender, f'unknown ({gender})')
    pct = count / len(sampled_ind) * 100
    print(f"  {label}: {count} ({pct:.2f}%)")

# 5.3 Employment-type distribution.
print("\nEmployment-type distribution:")
work_type_dist = sampled_ind['a3132d'].value_counts().sort_index()
work_type_labels = {
    2: 'temporary work',
    3: 'employer',
    4: 'self-employed work',
    5: 'family helper',
    6: 'freelance work'
}
for work_type, count in work_type_dist.items():
    label = work_type_labels.get(work_type, f'other ({work_type})')
    pct = count / len(sampled_ind) * 100
    print(f"  {label}: {count} ({pct:.2f}%)")

# 5.4 Urban-rural distribution.
print("\nUrban-rural distribution:")
if 'rural' in sampled_with_master.columns:
    rural_dist = sampled_with_master['rural'].value_counts().sort_index()
    rural_labels = {0: 'urban', 1: 'rural'}
    for rural_type, count in rural_dist.items():
        if pd.notna(rural_type):
            label = rural_labels.get(rural_type, f'unknown ({rural_type})')
            pct = count / len(sampled_with_master) * 100
            print(f"  {label}: {count} ({pct:.2f}%)")
    missing_rural = sampled_with_master['rural'].isna().sum()
    if missing_rural > 0:
        print(f"  Missing: {missing_rural}")
else:
    print("  Field 'rural' not found in the master table")

# 5.5 Regional distribution.
print("\nRegion distribution:")
if 'region' in sampled_with_master.columns:
    region_dist = sampled_with_master['region'].value_counts().sort_index()
    region_labels = {1: 'east', 2: 'central', 3: 'west'}
    for region_type, count in region_dist.items():
        if pd.notna(region_type):
            label = region_labels.get(region_type, f'region {region_type}')
            pct = count / len(sampled_with_master) * 100
            print(f"  {label}: {count} ({pct:.2f}%)")
    missing_region = sampled_with_master['region'].isna().sum()
    if missing_region > 0:
        print(f"  Missing: {missing_region}")
else:
    print("  Field 'region' not found in the master table")

# 5.6 Household-income distribution and terciles.
print("\nHousehold total-income distribution:")
if 'total_income' in sampled_with_master.columns:
    income_data = sampled_with_master['total_income'].dropna()
    
    if len(income_data) > 0:
        print(f"  Valid records: {len(income_data)}")
        print(f"  Mean: {income_data.mean():,.2f} yuan")
        print(f"  Median: {income_data.median():,.2f} yuan")
        print(f"  Standard deviation: {income_data.std():,.2f} yuan")
        print(f"  Minimum: {income_data.min():,.2f} yuan")
        print(f"  Maximum: {income_data.max():,.2f} yuan")
        
        # Compute income tercile cutoffs.
        print("\n  Income terciles:")
        quantile_33 = income_data.quantile(1/3)
        quantile_67 = income_data.quantile(2/3)
        print(f"    33.3% quantile: {quantile_33:,.2f} yuan")
        print(f"    66.7% quantile: {quantile_67:,.2f} yuan")
        
        # Summarize income tercile groups.
        print("\n  Household total-income groups:")
        low_income = (income_data <= quantile_33).sum()
        mid_income = ((income_data > quantile_33) & (income_data <= quantile_67)).sum()
        high_income = (income_data > quantile_67).sum()
        
        print(f"    Low (<= {quantile_33:,.2f} yuan): {low_income} ({low_income/len(income_data)*100:.2f}%)")
        print(f"    Middle ({quantile_33:,.2f}-{quantile_67:,.2f} yuan): {mid_income} ({mid_income/len(income_data)*100:.2f}%)")
        print(f"    High (> {quantile_67:,.2f} yuan): {high_income} ({high_income/len(income_data)*100:.2f}%)")
    else:
        print("  All total_income values are missing")
    
    missing_income = sampled_with_master['total_income'].isna().sum()
    if missing_income > 0:
        print(f"  Missing: {missing_income}")
else:
    print("  Field 'total_income' not found in the master table")

# 6. Save sampling artifacts.
print("\n" + "=" * 80)
print("[Step 5] Saving sampling results...")

# Save respondent IDs.
sample_ids.to_csv(PROCESSED_DIR / 'sampled_ids.csv', index=False, encoding='utf-8-sig')
print("Sample ID list saved to: sampled_ids.csv")

# Save the sampled master-table fields.
sampled_with_master.to_csv(PROCESSED_DIR / 'sampled_data_full.csv', index=False, encoding='utf-8-sig')
print("Full sampled data saved to: sampled_data_full.csv")

# Save the sample summary.
with (PROCESSED_DIR / 'sampling_report.txt').open('w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("CHFS 2019 Flexible-Worker Sample Analysis Report\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Eligible respondents: {len(sampled_ind)}\n")
    f.write(f"Distinct households: {sampled_ind['hhid'].nunique()}\n\n")
    
    f.write("Eligibility criteria:\n")
    f.write("  - Age: male [16, 60), female [16, 55)\n")
    f.write("  - Currently working: a3132b = 1\n")
    f.write("  - Flexible-employment codes: a3132d in {2, 3, 4, 5, 6}\n\n")
    
    if 'total_income' in sampled_with_master.columns:
        income_data = sampled_with_master['total_income'].dropna()
        if len(income_data) > 0:
            f.write("Household-income tercile thresholds:\n")
            f.write(f"  33.3rd percentile: {income_data.quantile(1/3):,.2f} CNY\n")
            f.write(f"  66.7th percentile: {income_data.quantile(2/3):,.2f} CNY\n")

print("Statistical report saved to: sampling_report.txt")

print("\n" + "=" * 80)
print("Sample analysis complete")
print("=" * 80)
