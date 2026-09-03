#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Draw 30 CHFS 2019 respondents using joint stratified sampling.

The strata combine rural status, region, and household-income tercile. Each
selected respondent comes from a distinct household.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chfs2019"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# Global seed for deterministic NumPy and pandas sampling.
# -----------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("CHFS 2019 second-stage stratified sample")
print("=" * 80)

# 1. Load filtered IDs and source tables.
print("\n[Step 1] Load data...")
try:
    import pyreadstat
    
    # Filtered respondent IDs.
    sampled_ids = pd.read_csv(PROCESSED_DIR / 'sampled_ids_filtered.csv', dtype={'hhid': str})
    print(f"✓ Loaded {len(sampled_ids)} filtered respondent IDs")
    
    # Individual and master tables.
    ind_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019_ind_202112.dta'))
    ind_data['hhid'] = ind_data['hhid'].astype(str)
    master_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019_master_202112.dta'))
    master_data['hhid'] = master_data['hhid'].astype(str)
    
    # Join respondent and stratification fields.
    sampled_data = sampled_ids.merge(ind_data, on=['hhid', 'pline'], how='left')
    sampled_data = sampled_data.merge(master_data[['hhid', 'total_income', 'rural', 'region']], on='hhid', how='left')
    
    # Stable sorting removes merge-order dependence.
    sampled_data = sampled_data.sort_values(['hhid', 'pline'], kind='mergesort').reset_index(drop=True)
    
    print(f"✓ Records after merging: {len(sampled_data)}")
    print(f"✓ Distinct households: {sampled_data['hhid'].nunique()}")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    exit(1)

# 2. Preprocess the sampling frame.
print("\n[Step 2] Prepare the sampling frame...")

# Normalize household ID type.
sampled_data['hhid'] = sampled_data['hhid'].astype(str)

# Stable sorting followed by drop_duplicates keeps one coherent row per household.
# This avoids column-wise first-nonmissing behavior from groupby().first().
sampled_unique = (
    sampled_data
    .sort_values(['hhid', 'pline'], kind='mergesort')
    .drop_duplicates(subset='hhid', keep='first')
    .reset_index(drop=True)
)
print(f"✓ Retained {len(sampled_unique)} distinct households")

# Compute household-income terciles with explicit linear interpolation.
income_tercile_33 = sampled_unique['total_income'].quantile(1/3, interpolation='linear')
income_tercile_67 = sampled_unique['total_income'].quantile(2/3, interpolation='linear')
print("\nHousehold-income tercile thresholds:")
print(f"  33.3rd percentile: {income_tercile_33:,.2f} CNY")
print(f"  66.7th percentile: {income_tercile_67:,.2f} CNY")

# Assign income tercile groups.
def income_group(income):
    if pd.isna(income):
        return None
    elif income <= income_tercile_33:
        return 'low'
    elif income <= income_tercile_67:
        return 'mid'
    else:
        return 'high'

sampled_unique['income_tercile'] = sampled_unique['total_income'].apply(income_group)

# Remove records missing any stratification field.
sampled_unique = sampled_unique.dropna(subset=['income_tercile', 'rural', 'region'])
print(f"✓ Records after removing missing stratification fields: {len(sampled_unique)}")

# 3. Construct joint strata.
print("\n[Step 3] Construct joint strata...")
sampled_unique['stratum'] = (
    sampled_unique['rural'].astype(str) + '_' + 
    sampled_unique['region'].astype(str) + '_' + 
    sampled_unique['income_tercile']
)

# Count available respondents by stratum.
stratum_counts = sampled_unique['stratum'].value_counts().sort_index()
print("\nAvailable respondents by stratum:")
for stratum, count in stratum_counts.items():
    print(f"  {stratum}: {count}")

# 4. Draw the stratified sample.
print("\n[Step 4] Draw the stratified sample...")

# Count nonempty strata.
n_strata = len(stratum_counts)
target_sample_size = 30

# Allocate proportionally with at least one case per nonempty stratum.
stratum_sample_sizes = {}
total_available = len(sampled_unique)

# Compute initial allocations.
for stratum, count in stratum_counts.items():
    proportion = count / total_available
    allocated = max(1, round(proportion * target_sample_size))  # At least one.
    stratum_sample_sizes[stratum] = min(allocated, count)  # Do not exceed availability.

# Adjust allocations to exactly 30 cases.
current_total = sum(stratum_sample_sizes.values())
if current_total > target_sample_size:
    # Reduce allocations from larger strata.
    while current_total > target_sample_size:
        # Consider strata allocated more than one case.
        candidates = {s: stratum_sample_sizes[s] for s in stratum_sample_sizes 
                     if stratum_sample_sizes[s] > 1}
        if not candidates:
            break
        # Break ties by stratum name for deterministic ordering.
        max_stratum = max(candidates, key=lambda s: (candidates[s], s))
        stratum_sample_sizes[max_stratum] -= 1
        current_total -= 1
elif current_total < target_sample_size:
    # Increase allocations in strata with remaining capacity.
    while current_total < target_sample_size:
        # Consider strata with unallocated cases.
        candidates = {s: stratum_counts[s] - stratum_sample_sizes[s] 
                     for s in stratum_sample_sizes 
                     if stratum_sample_sizes[s] < stratum_counts[s]}
        if not candidates:
            break
        # Break ties by stratum name for deterministic ordering.
        max_stratum = max(candidates, key=lambda s: (candidates[s], s))
        stratum_sample_sizes[max_stratum] += 1
        current_total += 1

print("\nSample allocation by stratum:")
for stratum in sorted(stratum_sample_sizes.keys()):
    print(f"  {stratum}: {stratum_sample_sizes[stratum]}/{stratum_counts[stratum]}")

# Draw the stratified random sample.
# Iterate by sorted stratum name to stabilize concatenation order.
# Sort each stratum by household ID before sampling.
# Use the fixed random state independently in each stratum.
sampled_list = []

for stratum, n_samples in sorted(stratum_sample_sizes.items()):
    stratum_data = (
        sampled_unique[sampled_unique['stratum'] == stratum]
        .sort_values('hhid', kind='mergesort')
    )
    if len(stratum_data) >= n_samples:
        sampled_stratum = stratum_data.sample(n=n_samples, random_state=RANDOM_SEED)
        sampled_list.append(sampled_stratum)

final_sample = pd.concat(sampled_list, ignore_index=True)
print(f"\n✓ Drew {len(final_sample)} respondents")

# 5. Validate the sample.
print("\n[Step 5] Validate the sample...")

# Verify one respondent per household.
hhid_unique = final_sample['hhid'].nunique()
print(f"✓ Household-ID uniqueness: {hhid_unique}/{len(final_sample)} (must match)")

if hhid_unique != len(final_sample):
    print("✗ WARNING: duplicate household IDs detected")
else:
    print("✓ Every respondent comes from a distinct household")

# 6. Print sample statistics.
print("\n" + "=" * 80)
print("Sample Summary")
print("=" * 80)

print(f"\nTotal respondents: {len(final_sample)}")

print("\nRural/urban distribution:")
rural_dist = final_sample['rural'].value_counts().sort_index()
for rural, count in rural_dist.items():
    label = 'urban' if rural == 0 else 'rural'
    print(f"  {label}: {count} ({count/len(final_sample)*100:.1f}%)")

print("\nRegional distribution:")
region_dist = final_sample['region'].value_counts().sort_index()
region_labels = {1: 'east', 2: 'central', 3: 'west', 4: 'other'}
for region, count in region_dist.items():
    label = region_labels.get(region, f'region {region}')
    print(f"  {label}: {count} ({count/len(final_sample)*100:.1f}%)")

print("\nIncome-tercile distribution:")
income_dist = final_sample['income_tercile'].value_counts()
income_labels = {'low': 'low income', 'mid': 'middle income', 'high': 'high income'}
for income_level in ['low', 'mid', 'high']:
    if income_level in income_dist:
        count = income_dist[income_level]
        print(f"  {income_labels[income_level]}: {count} ({count/len(final_sample)*100:.1f}%)")

print("\nEmployment-type distribution:")
work_type_dist = final_sample['a3132d'].value_counts().sort_index()
work_type_labels = {
    2: 'temporary work',
    3: 'employer',
    4: 'self-employed work',
    5: 'family helper',
    6: 'freelance work'
}
for work_type, count in work_type_dist.items():
    label = work_type_labels.get(work_type, f'type {work_type}')
    print(f"  {label}: {count} ({count/len(final_sample)*100:.1f}%)")

print("\nAge summary:")
print(f"  Mean: {final_sample['a2005'].apply(lambda x: 2019-x).mean():.1f} years")
print(f"  Minimum: {final_sample['a2005'].apply(lambda x: 2019-x).min()} years")
print(f"  Maximum: {final_sample['a2005'].apply(lambda x: 2019-x).max()} years")

print("\nGender distribution:")
gender_dist = final_sample['a2003'].value_counts().sort_index()
for gender, count in gender_dist.items():
    label = 'male' if gender == 1 else 'female'
    print(f"  {label}: {count} ({count/len(final_sample)*100:.1f}%)")

# 7. Save sampling artifacts.
print("\n" + "=" * 80)
print("[Step 6] Save sampling artifacts...")

# Save respondent IDs only.
sample_30_ids = final_sample[['hhid', 'pline']].copy()
sample_30_ids.to_csv(PROCESSED_DIR / 'sample_30.csv', index=False, encoding='utf-8-sig')
print("✓ Respondent IDs saved to sample_30.csv")

# Save the sampling report.
with (PROCESSED_DIR / 'sample_30_summary.txt').open('w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("CHFS 2019 Second-Stage Stratified Sampling Report\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Sampling Method\n")
    f.write("- Stratification: rural x region x income_tercile\n")
    f.write(f"- Number of nonempty strata: {n_strata}\n")
    f.write("- Target sample size: 30\n")
    f.write("- Strategy: proportional stratified random sampling with distinct hhid values\n\n")
    
    f.write("Sample Statistics\n")
    f.write(f"- Respondents: {len(final_sample)}\n")
    f.write(f"- Distinct households: {hhid_unique}\n")
    f.write(f"- Mean age: {final_sample['a2005'].apply(lambda x: 2019-x).mean():.1f} years\n\n")
    
    f.write("Sampled Respondents by Stratum\n")
    for stratum in sorted(stratum_sample_sizes.keys()):
        count = len(final_sample[final_sample['stratum'] == stratum])
        f.write(f"  {stratum}: {count}\n")

print("✓ Sampling report saved to sample_30_summary.txt")

print("\n" + "=" * 80)
print("Second-stage stratified sampling complete")
print("=" * 80)
