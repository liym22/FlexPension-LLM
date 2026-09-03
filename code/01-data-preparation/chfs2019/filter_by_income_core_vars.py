#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Filter respondents with missing employment-specific income variables."""

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chfs2019"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Filter respondents by core personal-income variables")
print("=" * 80)

# 1. Load source data.
print("\n[Step 1] Load data...")
ind_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019_ind_202112.dta'))
hh_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019_hh_202112.dta'))
sampled_ids = pd.read_csv(PROCESSED_DIR / 'sampled_ids.csv')

print(f"✓ Individual records: {ind_data.shape[0]} rows")
print(f"✓ Household records: {hh_data.shape[0]} rows")
print(f"✓ Initial sampled respondents: {len(sampled_ids)}")

# 2. Join sampled respondents to individual and household tables.
print("\n[Step 2] Merge data...")
sampled_ids['hhid'] = sampled_ids['hhid'].astype(str)
sampled_ids['pline'] = sampled_ids['pline'].astype(int)
ind_data['hhid'] = ind_data['hhid'].astype(str)
ind_data['pline'] = ind_data['pline'].astype(int)
hh_data['hhid'] = hh_data['hhid'].astype(str)

sampled_ind = ind_data.merge(sampled_ids, on=['hhid', 'pline'], how='inner')
sampled_full = sampled_ind.merge(hh_data, on='hhid', how='left', suffixes=('', '_hh'))
print(f"✓ Respondents after merging: {len(sampled_full)}")

# 2.5 Filter valid pension-enrollment status codes.
print("\n[Step 2.5] Filter by pension-enrollment status (f1001a)...")

# Inspect the original f1001a distribution.
print("\nOriginal f1001a distribution:")
print(sampled_full['f1001a'].value_counts(dropna=False).sort_index())

# Valid pension-enrollment status codes.
valid_insurance_status = [1, 2, 3, 4, 5, 7788]

# Retain f1001a in {1, 2, 3, 4, 5, 7788}.
before_filter = len(sampled_full)
sampled_full = sampled_full[sampled_full['f1001a'].isin(valid_insurance_status)].copy()
after_filter = len(sampled_full)

print(f"\n✓ Enrollment-status filter: retained {after_filter} "
      f"(removed {before_filter - after_filter})")
print(f"  Retention rule: f1001a in {valid_insurance_status}")

# 2.6 For enrolled respondents, require at least one contribution field.
print("\n[Step 2.6] Remove enrolled respondents missing both contribution fields...")

mask_insured = sampled_full['f1001a'].isin([1, 2, 3, 4, 5])
mask_both_null = sampled_full['f1008_imp'].isna() & sampled_full['f1008a_imp'].isna()
mask_to_drop = mask_insured & mask_both_null

cnt_drop_f1008 = mask_to_drop.sum()
before_f1008_filter = len(sampled_full)
sampled_full = sampled_full[~mask_to_drop].copy()
after_f1008_filter = len(sampled_full)

print("  Removal rule: f1001a in [1, 2, 3, 4, 5] and both "
      "f1008_imp and f1008a_imp are missing")
print(f"✓ Removed {cnt_drop_f1008}; {after_f1008_filter} remain")

# 3. Apply employment-specific income completeness rules.
print("\n" + "=" * 80)
print("Filter by core income variables")
print("=" * 80)

# Track excluded respondents and reasons.
to_remove = []
removal_reasons = {}

work_type_mapping = {
    2: 'temporary worker',
    3: 'employer',
    4: 'self-employed worker',
    5: 'family helper',
    6: 'freelancer'
}

for idx, row in sampled_full.iterrows():
    work_type = row['a3132d']
    hhid = row['hhid']
    pline = row['pline']
    key = f"{hhid}_{pline}"
    
    if work_type == 2:  # Temporary worker.
        if pd.isna(row['a3136_imp']):
            to_remove.append(idx)
            removal_reasons[key] = "temporary worker: a3136_imp is missing"
    
    elif work_type in [3, 4]:  # Employer or self-employed worker.
        work_name = work_type_mapping[work_type]
        b2002 = row['b2002']
        
        # Branch on the number of business projects.
        if pd.notna(b2002):
            if b2002 > 1:  # Multiple projects.
                if pd.isna(row['b2003e_imp']):
                    to_remove.append(idx)
                    removal_reasons[key] = f"{work_name}: multiple projects but b2003e_imp is missing"
            elif b2002 == 1:  # Single project.
                # Check profit or loss status.
                b2054 = row['b2054']
                if pd.notna(b2054) and b2054 in [1, 2]:  # Profit or loss.
                    if pd.isna(row['b2055_imp']):
                        to_remove.append(idx)
                        project_result = 'profit' if b2054 == 1 else 'loss'
                        removal_reasons[key] = (
                            f"{work_name}: single project with {project_result}, "
                            "but b2055_imp is missing"
                        )
                # b2054 = 3 denotes break-even and does not require b2055_imp.
        else:  # Unknown project count.
            # Require at least one single- or multi-project income field.
            if pd.isna(row['b2055_imp']) and pd.isna(row['b2003e_imp']):
                to_remove.append(idx)
                removal_reasons[key] = (
                    f"{work_name}: project count is unknown and both "
                    "single- and multi-project income fields are missing"
                )
    
    elif work_type == 6:  # Freelancer.
        if pd.isna(row['a3136aa_imp']):
            to_remove.append(idx)
            removal_reasons[key] = "freelancer: a3136aa_imp is missing"
    
    # Family helpers need no income field because their individual income defaults to zero.

# 4. Apply exclusions.
print("\n[Step 3] Apply filters...")
print(f"Respondents flagged for removal: {len(to_remove)}")

if len(to_remove) > 0:
    print("\nRemoval-reason counts:")
    reason_counts = {}
    for reason in removal_reasons.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")
    
# Remove flagged respondents.
    sampled_filtered = sampled_full.drop(to_remove)
    print(f"\nRespondents after filtering: {len(sampled_filtered)} "
          f"(removed {len(to_remove)})")
else:
    sampled_filtered = sampled_full
    print("✓ No respondents need to be removed")

# 5. Save filtered respondent IDs.
print("\n[Step 4] Save filtered respondent IDs...")
filtered_ids = sampled_filtered[['hhid', 'pline']].copy()
filtered_ids.to_csv(PROCESSED_DIR / 'sampled_ids_filtered.csv', index=False, encoding='utf-8-sig')
print("✓ Filtered respondent IDs saved to sampled_ids_filtered.csv")

# 6. Generate the filtering report.
print("\n[Step 5] Generate filtering report...")
with (PROCESSED_DIR / 'filtering_report.txt').open('w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Core Personal-Income Variable Filtering Report\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Initial sample after enrollment-status filtering: {before_f1008_filter}\n")
    f.write(f"Removed in Step 2.6 (both contribution fields missing): {cnt_drop_f1008}\n")
    f.write(f"Removed in Step 3 (income variables missing): {len(to_remove)}\n")
    f.write(f"Final filtered sample: {len(sampled_filtered)}\n")
    f.write(f"Retention rate: {len(sampled_filtered)/before_f1008_filter*100:.2f}%\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("Removal-Reason Counts\n")
    f.write("=" * 80 + "\n\n")
    
    if len(to_remove) > 0:
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{reason}: {count} ({count/len(to_remove)*100:.2f}%)\n")
    else:
        f.write("No respondents removed\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("Counts by Employment Type\n")
    f.write("=" * 80 + "\n\n")
    
    for work_type_code, work_type_name in work_type_mapping.items():
        original = len(sampled_full[sampled_full['a3132d'] == work_type_code])
        filtered = len(sampled_filtered[sampled_filtered['a3132d'] == work_type_code])
        removed = original - filtered
        if original > 0:
            f.write(f"{work_type_name} (a3132d={work_type_code}):\n")
            f.write(f"  Initial: {original}\n")
            f.write(f"  Removed: {removed}\n")
            f.write(f"  Retained: {filtered} ({filtered/original*100:.2f}%)\n\n")

print("✓ Filtering report saved to filtering_report.txt")

# 7. Report the filtered employment-type distribution.
print("\n" + "=" * 80)
print("Filtered Sample Distribution")
print("=" * 80)

print("\nDistribution by employment type:")
work_dist = sampled_filtered['a3132d'].value_counts().sort_index()
for work_code, count in work_dist.items():
    work_name = work_type_mapping.get(work_code, f"unknown ({work_code})")
    print(f"  {work_name}: {count} ({count/len(sampled_filtered)*100:.2f}%)")

print("\n" + "=" * 80)
print("Data Integrity Checks")
print("=" * 80)

# Check 1: f1008_imp is present when f1001a = 2.
df_f1001a_2 = sampled_filtered[sampled_filtered['f1001a'] == 2]
missing_f1008 = df_f1001a_2['f1008_imp'].isna().sum()
total_f1001a_2 = len(df_f1001a_2)
if missing_f1008 == 0:
    print(f"\n✓ [PASS] All {total_f1001a_2} respondents with f1001a=2 have f1008_imp")
else:
    print(f"\n✗ [FAIL] {missing_f1008} of {total_f1001a_2} respondents with "
          "f1001a=2 have missing f1008_imp")
    problem_ids = df_f1001a_2[df_f1001a_2['f1008_imp'].isna()][['hhid', 'pline', 'f1001a', 'f1008_imp']]
    print(problem_ids.to_string(index=False))

# Check 2: f1008a_imp is present when f1001a is 3, 4, or 5.
df_f1001a_345 = sampled_filtered[sampled_filtered['f1001a'].isin([3, 4, 5])]
missing_f1008a = df_f1001a_345['f1008a_imp'].isna().sum()
total_f1001a_345 = len(df_f1001a_345)
if missing_f1008a == 0:
    print(f"\n✓ [PASS] All {total_f1001a_345} respondents with f1001a in [3, 4, 5] "
          "have f1008a_imp")
else:
    print(f"\n✗ [FAIL] {missing_f1008a} of {total_f1001a_345} respondents with "
          "f1001a in [3, 4, 5] have missing f1008a_imp")
    problem_ids = df_f1001a_345[df_f1001a_345['f1008a_imp'].isna()][['hhid', 'pline', 'f1001a', 'f1008a_imp']]
    print(problem_ids.to_string(index=False))

print("\n" + "=" * 80)

# Check 3: missingness patterns for f1009 and f1009a among enrolled respondents.
print("\n" + "=" * 80)
print("Data Integrity Checks (continued)")
print("=" * 80)

df_f1001a_2345 = sampled_filtered[sampled_filtered['f1001a'].isin([2, 3, 4, 5])]
total_f1001a_2345 = len(df_f1001a_2345)

if total_f1001a_2345 > 0:
    mask_f1009_notnull = df_f1001a_2345['f1009'].notna()
    mask_f1009a_notnull = df_f1001a_2345['f1009a'].notna()
    
    # Four start-year and contribution-year missingness patterns.
    only_f1009 = (mask_f1009_notnull & ~mask_f1009a_notnull).sum()
    only_f1009a = (~mask_f1009_notnull & mask_f1009a_notnull).sum()
    both_null = (~mask_f1009_notnull & ~mask_f1009a_notnull).sum()
    both_notnull = (mask_f1009_notnull & mask_f1009a_notnull).sum()
    
    print(f"\n[Check 3] Missingness of f1009/f1009a among {total_f1001a_2345} "
          "respondents with f1001a in [2, 3, 4, 5]:")
    print(f"  Only f1009 present : {only_f1009} ({only_f1009/total_f1001a_2345*100:.2f}%)")
    print(f"  Only f1009a present: {only_f1009a} ({only_f1009a/total_f1001a_2345*100:.2f}%)")
    print(f"  Both missing       : {both_null} ({both_null/total_f1001a_2345*100:.2f}%)")
    print(f"  Both present       : {both_notnull} ({both_notnull/total_f1001a_2345*100:.2f}%)")
    
    # Sample IDs can be inspected here if both fields are missing.
    #     problem_ids = df_f1001a_2345[~mask_f1009_notnull & ~mask_f1009a_notnull][['hhid', 'pline', 'f1001a', 'f1009', 'f1009a']]
    #     print(problem_ids.to_string(index=False))
else:
    print("\n[Check 3] No respondents have f1001a in [2, 3, 4, 5]")

print("Filtering complete")
print("=" * 80)
print("\nNext step: use sampled_ids_filtered.csv for the second-stage sample of 30")
