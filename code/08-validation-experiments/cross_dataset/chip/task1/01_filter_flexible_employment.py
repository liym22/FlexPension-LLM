"""
01_filter_flexible_employment.py
Identify flexible workers with pension data in CHIP 2018 and save the result.

Selection criteria:
  Criterion 1  Working age: men 16-59, women 16-54 (age = 2018 - A04_1)
  Criterion 7  Valid A23 response: neither -88 (不适用) nor -99 (不知道)
  Criterion 6  Exclude enterprise annuity and government/institution pension:
               A23 contains neither 2 nor 3
  Criterion 2  New-form employment: C09_5 contains 1 (快递/外卖),
               2 (网店), or 3 (网约车)
  Criterion 3  Urban flexible-employment pension: A23 contains 4
  Criterion 4  Employment status is 雇主/自营/家庭帮工: C03_1 in {1, 3, 4}
  Criterion 5  No labor contract: C07_1 == 4

Combined condition: 1 & 7 & 6 & (2 | 3 | 4 | 5)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chip2018"
OUT_PATH = SCRIPT_DIR / "filtered_flexible_workers.csv"


# ============================================================
# Parse comma-separated multiple-choice fields.
# ============================================================
def parse_multiselect(val):
    """
    Parse a comma-separated string into a Python set.

    Keep '-88' and '-99' as singleton sets; return an empty set for NaN.
    """
    if pd.isna(val):
        return set()
    s = str(val).strip()
    if s in ('-88', '-99'):
        return {s}
    return {item.strip() for item in s.split(',') if item.strip()}


# ============================================================
# 1. Load individual data.
# ============================================================
print("Loading data...")
urban_person = pd.read_stata(DATA_DIR / 'chip2018_urban_person.dta',
                             convert_categoricals=False)
rural_person = pd.read_stata(DATA_DIR / 'chip2018_rural_person.dta',
                             convert_categoricals=False)

urban_person['urban_rural'] = 1   # Urban.
rural_person['urban_rural'] = 0   # Rural.

# C09_4 is urban-only; concatenation leaves it missing for rural cases.
person = pd.concat([urban_person, rural_person], ignore_index=True, sort=False)
print(f"  Urban individual records: {len(urban_person):,}")
print(f"  Rural individual records: {len(rural_person):,}")
print(f"  Combined records: {len(person):,}")

# ============================================================
# 2. Preprocess fields.
# ============================================================
# Age.
person['age'] = 2018 - person['A04_1']

# Parse multiple-choice fields.
person['A23_set']   = person['A23'].apply(parse_multiselect)
person['C09_5_set'] = person['C09_5'].apply(parse_multiselect)

# ============================================================
# 3. Build inclusion criteria.
# ============================================================
# Criterion 1: working age.
cond1 = (
    ((person['A03'] == 1) & (person['age'] >= 16) & (person['age'] < 60)) |
    ((person['A03'] == 2) & (person['age'] >= 16) & (person['age'] < 55))
)

# Criterion 7: valid A23 response, excluding -88 and -99.
cond7 = person['A23'].apply(lambda x: str(x).strip() not in ('-88', '-99'))

# Criterion 6: A23 excludes codes 2 and 3.
cond6 = person['A23_set'].apply(lambda s: '2' not in s and '3' not in s)

# Criterion 2: C09_5 contains code 1, 2, or 3.
cond2 = person['C09_5_set'].apply(lambda s: bool(s & {'1', '2', '3'}))

# Criterion 3: A23 contains code 4.
cond3 = person['A23_set'].apply(lambda s: '4' in s)

# Criterion 4: C03_1 is in {1, 3, 4}.
cond4 = person['C03_1'].isin([1, 3, 4])

# Criterion 5: C07_1 equals 4.
cond5 = person['C07_1'] == 4

# ============================================================
# 4. Apply all criteria and report counts.
# ============================================================
total_cond = cond1 & cond7 & cond6 & (cond2 | cond3 | cond4 | cond5)

print("\n" + "=" * 50)
print("Selection counts:")
print(f"  Original sample                                      : {len(person):>6,}")
print(f"  Criterion 1  Working age                             : {cond1.sum():>6,}")
print(f"  + Criterion 7  Valid A23                             : {(cond1 & cond7).sum():>6,}")
print(f"  + Criterion 6  Exclude non-target pension schemes   : {(cond1 & cond7 & cond6).sum():>6,}")
_base = cond1 & cond7 & cond6
print(f"    Meeting criterion 2 (new-form employment)          : {(_base & cond2).sum():>6,}")
print(f"    Meeting criterion 3 (urban flexible pension)       : {(_base & cond3).sum():>6,}")
print(f"    Meeting criterion 4 (employment status)            : {(_base & cond4).sum():>6,}")
print(f"    Meeting criterion 5 (no contract)                  : {(_base & cond5).sum():>6,}")
print(f"  Final (any flexible-employment criterion)             : {total_cond.sum():>6,}")
print("=" * 50)

filtered = person[total_cond].copy()
print(f"\nFinal sample size: {len(filtered):,}")
print(f"  Urban: {(filtered['urban_rural'] == 1).sum():,}")
print(f"  Rural: {(filtered['urban_rural'] == 0).sum():,}")

# ============================================================
# 5. Join household-level variables.
# ============================================================
print("\nJoining household data...")

# P07_3 economic expectations come from the household files.
urban_hh = pd.read_stata(DATA_DIR / 'chip2018_urban_household.dta',
                         convert_categoricals=False)[['hhcode', 'P07_3']]
rural_hh = pd.read_stata(DATA_DIR / 'chip2018_rural_household.dta',
                         convert_categoricals=False)[['hhcode', 'P07_3']]
hh_p07 = pd.concat([urban_hh, rural_hh], ignore_index=True).drop_duplicates('hhcode')

# n3701 income and n4202 consumption come from the income-consumption file.
ic = pd.read_stata(DATA_DIR / 'chip2018_income_consumption.dta',
                   convert_categoricals=False)[['hhcode', 'n3701', 'n4202']]

filtered = filtered.merge(hh_p07, on='hhcode', how='left')
filtered = filtered.merge(ic, on='hhcode', how='left')

p07_match = filtered['P07_3'].notna().sum()
ic_match  = filtered['n3701'].notna().sum()
print(f"  P07_3 matched: {p07_match}/{len(filtered)}")
print(f"  n3701 matched: {ic_match}/{len(filtered)}")

# ============================================================
# 6. Save filtered data.
# ============================================================
# Remove temporary set-valued columns before CSV serialization.
filtered.drop(columns=['A23_set', 'C09_5_set'], inplace=True)

filtered.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUT_PATH}")
print(f"Columns: {len(filtered.columns)}, rows: {len(filtered)}")
