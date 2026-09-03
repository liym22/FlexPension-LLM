#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_filter_flexible_employment.py
Filter flexible workers from CFPS 2018 individual data.

Selection criteria:
  Criterion 1  Working age: men 16-59, women 16-54 (age is provided)
  Criterion 2  Flexible employment (either):
           A: jobclass_base == 2  ("私营企业/个体工商户/其它自雇")
           B: jobclass_base == 4 AND qg5 == 0  (employed without a labor contract)
  Criterion 3  Valid pension response: qi301_a_2 != -8
           (-8 means the pension question was skipped, so ground truth is unavailable)
  Criterion 4  Exclude pension schemes that cannot be classified:
           qi301_a_1 != 1  机关事业离退休金
           qi301_a_3 != 1  企业补充养老保险
           qi301_a_77 != 1 其他 (unclassifiable)
  Criterion 5  Exclude overseas hukou: qa302 != 7
  Criterion 6  Known province: valid provcd18 (>0)
           provcd18 is a six-digit county code; its first two digits identify
           the province for region assignment and policy matching

Ground-truth mapping:
  qi301_a_2 == 1                  -> 城镇职工养老保险
  any qi301_a_5|6|7 == 1         -> 城乡居民养老保险
  otherwise (including qi301_a_78==1) -> 不参保

Outputs:
  task1/filtered_flexible_workers.csv
  task1/01_filter_log.txt
"""

from pathlib import Path
import pandas as pd
import pyreadstat

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
PERSON_DTA = PROJECT_ROOT / "data" / "raw" / "cfps2018" / "cfps2018person_202512.dta"
OUT_CSV  = SCRIPT_DIR / "filtered_flexible_workers.csv"
LOG_PATH = SCRIPT_DIR / "01_filter_log.txt"

# Logging
_log = open(LOG_PATH, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log.write(msg + "\n")
    _log.flush()

log("=" * 70)
log("Task 1  CFPS 2018 flexible-worker sample filtering")
log("=" * 70)

# Load required DTA columns
log("\n[1] Loading selected columns from cfps2018person_202512.dta...")

USECOLS = [
    "pid", "fid18",
    "gender",           # 0 = female, 1 = male.
    "age",              # Age.
    "urban18",          # 0 = rural, 1 = urban.
    "jobclass_base",    # Employment status.
    "qg5",              # 0 = no contract, 1 = contract.
    # Pension fields used for ground truth and exclusions.
    "qi301_a_1",        # Government pension; excluded.
    "qi301_a_2",        # Basic employee pension.
    "qi301_a_3",        # Enterprise supplementary pension; excluded.
    "qi301_a_5",        # Old rural pension; resident scheme.
    "qi301_a_6",        # New rural pension; resident scheme.
    "qi301_a_7",        # Urban resident pension; resident scheme.
    "qi301_a_77",       # Other scheme; excluded.
    "qi301_a_78",       # No listed pension; non-participation.
    "qi2001",           # Pension receipt for household summaries.
    # Province fields for eligibility and task 3 reconstruction.
    "qa302",            # Hukou location type; code 7 is overseas.
    "qa302a_code",      # Hukou province when qa302 >= 2.
    "ear201a",          # Residence province.
    "provcd18",         # Six-digit residence county code.
]

ind, meta = pyreadstat.read_dta(str(PERSON_DTA), usecols=USECOLS)
log(f"  Raw individual records: {len(ind):,}")

# Build inclusion criteria
log("\n[2] Building selection criteria...")

# Criterion 1: working age, gender 1 = male and 0 = female.
cond_age = (
    ((ind["gender"] == 1) & ind["age"].between(16, 59)) |
    ((ind["gender"] == 0) & ind["age"].between(16, 54))
)

# Criterion 2: flexible employment.
cond_flex_self  = ind["jobclass_base"] == 2                          # Self-employed.
cond_flex_hired = (ind["jobclass_base"] == 4) & (ind["qg5"] == 0)   # Employee without a contract.
cond_flex = cond_flex_self | cond_flex_hired

# Criterion 3: valid pension responses.
cond_valid_ins = ind["qi301_a_2"] != -8

# Criterion 4: exclude non-target pension schemes.
cond_exclude = (
    (ind["qi301_a_1"]  == 1) |   # Government pension.
    (ind["qi301_a_3"]  == 1) |   # Enterprise supplementary pension.
    (ind["qi301_a_77"] == 1)     # Other pension.
)

# Criterion 5: exclude overseas hukou.
cond_abroad = ind["qa302"] == 7

# Criterion 6: valid province code.
cond_no_prov = ind["provcd18"].isna() | (ind["provcd18"] <= 0)

total_cond = cond_age & cond_flex & cond_valid_ins & ~cond_exclude & ~cond_abroad & ~cond_no_prov

log(f"\n  Selection counts:")
log(f"  Original sample                                      : {len(ind):>8,}")
log(f"  Criterion 1  Working age                             : {cond_age.sum():>8,}")
log(f"  + Criterion 2  Flexible employment                   : {(cond_age & cond_flex).sum():>8,}")
log(f"    jobclass_base==2 (self-employed)                    : {(cond_age & cond_flex_self).sum():>8,}")
log(f"    jobclass_base==4+qg5==0 (employee without contract): {(cond_age & cond_flex_hired).sum():>8,}")
log(f"  + Criterion 3  Valid qi301_a_2 (not -8)              : {(cond_age & cond_flex & cond_valid_ins).sum():>8,}")
log(f"  - Criterion 4  Exclude unclassified pension schemes  : {(cond_age&cond_flex&cond_valid_ins&~cond_exclude).sum():>8,}")
log(f"  - Criterion 5  Exclude overseas hukou (qa302=7)      : {(cond_age&cond_flex&cond_valid_ins&~cond_exclude&~cond_abroad).sum():>8,}")
log(f"  - Criterion 6  Require valid province (provcd18)      : {total_cond.sum():>8,}")

# Apply filters
filtered = ind[total_cond].copy()

log(f"\n  Final sample size: {len(filtered):,}")

# Ground-truth distribution
def get_ins_type(row):
    if row["qi301_a_2"] == 1:
        return "城镇职工养老保险"
    if row["qi301_a_5"] == 1 or row["qi301_a_6"] == 1 or row["qi301_a_7"] == 1:
        return "城乡居民养老保险"
    return "不参保"

filtered["ins_type"] = filtered.apply(get_ins_type, axis=1)

log(f"\n  Ground-truth pension scheme distribution:")
for ins, cnt in filtered["ins_type"].value_counts().items():
    log(f"    {ins}: {cnt:,} ({cnt/len(filtered)*100:.1f}%)")

log(f"\n  Urban-rural distribution:")
urban_map = {0: "乡村", 1: "城镇"}
for code, cnt in filtered["urban18"].value_counts().sort_index().items():
    label = urban_map.get(int(code), f"code={code}")
    log(f"    {label}: {cnt:,}")

log(f"\n  Employment-type distribution:")
log(f"    Self-employed (jobclass_base==2): {cond_flex_self[total_cond].sum():,}")
log(f"    Employee without a contract (jobclass_base==4, qg5==0): {cond_flex_hired[total_cond].sum():,}")

# Save artifacts
log(f"\n[3] Saving...")
filtered.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
log(f"  Saved: {OUT_CSV}")
log(f"  Rows: {len(filtered):,}  Columns: {len(filtered.columns)}")
log(f"  Column names: {list(filtered.columns)}")
log("\nDone.")
_log.close()
