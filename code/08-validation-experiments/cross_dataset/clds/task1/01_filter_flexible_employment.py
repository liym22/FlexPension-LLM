#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_filter_flexible_employment.py
从 CLDS2018 个人数据中筛选灵活就业样本。

筛选条件：
  条件1  劳动年龄：男 16-59 岁，女 16-54 岁（2018 - birthyear）
  条件2  灵活就业（任一）：
           A: I3a_9 ∈ {9, 11, 12}  （个体工商户/自由职业者/无固定工作者）
           B: I3a_9 ∈ {1-8} AND I3a1_5 == 2  （有单位但无劳动合同）
  条件3  参保问题有效作答：I1_20_2 不缺失
  条件4  排除不可分类险种：
           I1_20_1 != 1  单位退休金/机关事业（同 CFPS qi301_a_1）
           I1_20_5 != 1  企业年金（同 CFPS qi301_a_3）
           I1_20_7 无非空有效填写（同 CHFS2019 排除 f1001a=7777）
  条件5  省份可确定：PROV2018 不缺失

Ground Truth 映射：
  I1_20_2 == 1                        → 城镇职工养老保险
  I1_20_3==1 OR I1_20_4==1 OR I1_20_50==1 → 城乡居民养老保险
  其他（含仅有商业保险 I1_20_6==1）   → 不参保

注意：CLDS 缺失值编码为 99997/99998/99999（拒绝/不适用/不清楚），
      与 CFPS 的 -8/-9/-1 不同，需特别处理。

输出:
  task1/filtered_flexible_workers.csv
  task1/01_filter_log.txt
"""

from pathlib import Path
import pandas as pd
import pyreadstat

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
PERSON_DTA = PROJECT_ROOT / "data" / "raw" / "clds2018" / "18个人.dta"
OUT_CSV  = SCRIPT_DIR / "filtered_flexible_workers.csv"
LOG_PATH = SCRIPT_DIR / "01_filter_log.txt"

SURVEY_YEAR = 2018

# Logging
_log = open(LOG_PATH, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log.write(msg + "\n")
    _log.flush()

log("=" * 70)
log("Task 1: Filter the CLDS 2018 flexible-worker sample")
log("=" * 70)

# Load required DTA columns
log("\n[1] Loading selected columns from 18个人.dta...")

USECOLS = [
    "IID2018", "FID2018",
    "Igender",        # 1 = male, 2 = female.
    "birthyear",      # Birth year.
    "PROV2018",       # Two-digit province code.
    "I1_3_1_psu",     # Six-digit hukou location code.
    "I1_14",          # Hukou type.
    # Flexible-employment fields.
    "I3a_9",          # Employer type.
    "I3a1_5",         # Labor contract: 1 = yes, 2 = no.
    "I3a_16",         # Employment status.
    # Pension fields used for ground truth and exclusions.
    "I1_20_1",        # Government pension; excluded.
    "I1_20_2",        # Employee pension.
    "I1_20_3",        # Urban resident pension; resident scheme.
    "I1_20_4",        # New rural pension; resident scheme.
    "I1_20_50",       # Unified resident pension.
    "I1_20_5",        # Enterprise annuity; excluded.
    "I1_20_6",        # Commercial pension; not a target scheme.
    "I1_20_7",        # Substantive other-scheme text; excluded.
]

ind, meta = pyreadstat.read_dta(str(PERSON_DTA), usecols=USECOLS)
log(f"  Raw individual records: {len(ind):,}")

# Calculate age
ind["age"] = SURVEY_YEAR - ind["birthyear"]

# Build inclusion criteria
log("\n[2] Building filter conditions...")

# Criterion 1: working age, Igender 1 = male and 2 = female.
cond_age = (
    ((ind["Igender"] == 1) & ind["age"].between(16, 59)) |
    ((ind["Igender"] == 2) & ind["age"].between(16, 54))
)

# Criterion 2: flexible employment.
# A: no affiliated employer.
cond_flex_no_unit = ind["I3a_9"].isin([9, 11, 12])
# B: employer present but no labor contract.
cond_flex_no_contract = (
    ind["I3a_9"].isin([1, 2, 3, 4, 5, 6, 7, 8]) & (ind["I3a1_5"] == 2)
)
cond_flex = cond_flex_no_unit | cond_flex_no_contract

# Criterion 3: valid pension response.
cond_valid_ins = ind["I1_20_2"].notna()

# Criterion 4: exclude non-target pension schemes.
# 4a: government pension.
cond_excl_gov = ind["I1_20_1"] == 1
# 4b: enterprise annuity.
cond_excl_supp = ind["I1_20_5"] == 1
# 4c: substantive I1_20_7 text after removing special missing codes.
# Empty, NaN, 99997-99999, 0, and "无" are treated as no substantive response.
_INVALID_7 = {"", "99997", "99998", "99999", "0", "无"}

def is_other_insured(val):
    if pd.isna(val):
        return False
    s = str(val).strip()
    return s not in _INVALID_7

cond_excl_other = ind["I1_20_7"].apply(is_other_insured)

cond_exclude = cond_excl_gov | cond_excl_supp | cond_excl_other

# Criterion 5: province is observed.
cond_no_prov = ind["PROV2018"].isna()

# Combine all criteria.
total_cond = cond_age & cond_flex & cond_valid_ins & ~cond_exclude & ~cond_no_prov

log("\n  Filter sequence (sample counts):")
log(f"  Raw sample                                             : {len(ind):>8,}")
log(f"  Condition 1: working age (men 16-59; women 16-54)     : {cond_age.sum():>8,}")
log(f"  + Condition 2: flexible work                          : {(cond_age & cond_flex).sum():>8,}")
log(f"    I3a_9 in {{9,11,12}} (no employer)                  : {(cond_age & cond_flex_no_unit).sum():>8,}")
log(f"    I3a_9 in {{1-8}} and I3a1_5=2 (no contract)         : {(cond_age & cond_flex_no_contract).sum():>8,}")
log(f"  + Condition 3: nonmissing I1_20_2                     : {(cond_age & cond_flex & cond_valid_ins).sum():>8,}")
log(f"  - Exclude government/institution workers (I1_20_1=1) : {cond_excl_gov[cond_age & cond_flex & cond_valid_ins].sum():>8,}")
log(f"  - Exclude enterprise annuity (I1_20_5=1)             : {cond_excl_supp[cond_age & cond_flex & cond_valid_ins].sum():>8,}")
log(f"  - Exclude substantive other insurance (I1_20_7)      : {cond_excl_other[cond_age & cond_flex & cond_valid_ins].sum():>8,}")
log(f"  Remaining after exclusions                            : {(cond_age & cond_flex & cond_valid_ins & ~cond_exclude).sum():>8,}")
log(f"  + Identifiable province (nonmissing PROV2018)         : {total_cond.sum():>8,}")

# Apply filters
filtered = ind[total_cond].copy()
log(f"\n  Final sample size: {len(filtered):,}")

# Ground-truth distribution
def get_ins_type(row):
    if row["I1_20_2"] == 1:
        return "城镇职工养老保险"
    if row["I1_20_3"] == 1 or row["I1_20_4"] == 1 or row["I1_20_50"] == 1:
        return "城乡居民养老保险"
    return "不参保"

filtered["ins_type"] = filtered.apply(get_ins_type, axis=1)

log("\n  Pension-channel distribution (ground truth):")
for ins, cnt in filtered["ins_type"].value_counts().items():
    log(f"    {ins}: {cnt:,} ({cnt/len(filtered)*100:.1f}%)")

log("\n  Flexible-work type distribution:")
log(f"    No employer (I3a_9 in {{9,11,12}}): {cond_flex_no_unit[total_cond].sum():,}")
log(f"    Employer but no contract (I3a_9 in {{1-8}}, I3a1_5=2): {cond_flex_no_contract[total_cond].sum():,}")

log("\n  Detailed I3a_9 distribution after filtering:")
I3A9_LABELS = {
    1: "党政机关", 2: "国有/集体事业单位", 3: "国营企业", 4: "集体企业",
    5: "村居委会", 6: "民营/私营企业", 7: "外资/合资", 8: "民办非企业",
    9: "个体工商户", 11: "自由职业者", 12: "无固定工作者",
}
for v, cnt in filtered["I3a_9"].value_counts().sort_index().items():
    label = I3A9_LABELS.get(int(v), f"code={v}")
    log(f"    {int(v):2d} {label}: {cnt:,}")

log("\n  Ten most frequent provinces:")
prov_dist = filtered["PROV2018"].astype(int).value_counts().sort_index()
for prov, cnt in prov_dist.head(10).items():
    log(f"    PROV={prov}: {cnt}")

# Save artifacts
log("\n[3] Saving...")
filtered.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
log(f"  Saved: {OUT_CSV}")
log(f"  Rows: {len(filtered):,}; columns: {len(filtered.columns)}")
log(f"  Column names: {list(filtered.columns)}")
log("\nComplete")
_log.close()
