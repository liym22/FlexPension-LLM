#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reconstruct prompt variables for the 500-case CHFS 2017 sample.

Version 2 infers accumulated contribution years from account balance
(`f1010_imp`) and monthly contribution (`f1008_imp`) with a compound-interest
formula. The calculation applies only when the historical pension channel is
known, and the raw inferred value is retained for diagnostics.

Inputs include the 2017 individual, household, and master files; the 2015
household file used to supplement risk preference for panel households; and
the 2016 province-level pension-policy workbook.

Compared with the 2019 workflow, this script uses the 2017 employment,
industry, occupation, and employer codes; includes subsidy income; infers
participation history from account balances; combines 2017 and 2015 risk
preference; derives region from province; and computes burden ratios from the
largest positive denominator among personal income, per-capita household
income, and per-capita household net assets.
"""

import sys
import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
import openpyxl

# ==================== Version 2 constants ====================
RATE_ZHIGONG    = 0.03   # Historical approximation for the employee-pension account interest rate.
RATE_JUMIN      = 0.02   # Historical approximation for the resident-pension account interest rate.
MAX_YEARS_MALE  = 44     # Male cap: retirement at 60 minus earliest employment at 16.
MAX_YEARS_FEMALE = 39    # Female cap: retirement at 55 minus earliest employment at 16.

_RATE_MAP = {
    "城镇职工养老保险": RATE_ZHIGONG,
    "城乡居民养老保险": RATE_JUMIN,
}

# ==================== Paths ====================
SCRIPT_DIR   = Path(__file__).resolve().parent
TASK2_DIR    = SCRIPT_DIR.parent / "task2_sampling"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from common.household_enrollment import count_other_enrolled_members

DTA_2017_DIR = PROJECT_ROOT / "data" / "raw" / "chfs2017"
DTA_2015_DIR = PROJECT_ROOT / "data" / "raw" / "chfs2015"
POLICY_XLS   = PROJECT_ROOT / "data" / "raw" / "policy" / "province_policy_2016.xlsx"
OUT_CSV      = SCRIPT_DIR / "reconstructed_500_v2.csv"
LOG_PATH     = SCRIPT_DIR / "03_reconstruct_v2_log.txt"

_log = open(LOG_PATH, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log.write(msg + "\n")
    _log.flush()

log("=" * 70)
log("Task 3 (v2): Reconstruct CHFS 2017 variables and contribution years")
log("=" * 70)

# ==================== Static mappings ====================
PROV_TO_REGION = {
    11: "东部", 12: "东部", 13: "东部", 31: "东部", 32: "东部",
    33: "东部", 35: "东部", 37: "东部", 44: "东部", 46: "东部",
    14: "中部", 34: "中部", 36: "中部", 41: "中部", 42: "中部", 43: "中部",
    15: "西部", 45: "西部", 50: "西部", 51: "西部", 52: "西部",
    53: "西部", 54: "西部", 61: "西部", 62: "西部", 63: "西部",
    64: "西部", 65: "西部",
    21: "东北", 22: "东北", 23: "东北",
}

MISSING_VALS = {-88, -99, -8, -9}

def is_missing(v):
    if v is None: return True
    if isinstance(v, float) and np.isnan(v): return True
    try:
        return int(v) in MISSING_VALS
    except (ValueError, TypeError):
        return False

def safe_map(val, mapping):
    if is_missing(val): return "不清楚"
    try:
        k = int(val)
    except (ValueError, TypeError):
        return "不清楚"
    return mapping.get(k, "不清楚")

# ==================== 1. Load data ====================
log("\n[1] Loading data (DTA files may take a moment)...")

sampled = pd.read_csv(
    TASK2_DIR / "sampled_500.csv",
    dtype={"hhid": str, "pline": str},
)
log(f"  sampled_500: {len(sampled)} rows")
target_hhids   = set(sampled["hhid"].astype(str))
target_samples = set(zip(sampled["hhid"].astype(str), sampled["pline"].astype(str)))

# Individual table: load all required rows to compute household statistics.
IND_COLS = [
    "hhid", "pline", "hhid_2015",
    "a2003", "a2005",           # Gender and age
    "a2012", "a2025b",          # Education and health
    "a2022", "a2019_prov",      # Hukou type and province
    "a2000c", "a2016b_prov",    # Migration status and province of residence
    "a3132a",                   # Employment arrangement
    "a3110", "a3141",           # Current/former industry
    "a3111", "a3142",           # Current/former occupation
    "a3106", "a3140",           # Current/former employer type
    "a3136_imp", "a3136a_imp", "a3136b_imp",  # Wages, bonuses, and allowances
    "f1001a", "f1008_imp", "f1010_imp",       # Pension enrollment
    "f1003", "f1005_imp",                      # Pension benefit receipt
]
# Read only columns present in the source file.
ind_all, _ = pyreadstat.read_dta(str(DTA_2017_DIR / "chfs2017_ind_202206.dta"),
                                  usecols=IND_COLS)
ind_all["hhid"]  = ind_all["hhid"].astype(str)
ind_all["pline"] = ind_all["pline"].astype(str)
log(f"  Individual table (selected columns): {len(ind_all)} rows x {len(ind_all.columns)} columns")

# Household table: self-employment income and risk preference.
HH_COLS = ["hhid", "h3104", "b2002", "b2054", "b2055_imp", "b2003e_imp"]
hh_data, _ = pyreadstat.read_dta(str(DTA_2017_DIR / "chfs2017_hh_202206.dta"),
                                   usecols=[c for c in HH_COLS
                                            if c in pyreadstat.read_dta(
                                                str(DTA_2017_DIR / "chfs2017_hh_202206.dta"),
                                                metadataonly=True)[1].column_names])
hh_data["hhid"] = hh_data["hhid"].astype(str)
log(f"  Household table (selected columns): {len(hh_data)} rows x {len(hh_data.columns)} columns")

# Master table: household totals, urban/rural status, and province.
MASTER_COLS = ["hhid", "pline", "total_income", "total_consump",
               "total_asset", "total_debt", "rural", "prov", "prov_code"]
master_data, _ = pyreadstat.read_dta(
    str(DTA_2017_DIR / "chfs2017_master_202206.dta"),
    usecols=[c for c in MASTER_COLS
             if c in pyreadstat.read_dta(
                 str(DTA_2017_DIR / "chfs2017_master_202206.dta"),
                 metadataonly=True)[1].column_names])
master_data["hhid"]  = master_data["hhid"].astype(str)
master_data["pline"] = master_data["pline"].astype(str)
log(f"  Master table (selected columns): {len(master_data)} rows x {len(master_data.columns)} columns")

# 2015 household table: risk preference (a4003) for the panel sample.
hh15_data = None
if (DTA_2015_DIR / "chfs2015_hh_20191120_version14.dta").exists():
    try:
        hh15_data, _ = pyreadstat.read_dta(
            str(DTA_2015_DIR / "chfs2015_hh_20191120_version14.dta"),
            usecols=["hhid_2015", "a4003"])
        hh15_data["hhid_2015"] = hh15_data["hhid_2015"].astype(str)
        log(f"  2015 household table: {len(hh15_data)} rows (a4003 panel risk preference)")
    except Exception as e:
        log(f"  Failed to load the 2015 household table: {e}; panel risk preference will be missing")

# Policy workbook: the 2016 sheet uses formulas, so data_only may return None.
log("  Loading the 2016 policy workbook...")
wb = openpyxl.load_workbook(POLICY_XLS, data_only=True)
ws = wb["总表"]
all_rows   = list(ws.iter_rows(values_only=True))
header     = [str(c) if c is not None else f"_col{i}" for i, c in enumerate(all_rows[0])]
policy_df  = pd.DataFrame(all_rows[1:], columns=header)
policy_df  = policy_df[policy_df["省名"].notna()].copy()

# Recompute missing formula values.
# Annual contribution base (lower) = average wage * contribution-index lower bound.
# Annual contribution base (upper) = average wage * contribution-index upper bound.
# Annual contribution (lower) = contribution-base lower bound * contribution rate.
# Annual contribution (upper) = contribution-base upper bound * contribution rate.
def safe_float(v):
    try:
        return float(v) if pd.notna(v) else np.nan
    except:
        return np.nan

for idx, row in policy_df.iterrows():
    wage = safe_float(row.get("社平工资（年）"))
    idx_low = safe_float(row.get("缴费指数（下限）"))
    idx_high = safe_float(row.get("缴费指数（上限）"))
    ratio = safe_float(row.get("缴费比例"))
    
    # Compute the annual contribution-base lower bound.
    base_low = row.get("缴费基数（年，下限）")
    if pd.isna(base_low) or base_low is None:
        if not np.isnan(wage) and not np.isnan(idx_low):
            policy_df.at[idx, "缴费基数（年，下限）"] = wage * idx_low
            base_low = wage * idx_low
        else:
            base_low = np.nan
    else:
        base_low = float(base_low)
    
    # Compute the annual contribution-base upper bound.
    base_high = row.get("缴费基数（年，上限）")
    if pd.isna(base_high) or base_high is None:
        if not np.isnan(wage) and not np.isnan(idx_high):
            policy_df.at[idx, "缴费基数（年，上限）"] = wage * idx_high
            base_high = wage * idx_high
        else:
            base_high = np.nan
    else:
        base_high = float(base_high)
    
    # Compute the annual contribution lower bound.
    fee_low = row.get("缴费金额（年，下限）")
    if pd.isna(fee_low) or fee_low is None:
        if not np.isnan(base_low) and not np.isnan(ratio):
            policy_df.at[idx, "缴费金额（年，下限）"] = base_low * ratio
    
    # Compute the annual contribution upper bound.
    fee_high = row.get("缴费金额（年，上限）")
    if pd.isna(fee_high) or fee_high is None:
        if not np.isnan(base_high) and not np.isnan(ratio):
            policy_df.at[idx, "缴费金额（年，上限）"] = base_high * ratio

policy_dict = {row["省名"]: row for _, row in policy_df.iterrows()}
log(f"  Policy table: {len(policy_dict)} provinces")

# ==================== 2. Income terciles ====================
log("\n[2] Computing personal-income terciles from all flexible workers...")

# Use personal income from the sampled frame, consistent with the CHIP pipeline.
# Compute income for all screened sampled cases before deriving terciles.
flex_ind = ind_all[ind_all["a3132a"].isin([2, 3, 4, 5, 6])].copy()

def _calc_income_for_tercile(row):
    work_type = row.get("a3132a", np.nan)
    income    = 0.0
    if work_type in [1, 2, 5, 6]:
        income += row.get("a3136_imp",  0) or 0
        income += row.get("a3136a_imp", 0) or 0
        income += row.get("a3136b_imp", 0) or 0
    return income

flex_ind["_income"] = flex_ind.apply(_calc_income_for_tercile, axis=1)
valid_inc = flex_ind["_income"][flex_ind["_income"] > 0]
q33 = valid_inc.quantile(1/3)
q67 = valid_inc.quantile(2/3)
log(f"  1/3 quantile: {q33:,.0f} yuan | 2/3 quantile: {q67:,.0f} yuan (valid N={len(valid_inc):,})")

def income_tercile(income_val):
    if pd.isna(income_val) or income_val <= 0:
        return "不清楚"
    if income_val <= q33:
        return "低收入"
    elif income_val <= q67:
        return "中收入"
    else:
        return "高收入"

# ==================== 3. Household statistics from all individuals ====================
log("\n[3] Computing household statistics from all individual records...")

fam = ind_all[ind_all["hhid"].isin(target_hhids)].copy()
fam["_age"] = 2017 - pd.to_numeric(fam["a2005"], errors="coerce")
fam["_insured_for_household_count"] = (
    (fam["f1001a"] != 7788) & fam["f1001a"].notna()
).astype(int)
log(f"  Household members: {len(fam)} across {fam['hhid'].nunique()} households")

def agg_family(grp):
    size     = len(grp)
    children = int((grp["_age"] < 16).sum())
    elderly  = int((grp["_age"] >= 60).sum())
    # Receiving pension benefits when f1003 equals 1.
    receiving = grp["f1003"] == 1 if "f1003" in grp.columns else pd.Series(False, index=grp.index)
    recv      = int(receiving.sum())
    # Total monthly household pension benefits
    pension   = float(
        grp.loc[receiving, "f1005_imp"].sum()
        if "f1005_imp" in grp.columns else 0
    )
    return pd.Series({
        "家庭人数":         size,
        "子女数":           children,
        "老人数":           elderly,
        "家庭领取人数":     recv,
        "家庭月均养老金_数值": pension,
    })

family_stats = fam.groupby("hhid").apply(agg_family).reset_index()
log(f"  Household statistics complete for {len(family_stats)} households")

# ==================== 4. Build the merged 500-case table ====================
log("\n[4] Merging tables for the 500-case sample...")

# Restrict the individual table to the 500 sampled cases.
ind_500 = ind_all[
    ind_all.apply(lambda r: (r["hhid"], r["pline"]) in target_samples, axis=1)
].copy()
log(f"  ind_500: {len(ind_500)} rows")

# Merge the master table by hhid and pline.
master_cols = [c for c in MASTER_COLS if c in master_data.columns]
df = ind_500.merge(master_data[master_cols], on=["hhid", "pline"], how="left")

# Merge the 2017 household table by hhid.
hh_keep = [c for c in ["hhid", "h3104", "b2002", "b2054", "b2055_imp", "b2003e_imp"]
           if c in hh_data.columns]
df = df.merge(hh_data[hh_keep], on="hhid", how="left")

# Merge the 2015 household table for panel cases by hhid_2015.
if hh15_data is not None and "hhid_2015" in df.columns:
    df["hhid_2015"] = df["hhid_2015"].astype(str)
    df = df.merge(hh15_data[["hhid_2015", "a4003"]], on="hhid_2015", how="left")
    log(f"  After merging the 2015 household table: {df['a4003'].notna().sum()} nonmissing a4003 values")
else:
    df["a4003"] = np.nan

# Merge household statistics.
df = df.merge(family_stats, on="hhid", how="left")
df["家庭参保人数"] = count_other_enrolled_members(
    fam,
    df,
    household_col="hhid",
    person_col="pline",
    enrolled_col="_insured_for_household_count",
)

# Derive region.
if "prov_code" in df.columns:
    df["region"] = df["prov_code"].apply(
        lambda v: PROV_TO_REGION.get(int(v), "不清楚") if pd.notna(v) else "不清楚"
    )
else:
    df["region"] = "不清楚"

log(f"  Merge complete: {len(df)} rows x {len(df.columns)} columns")

# Merge sampled strata.
df = df.merge(sampled[["hhid", "pline", "stratum"]], on=["hhid", "pline"], how="left")

# ==================== 5. Mapping functions ====================
def map_gender(v): return {1:"男", 2:"女"}.get(v, "不清楚")
def map_edu(v):
    if is_missing(v): return "不清楚"
    return {1:"未上学", 2:"小学", 3:"初中", 4:"高中", 5:"中专/职高",
            6:"大专", 7:"本科", 8:"硕士", 9:"博士"}.get(int(v), "不清楚")
def map_health(v):  return {1:"非常好", 2:"好", 3:"一般", 4:"不好", 5:"非常不好"}.get(v, "不清楚")
def map_hukou(v):   return {1:"农业", 2:"非农业", 3:"统一居民", 7777:"其他"}.get(v, "不清楚")

def get_residence_prov(row):
    a2000c = row.get("a2000c")
    if a2000c == 1:
        return str(row.get("a2019_prov", "")) or ""
    elif a2000c == 2:
        res = str(row.get("a2016b_prov", ""))
        return res if res and res not in ("nan", "") else str(row.get("a2019_prov", ""))
    return str(row.get("a2019_prov", "")) or ""

def map_work_type(v):
    return {2:"临时工", 3:"务农", 4:"自营", 5:"自由职业", 6:"其他"}.get(v, "不清楚")

def map_industry(v):
    if is_missing(v): return "不清楚"
    m = {
        1:"农林牧渔",   2:"采矿业",       3:"制造业",
        4:"电力燃气水", 5:"建筑",         6:"批发零售",
        7:"交通运输",   8:"住宿餐饮",     9:"信息技术",
        10:"金融",     11:"房地产",       12:"租赁商务服务",
        13:"科研技术",  14:"水利环境",    15:"居民服务",
        16:"教育",     17:"卫生社会工作", 18:"文化体育娱乐",
        19:"公共管理",  20:"国际组织",    7777:"其他",
    }
    return m.get(int(v), "不清楚")

def map_occupation(v):
    if is_missing(v): return "不清楚"
    m = {1:"负责人", 2:"专业技术", 3:"办事人员", 4:"服务人员",
         5:"农林牧渔", 6:"生产制造", 7:"其他", 8:"其他", 7777:"其他"}
    return m.get(int(v), "不清楚")

def map_employer(v):
    if is_missing(v): return "不清楚"
    m = {1:"机关事业", 2:"国企", 3:"集体企业", 4:"个体户",
         5:"私企", 6:"外资/港澳台", 7:"其他", 8:"其他", 7777:"其他"}
    return m.get(int(v), "不清楚")

def map_risk(v):
    if is_missing(v): return "不知道"
    return {1:"高风险偏好", 2:"略高风险偏好", 3:"平均风险偏好",
            4:"略低风险偏好", 5:"极度厌恶风险", 6:"不知道"}.get(int(v), "不知道")

def map_rural(v): return {0:"城镇", 1:"农村"}.get(v, "不清楚")

def fmt_yuan(v):
    if pd.isna(v): return "不清楚"
    try:   return f"{int(v)}元"
    except: return "不清楚"

def coalesce(row, primary, fallback):
    v = row.get(primary, np.nan)
    return v if pd.notna(v) else row.get(fallback, np.nan)

# ==================== 6. Individual-income calculation ====================
def calc_personal_income(row):
    work_type = row.get("a3132a", np.nan)
    income    = 0.0
    if work_type in [1, 2, 5, 6]:
        income += row.get("a3136_imp",  0) or 0
        income += row.get("a3136a_imp", 0) or 0
        income += row.get("a3136b_imp", 0) or 0  # ≡ a3136bb+a3136bd
    elif work_type == 3:    # Exclude agricultural work.
        pass
    elif work_type == 4:    # Self-employment; b2002b_imp is unavailable.
        b2002 = row.get("b2002", np.nan)
        if pd.notna(b2002):
            if b2002 > 1:
                v = row.get("b2003e_imp", np.nan)
                income += v if pd.notna(v) else 0
            elif b2002 == 1:
                b2054 = row.get("b2054", np.nan)
                v     = row.get("b2055_imp", np.nan)
                if pd.notna(b2054) and pd.notna(v):
                    income += v if b2054 == 1 else (-v if b2054 == 2 else 0)
    return income

# ==================== 7. Derived enrollment variables ====================
def _calc_hist_years(f1010_imp, f1008_imp, hist_type, gender):
    """
    利用复利公式反推累计缴纳年限。
    返回 (accumulated_years, n_raw)：
      - accumulated_years: int ≥ 0，或 None（无法推算）
      - n_raw: float（原始n值，保留两位小数），或 None

    仅当 hist_type ∈ {"城镇职工养老保险", "城乡居民养老保险"} 时计算。
    "不参保" 和 "不清楚" 均返回 (None, None)，不参与计算。

    公式：A = P × (1 + r/2) × [(1+r)^n − 1] / r
    反推：n = log(1 + A×r / [P×(1+r/2)]) / log(1+r)
    累计年限 = floor(n) − 1（保守估计，排除当前进行年）
    """
    r = _RATE_MAP.get(hist_type)
    if r is None:                          # Do not calculate for non-enrollment or unknown status.
        return None, None

    A = f1010_imp
    P = f1008_imp * 12                     # Convert monthly to annual contributions.

    if not (A > 0 and P > 0):
        return None, None

    arg = 1.0 + A * r / (P * (1.0 + r / 2.0))
    if arg <= 1.0:                         # Invalid logarithm argument
        return None, None

    n_raw = math.log(arg) / math.log(1.0 + r)
    n_raw_rounded = round(n_raw, 2)
    accumulated = round(n_raw) - 1         # Round first, then subtract one.

    max_years = MAX_YEARS_MALE if gender == "男" else MAX_YEARS_FEMALE
    if accumulated < 0 or n_raw > max_years:
        return None, n_raw_rounded         # Retain the raw value for diagnostics.

    return accumulated, n_raw_rounded


def calc_insurance_vars(row):
    f1001a    = row.get("f1001a", np.nan)
    f1008_imp = row.get("f1008_imp", np.nan)
    f1010_imp = row.get("f1010_imp", np.nan)
    gender    = map_gender(row.get("a2003"))   # Protocol values: male, female, or unknown in Chinese.

    uninsured_2016 = (
        f1001a == 7788
        or (f1001a in [2, 3, 4, 5, 7777] and pd.notna(f1008_imp) and f1008_imp == 0)
    )
    decision_2016 = "不参保" if uninsured_2016 else "参保"

    if f1001a == 2:
        account_2017 = "城镇职工养老保险"
    elif f1001a in [3, 4, 5]:
        account_2017 = "城乡居民养老保险"
    else:
        account_2017 = "不参保"

    if decision_2016 == "不参保":
        hist_type = "不参保"
    elif (pd.notna(f1010_imp) and pd.notna(f1008_imp)
          and f1008_imp > 0 and f1010_imp >= 2 * f1008_imp * 12):
        hist_type = account_2017
    else:
        hist_type = "不清楚"

    hist_years, hist_years_raw = _calc_hist_years(f1010_imp, f1008_imp, hist_type, gender)
    hist_gap = "不存在" if hist_type == "不参保" else "不清楚"

    return decision_2016, account_2017, hist_type, hist_years, hist_years_raw, hist_gap

# ==================== 8. Policy-matching functions ====================
def match_policy(province_name):
    if not isinstance(province_name, str) or province_name.strip() in ("", "不清楚", "nan"):
        return None
    
    # Try a direct match.
    pol = policy_dict.get(province_name)
    if pol is not None:
        return pol
    
    # Map abbreviated province names to full names.
    PROVINCE_ABBR = {
        "新疆": "新疆维吾尔自治区",
        "广西": "广西壮族自治区",
        "内蒙古": "内蒙古自治区",
        "宁夏": "宁夏回族自治区",
        "西藏": "西藏自治区",
    }
    
    # Try the abbreviation mapping.
    for abbr, full in PROVINCE_ABBR.items():
        if abbr in province_name:
            pol = policy_dict.get(full)
            if pol is not None:
                return pol
    
    # Try matching after removing administrative suffixes.
    for suffix in ["省","市","自治区","壮族自治区","回族自治区","维吾尔自治区","藏族自治区"]:
        stripped = province_name.replace(suffix, "")
        # First try the abbreviation mapping.
        if stripped in PROVINCE_ABBR:
            pol = policy_dict.get(PROVINCE_ABBR[stripped])
            if pol is not None:
                return pol
        # Then try a suffix-free direct match, such as Beijing to Beijing Municipality.
        for full_name in policy_dict.keys():
            if full_name.replace(suffix, "") == stripped:
                return policy_dict.get(full_name)
    
    return pol

def safe_pol_int(pol, key):
    """Read an integer policy value; return '不清楚' when invalid."""
    v = pol.get(key)
    try:
        if pd.isna(v) or v is None:
            return "不清楚"
        return int(float(v))
    except (TypeError, ValueError):
        return "不清楚"

def safe_pol_float(pol, key):
    """Read a floating-point policy value; return NaN when invalid."""
    v = pol.get(key)
    try:
        if pd.isna(v) or v is None:
            return np.nan
        return float(v)
    except (TypeError, ValueError):
        return np.nan

# ==================== 9. Reconstruct each case ====================
log("\n[5] Reconstructing variables by record...")

records      = []
warn_no_pol  = set()

for _, row in df.iterrows():
    r = {}
    hhid  = str(row["hhid"])
    pline = str(row["pline"])

    # Identifiers
    r["家庭ID"] = hhid
    r["个人ID"] = pline

    # Individual demographics
    a2005 = pd.to_numeric(row.get("a2005"), errors="coerce")
    r["年龄"]     = int(2017 - a2005) if not pd.isna(a2005) else "不清楚"
    r["性别"]     = map_gender(row.get("a2003"))
    r["文化程度"] = map_edu(row.get("a2012"))
    r["健康状况"] = map_health(row.get("a2025b"))

    # Hukou and migration
    r["户口性质"] = map_hukou(row.get("a2022"))
    hk_prov       = str(row.get("a2019_prov", "")).strip()
    r["户口省份"] = hk_prov if hk_prov and hk_prov != "nan" else "不清楚"
    r["常住省份"] = get_residence_prov(row)
    r["是否流动"] = {1:"否", 2:"是"}.get(row.get("a2000c"), "否")

    # Employment
    r["工作性质"] = map_work_type(row.get("a3132a"))
    r["工作行业"] = map_industry(coalesce(row, "a3110", "a3141"))
    r["工作职业"] = map_occupation(coalesce(row, "a3111", "a3142"))
    r["单位类型"] = map_employer(coalesce(row, "a3106", "a3140"))

    # Annual individual income
    personal_income_num = calc_personal_income(row)
    if pd.isna(personal_income_num):
        personal_income_str = "不清楚"
    elif personal_income_num == 0 and row.get("a3132a") == 3:
        personal_income_str = "不清楚"   # Exclude agricultural income.
        personal_income_num = np.nan
    else:
        personal_income_str = f"{int(personal_income_num)}元"
    r["个人年收入"] = personal_income_str

    # Household finances
    total_income  = pd.to_numeric(row.get("total_income"),  errors="coerce")
    total_consump = pd.to_numeric(row.get("total_consump"), errors="coerce")
    total_asset   = pd.to_numeric(row.get("total_asset"),   errors="coerce")
    total_debt    = pd.to_numeric(row.get("total_debt"),    errors="coerce")
    r["家庭总收入"] = fmt_yuan(total_income)
    r["家庭总消费"] = fmt_yuan(total_consump)
    r["家庭总资产"] = fmt_yuan(total_asset)
    r["家庭总负债"] = fmt_yuan(total_debt)

    # Household composition
    r["家庭人数"] = int(row["家庭人数"]) if not pd.isna(row.get("家庭人数")) else "不清楚"
    r["子女数"]   = int(row["子女数"])   if not pd.isna(row.get("子女数"))   else "不清楚"
    r["老人数"]   = int(row["老人数"])   if not pd.isna(row.get("老人数"))   else "不清楚"

    # Derived enrollment variables
    dec_2016, acc_2017, hist_type, hist_years, hist_years_raw, hist_gap = calc_insurance_vars(row)
    r["参保决策"] = dec_2016    # Ground truth: 2016 enrollment decision.
    r["参保账户"] = acc_2017    # Ground truth: 2017 enrollment account.
    r["历史参保类型"] = hist_type
    r["累计缴纳年限"] = "不清楚" if hist_years is None else f"{hist_years}年"
    r["累计缴纳年限（原始n值）"] = "" if hist_years_raw is None else hist_years_raw
    r["是否存在断缴"] = hist_gap

    # Household enrollment, benefit receipt, and pension income
    r["家庭参保人数"] = int(row["家庭参保人数"]) if not pd.isna(row.get("家庭参保人数")) else 0
    r["家庭领取人数"] = int(row["家庭领取人数"]) if not pd.isna(row.get("家庭领取人数")) else 0
    pension_num      = row.get("家庭月均养老金_数值", 0)
    pension_for_calc = float(pension_num) if not pd.isna(pension_num) else 0.0
    r["家庭月均养老金"] = (f"{int(pension_for_calc)}元" if pension_for_calc > 0 else "未领取")

    # Psychological characteristics
    risk_raw = row.get("h3104", np.nan)
    if pd.isna(risk_raw) and "a4003" in row:
        risk_raw = row.get("a4003", np.nan)
    r["风险偏好"] = map_risk(risk_raw)
    r["经济预期"] = "不清楚"    # h3601 is unavailable in the 2017 survey.

    # Region
    r["城乡"] = map_rural(row.get("rural"))
    r["地区"] = str(row.get("region", "不清楚"))

    # Income tercile
    r["收入分组"] = income_tercile(personal_income_num)

    # Policy variables
    pol = match_policy(r["户口省份"])
    if pol is None:
        if r["户口省份"] != "不清楚":
            warn_no_pol.add(r["户口省份"])
        pol_keys = ["社平工资","缴费指数下限","缴费指数上限","缴费基数下限",
                    "缴费基数上限","可选档次规则","缴费金额下限","缴费金额上限",
                    "缴费档次与补贴明细","基础养老金"]
        for k in pol_keys:
            r[k] = "不清楚"
        min_jumin   = np.nan
        min_zhigong = np.nan
    else:
        # Read numeric fields, preferring recomputed formula values.
        r["社平工资"]           = safe_pol_int(pol, "社平工资（年）")
        r["缴费指数下限"]       = pol.get("缴费指数（下限）") if pd.notna(pol.get("缴费指数（下限）")) else "不清楚"
        r["缴费指数上限"]       = pol.get("缴费指数（上限）") if pd.notna(pol.get("缴费指数（上限）")) else "不清楚"
        r["缴费基数下限"]       = safe_pol_int(pol, "缴费基数（年，下限）")
        r["缴费基数上限"]       = safe_pol_int(pol, "缴费基数（年，上限）")
        r["可选档次规则"]       = pol.get("可选档次规则") if pd.notna(pol.get("可选档次规则")) else "不清楚"
        r["缴费金额下限"]       = safe_pol_int(pol, "缴费金额（年，下限）")
        r["缴费金额上限"]       = safe_pol_int(pol, "缴费金额（年，上限）")
        r["缴费档次与补贴明细"] = pol.get("缴费档次与补贴明细（年，分档次）") if pd.notna(pol.get("缴费档次与补贴明细（年，分档次）")) else "不清楚"
        r["基础养老金"]         = safe_pol_int(pol, "基础养老金（年）")
        
        # Numeric values for burden-rate calculations
        min_jumin   = safe_pol_float(pol, "最低缴费金额（年）")
        min_zhigong = safe_pol_float(pol, "缴费金额（年，下限）")

    # Precompute burden rates using the largest positive denominator.
    hhsize = r["家庭人数"] if isinstance(r["家庭人数"], int) and r["家庭人数"] > 0 else 1

    # Compute three candidate denominators.
    denom_personal = personal_income_num if not pd.isna(personal_income_num) and personal_income_num > 0 else np.nan
    denom_family = total_income / hhsize if not pd.isna(total_income) and total_income > 0 and hhsize > 0 else np.nan
    net_asset = (total_asset - (total_debt if not pd.isna(total_debt) else 0)) if not pd.isna(total_asset) else np.nan
    denom_net_asset = net_asset / hhsize if not pd.isna(net_asset) and net_asset > 0 and hhsize > 0 else np.nan

    # Compute three burden rates.
    def calc_rate(fee, denom):
        if np.isnan(fee) or fee <= 0 or np.isnan(denom) or denom <= 0:
            return np.nan
        return fee / denom

    # Compute the three resident-pension burden rates.
    r1_jumin = calc_rate(min_jumin, denom_personal)
    r2_jumin = calc_rate(min_jumin, denom_family)
    r3_jumin = calc_rate(min_jumin, denom_net_asset)
    # Select the largest positive value, representing the lightest burden.
    jumin_rates = [r for r in [r1_jumin, r2_jumin, r3_jumin] if not np.isnan(r)]
    if jumin_rates:
        r["居民保负担率"] = f"{max(jumin_rates) * 100:.2f}%"
    else:
        r["居民保负担率"] = "不清楚"

    # Compute the three employee-pension burden rates.
    r1_zhigong = calc_rate(min_zhigong, denom_personal)
    r2_zhigong = calc_rate(min_zhigong, denom_family)
    r3_zhigong = calc_rate(min_zhigong, denom_net_asset)
    # Select the largest positive value, representing the lightest burden.
    zhigong_rates = [r for r in [r1_zhigong, r2_zhigong, r3_zhigong] if not np.isnan(r)]
    if zhigong_rates:
        r["职工保负担率"] = f"{max(zhigong_rates) * 100:.2f}%"
    else:
        r["职工保负担率"] = "不清楚"

    if pension_for_calc > 0 and not pd.isna(total_income) and total_income > 0:
        r["家庭养老金依赖度"] = f"{pension_for_calc * 12 / total_income * 100:.2f}%"
    elif pension_for_calc == 0:
        r["家庭养老金依赖度"] = "0.00%"
    else:
        r["家庭养老金依赖度"] = "不清楚"

    records.append(r)

log(f"  Reconstruction complete: {len(records)} records")
if warn_no_pol:
    log(f"  Records without a matching policy province: {warn_no_pol}")

# ==================== 10. Output ====================
log("\n[6] Saving results...")

FINAL_COLS = [
    "家庭ID", "个人ID", "年龄", "性别", "文化程度", "健康状况",
    "户口性质", "户口省份", "常住省份", "是否流动",
    "工作性质", "工作行业", "工作职业", "单位类型",
    "个人年收入", "家庭总收入", "家庭总消费", "家庭总资产", "家庭总负债",
    "家庭人数", "子女数", "老人数",
    "风险偏好", "经济预期",
    "居民保负担率", "职工保负担率", "家庭养老金依赖度",
    "城乡", "地区", "收入分组",
    "历史参保类型", "累计缴纳年限", "累计缴纳年限（原始n值）", "是否存在断缴",
    "家庭参保人数", "家庭领取人数", "家庭月均养老金",
    "缴费档次与补贴明细", "基础养老金", "社平工资",
    "缴费指数下限", "缴费指数上限", "可选档次规则",
    "缴费基数下限", "缴费基数上限", "缴费金额下限", "缴费金额上限",
    "参保决策", "参保账户",
]

out_df = pd.DataFrame(records, columns=FINAL_COLS)
out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
log(f"  Saved: {OUT_CSV}")
log(f"  Rows: {len(out_df)}; columns: {len(out_df.columns)}")

# Summary statistics
log("\n[7] Summary statistics")
log(f"  Enrollment-decision distribution:\n{out_df['参保决策'].value_counts().to_string()}")
log(f"\n  Pension-account distribution:\n{out_df['参保账户'].value_counts().to_string()}")
log(f"\n  Urban-rural distribution:\n{out_df['城乡'].value_counts().to_string()}")
log(f"\n  Region distribution:\n{out_df['地区'].value_counts().to_string()}")
log(f"\n  Income-group distribution:\n{out_df['收入分组'].value_counts().to_string()}")
log(f"\n  Resident-scheme burden marked '不清楚': {(out_df['居民保负担率']=='不清楚').sum()}/{len(out_df)}")
log(f"  Employee-scheme burden marked '不清楚': {(out_df['职工保负担率']=='不清楚').sum()}/{len(out_df)}")

# Version 2: accumulated contribution-year statistics
log("\n[8] Inferred accumulated contribution years (v2)")
known_type = out_df[out_df["历史参保类型"].isin(["城镇职工养老保险", "城乡居民养老保险"])]
computable = known_type[known_type["累计缴纳年限"] != "不清楚"]
log(f"  Records with a known historical pension channel: {len(known_type)}")
log(f"  Records with computable years: {len(computable)} ({len(computable)/max(len(known_type),1)*100:.1f}%)")
log(f"  Records with uncomputable years ('不清楚'): {len(known_type)-len(computable)}")
if len(computable) > 0:
    log(f"  Accumulated-contribution-year distribution:\n{computable['累计缴纳年限'].value_counts().sort_index().to_string()}")
    raw_vals = pd.to_numeric(known_type["累计缴纳年限（原始n值）"], errors="coerce").dropna()
    if len(raw_vals) > 0:
        log(f"  Raw inferred n range: min={raw_vals.min():.2f}, max={raw_vals.max():.2f}, mean={raw_vals.mean():.2f}")

log("\nComplete")
_log.close()
