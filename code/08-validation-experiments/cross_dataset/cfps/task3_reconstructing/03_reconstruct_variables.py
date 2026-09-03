#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_reconstruct_variables.py
Reconstruct all variables required by the prompts for the 500 cases sampled in
task 2, and write reconstructed_500.csv.

Data sources:
  - cfps2018person_202512.dta     (individual records)
  - cfps2018famecon_202512.dta    (household finances)
  - cfps2018famconf_202512.dta    (household roster and composition)
  - data/raw/policy/province_policy_2018.xlsx (2018 provincial pension policy)

Key differences from CHIP/CHFS 2017:
  - Gender codes: 0=女, 1=男 (CHIP uses 1=男, 2=女)
  - Industry code: qg302code (integers 1-20 in national category order)
  - Occupation code: qg303code (five digits; first digit is the major group)
  - Assets and debts: sum seven components from famecon
  - Pension: fn301 annual total divided by 12 for the monthly average
  - Household composition: famconf rows with alive_a18_p==1 and tb1y_a_p>0
  - Household enrollment and receipt: all person rows using qi301_a_* and qi2001
  - 历史参保类型: always "不清楚" because historical data are unavailable
  - 累计缴纳年限: always "不清楚"
  - 是否存在断缴: always "不清楚"
"""

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
import openpyxl

# Paths
SCRIPT_DIR   = Path(__file__).resolve().parent
TASK1_DIR    = SCRIPT_DIR.parent / "task1"
TASK2_DIR    = SCRIPT_DIR.parent / "task2_sampling"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from common.household_enrollment import count_other_enrolled_members

CFPS_DIR     = PROJECT_ROOT / "data" / "raw" / "cfps2018"
PERSON_DTA   = CFPS_DIR / "cfps2018person_202512.dta"
FAMECON_DTA  = CFPS_DIR / "cfps2018famecon_202512.dta"
FAMCONF_DTA  = CFPS_DIR / "cfps2018famconf_202512.dta"
POLICY_XLS   = PROJECT_ROOT / "data" / "raw" / "policy" / "province_policy_2018.xlsx"
OUT_CSV      = SCRIPT_DIR / "reconstructed_500.csv"
LOG_PATH     = SCRIPT_DIR / "03_reconstruct_log.txt"

# Logging
_log = open(LOG_PATH, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log.write(msg + "\n")
    _log.flush()

log("=" * 70)
log("Task 3  CFPS 2018 variable reconstruction")
log("=" * 70)

# Static mappings
PROV_CODE_TO_NAME = {
    11: "北京市",   12: "天津市",   13: "河北省",   14: "山西省",
    15: "内蒙古自治区", 21: "辽宁省", 22: "吉林省",   23: "黑龙江省",
    31: "上海市",   32: "江苏省",   33: "浙江省",   34: "安徽省",
    35: "福建省",   36: "江西省",   37: "山东省",   41: "河南省",
    42: "湖北省",   43: "湖南省",   44: "广东省",   45: "广西壮族自治区",
    46: "海南省",   50: "重庆市",   51: "四川省",   52: "贵州省",
    53: "云南省",   54: "西藏自治区", 61: "陕西省",  62: "甘肃省",
    63: "青海省",   64: "宁夏回族自治区", 65: "新疆维吾尔自治区",
}
REGION_MAP = {
    11: "东部", 12: "东部", 13: "东部", 31: "东部", 32: "东部",
    33: "东部", 35: "东部", 37: "东部", 44: "东部", 46: "东部",
    14: "中部", 34: "中部", 36: "中部", 41: "中部", 42: "中部", 43: "中部",
    15: "西部", 45: "西部", 50: "西部", 51: "西部", 52: "西部",
    53: "西部", 54: "西部", 61: "西部", 62: "西部", 63: "西部",
    64: "西部", 65: "西部",
    21: "东北", 22: "东北", 23: "东北",
}
EDU_MAP = {
    1: "文盲/半文盲", 2: "小学", 3: "初中", 4: "高中/中专",
    5: "大专", 6: "本科", 7: "硕士", 8: "博士",
}
HEALTH_MAP = {
    1: "非常健康", 2: "很健康", 3: "比较健康", 4: "一般", 5: "不健康",
}
HUKOU_MAP = {1: "农业户口", 3: "非农业户口", 5: "无户口"}
IND_MAP = {
    1: "农林牧渔", 2: "采矿业", 3: "制造业", 4: "电力热力燃气水",
    5: "建筑业", 6: "批发零售", 7: "交通运输邮政", 8: "住宿餐饮",
    9: "信息软件", 10: "金融业", 11: "房地产", 12: "租赁商务服务",
    13: "科学研究", 14: "水利环境", 15: "居民服务",
    16: "教育", 17: "卫生社会工作", 18: "文化体育娱乐",
    19: "公共管理", 20: "国际组织",
}
OCC_MAP = {  # First digit of qg303code.
    1: "负责人", 2: "专业技术人员", 3: "办事人员", 4: "商业服务业",
    5: "农林牧渔生产", 6: "生产运输操作", 7: "军人",
}
UNIT_MAP = {  # qg2
    1: "党政机关", 2: "事业单位", 3: "国有企业", 4: "私营/个体",
    5: "外资/港澳台", 6: "其他企业", 7: "个人/家庭", 8: "社会组织",
    9: "不清楚", 77: "其他",
}
RISK_MAP_QBB = {  # Collapse qbb001 choices 1-7 into five risk categories.
    1: "极度厌恶风险",
    2: "略低风险偏好", 3: "略低风险偏好",
    4: "平均风险偏好",
    5: "略高风险偏好", 6: "略高风险偏好",
    7: "高风险偏好",
}
EXPECT_MAP = {  # wv108
    1: "非常悲观", 2: "悲观", 3: "乐观", 4: "非常乐观", 5: "不确定",
}

MISSING_VALS = {-8, -9, -1, -2, -88, -99}

def is_missing(v):
    if v is None: return True
    if isinstance(v, float) and math.isnan(v): return True
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

def fmt_yuan(v):
    if pd.isna(v): return "不清楚"
    try: return f"{int(v)}元"
    except: return "不清楚"

def safe_fill(v, default=0.0):
    """Replace missing values with default for aggregation."""
    if v is None or (isinstance(v, float) and math.isnan(v)): return default
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default

# 1. Load source data
log("\n[1] Loading data (DTA files may take a moment)...")

sampled = pd.read_csv(TASK2_DIR / "sampled_500.csv")
log(f"  sampled_500: {len(sampled)} rows")
target_pids  = set(sampled["pid"])
target_fid18 = set(sampled["fid18"])

filtered_flex = pd.read_csv(TASK1_DIR / "filtered_flexible_workers.csv")
log(f"  filtered_flexible_workers: {len(filtered_flex)} rows")
flex_pids = set(filtered_flex["pid"])

# Load required fields from the person table.
PERSON_COLS = [
    "pid", "fid18",
    "gender", "age",            # Individual characteristics.
    "urban18",                  # Urban-rural status.
    "cfps2018edu",              # Education.
    "qp201",                    # Health.
    "qa301",                    # Hukou type.
    "qa302",                    # Hukou location type.
    "qa302a_code",              # Hukou province code.
    "ear201a",                  # Residence province code.
    "provcd18",                 # Six-digit residence county code.
    "jobclass_base", "qg5",     # Employment status.
    "qg302code",                # Industry.
    "qg303code",                # Occupation.
    "qg2",                      # Employer type.
    "emp_income",               # Individual income.
    "qn101", "qn102", "qn103",  # Primary risk experiment.
    "qn104", "qn105",
    "qbb001",                   # Backup lottery risk measure.
    "wv108",                    # Economic expectations.
    "qi301_a_2", "qi301_a_5",   # Pension status for household statistics.
    "qi301_a_6", "qi301_a_7",
    "qi2001",                   # Pension receipt for household statistics.
]
person_all, _ = pyreadstat.read_dta(str(PERSON_DTA), usecols=PERSON_COLS)
log(f"  person table (selected columns): {len(person_all)} rows x {len(person_all.columns)} columns")

# Household economic table.
FAMECON_COLS = [
    "fid18",
    "finc", "fexp",                                    # Total income and expenditure.
    "houseasset_gross", "fm401", "fixed_asset",        # Asset components.
    "finance_asset", "durables_asset", "debit_other", "land_asset",
    "house_debts", "nonhousing_debts",                 # Debt components.
    "ft501", "ft601", "ft602",
    "fn301",                                           # Annual household pension total.
]
famecon_all, _ = pyreadstat.read_dta(str(FAMECON_DTA), usecols=FAMECON_COLS)
famecon_target = famecon_all[famecon_all["fid18"].isin(target_fid18)].copy()
log(f"  famecon table: {len(famecon_all)} rows ({len(famecon_target)} target-household rows)")

# Household roster.
FAMCONF_COLS = ["fid18", "pid", "alive_a18_p", "tb1y_a_p"]
famconf_all, _ = pyreadstat.read_dta(str(FAMCONF_DTA), usecols=FAMCONF_COLS)
famconf_target = famconf_all[famconf_all["fid18"].isin(target_fid18)].copy()
log(f"  famconf table: {len(famconf_all)} rows ({len(famconf_target)} target-household rows)")

# Policy workbook.
log(f"  Loading policy workbook...")
wb = openpyxl.load_workbook(POLICY_XLS, data_only=True)
ws = wb["总表"]
all_rows  = list(ws.iter_rows(values_only=True))
header    = [str(c) if c is not None else f"_col{i}" for i, c in enumerate(all_rows[0])]
policy_df = pd.DataFrame(all_rows[1:], columns=header)
policy_df = policy_df[policy_df["省名"].notna()].copy()

def safe_float(v):
    try:
        return float(v) if pd.notna(v) else np.nan
    except:
        return np.nan

# Recompute formula cells that openpyxl may return as None under data_only.
for idx, row in policy_df.iterrows():
    wage     = safe_float(row.get("社平工资（年）"))
    idx_low  = safe_float(row.get("缴费指数（下限）"))
    idx_high = safe_float(row.get("缴费指数（上限）"))
    ratio    = safe_float(row.get("缴费比例"))

    base_low = row.get("缴费基数（年，下限）")
    if pd.isna(base_low) and not math.isnan(wage) and not math.isnan(idx_low):
        base_low = wage * idx_low
        policy_df.at[idx, "缴费基数（年，下限）"] = base_low
    else:
        base_low = safe_float(base_low)

    base_high = row.get("缴费基数（年，上限）")
    if pd.isna(base_high) and not math.isnan(wage) and not math.isnan(idx_high):
        base_high = wage * idx_high
        policy_df.at[idx, "缴费基数（年，上限）"] = base_high
    else:
        base_high = safe_float(base_high)

    if pd.isna(row.get("缴费金额（年，下限）")) and not math.isnan(base_low) and not math.isnan(ratio):
        policy_df.at[idx, "缴费金额（年，下限）"] = base_low * ratio
    if pd.isna(row.get("缴费金额（年，上限）")) and not math.isnan(base_high) and not math.isnan(ratio):
        policy_df.at[idx, "缴费金额（年，上限）"] = base_high * ratio

policy_dict = {row["省名"]: row for _, row in policy_df.iterrows()}
log(f"  Policy table: {len(policy_dict)} provinces")

# 2. Individual-income terciles over all flexible workers
log("\n[2] Computing individual annual-income terciles for all flexible workers...")
person_flex = person_all[person_all["pid"].isin(flex_pids)].copy()
valid_emp = person_flex["emp_income"].dropna()
valid_emp = valid_emp[valid_emp > 0]
q33 = valid_emp.quantile(1/3)
q67 = valid_emp.quantile(2/3)
log(f"  1/3 quantile: {q33:,.0f} yuan | 2/3 quantile: {q67:,.0f} yuan (valid cases: {len(valid_emp):,})")

def income_tercile(v):
    if pd.isna(v) or v <= 0: return "不清楚"
    if v <= q33: return "低收入"
    elif v <= q67: return "中收入"
    else: return "高收入"

# 3. Household composition from famconf
log("\n[3] Computing household composition from famconf...")

fc = famconf_target.copy()
fc = fc[(fc["alive_a18_p"] == 1) & (fc["tb1y_a_p"] > 0)].copy()
fc["_age"] = 2018 - fc["tb1y_a_p"]
log(f"  Living members with valid birth year: {len(fc)} people, {fc['fid18'].nunique()} households")

def agg_pop(grp):
    size     = len(grp)
    children = int((grp["_age"] < 16).sum())
    elderly  = int((grp["_age"] >= 60).sum())
    return pd.Series({"家庭人数": size, "子女数": children, "老人数": elderly})

pop_stats = fc.groupby("fid18").apply(agg_pop).reset_index()
log(f"  Household composition complete: {len(pop_stats)} households")

# 4. Household enrollment and pension receipt from the full person table
log("\n[4] Computing household enrollment and receipt counts from all person rows...")

person_fam = person_all[person_all["fid18"].isin(target_fid18)].copy()
log(f"  Person records in target households: {len(person_fam)}")

def is_insured(row):
    """Return whether the person is enrolled and contributing, not receiving."""
    return (
        row["qi301_a_2"] == 1 or row["qi301_a_5"] == 1 or
        row["qi301_a_6"] == 1 or row["qi301_a_7"] == 1
    )

def is_receiving(row):
    """Return whether the person currently receives a pension."""
    return row["qi2001"] == 1

person_fam["_insured"]   = person_fam.apply(is_insured, axis=1)
person_fam["_receiving"] = person_fam.apply(is_receiving, axis=1)
person_fam["_enrollment_count"] = (
    person_fam["_insured"].astype(int) + person_fam["_receiving"].astype(int)
)

def agg_ins(grp):
    receiving = int(grp["_receiving"].sum())
    return pd.Series({
        "家庭领取人数": receiving,
    })

ins_stats = person_fam.groupby("fid18").apply(agg_ins).reset_index()
log(f"  Household enrollment and receipt counts complete: {len(ins_stats)} households")

# 5. Join reconstructed inputs
log("\n[5] Joining data for 500 cases...")

person_500 = person_all[person_all["pid"].isin(target_pids)].copy()
log(f"  person_500: {len(person_500)} rows")

# Start from sampled cases, retaining ins_type and stratum.
df = sampled[["pid", "fid18", "ins_type", "stratum"]].merge(
    person_500, on=["pid", "fid18"], how="left"
)
# Join household economic fields.
df = df.merge(famecon_target, on="fid18", how="left")
# Join household composition.
df = df.merge(pop_stats, on="fid18", how="left")
# Join household enrollment and receipt counts.
df = df.merge(ins_stats, on="fid18", how="left")
df["家庭参保人数"] = count_other_enrolled_members(
    person_fam,
    df,
    household_col="fid18",
    person_col="pid",
    enrolled_col="_enrollment_count",
)

log(f"  Join complete: {len(df)} rows x {len(df.columns)} columns")

# 6. Mapping helpers
def map_gender(v):
    return {0: "女", 1: "男"}.get(int(v) if not is_missing(v) else -1, "不清楚")

def map_work_cfps(jobclass, qg5):
    try:
        j = int(jobclass)
        if j == 2:
            return "个体/自雇"
        elif j == 4:
            q = int(qg5) if not is_missing(qg5) else -1
            if q == 0:
                return "受雇（无劳动合同）"
    except (ValueError, TypeError):
        pass
    return "不清楚"

def map_occ_cfps(v):
    if is_missing(v): return "不清楚"
    try:
        code = int(v)
        if code <= 0: return "不清楚"
        first = int(str(code)[0])
        return OCC_MAP.get(first, "不清楚")
    except:
        return "不清楚"

def _to_prov_int(code_val):
    """Convert a province code to a positive integer, or return None."""
    try:
        v = int(code_val)
        return v if v > 0 else None
    except:
        return None

def get_province_name(code_val):
    p = _to_prov_int(code_val)
    if p is None: return "不清楚"
    return PROV_CODE_TO_NAME.get(p, "不清楚")

def get_hukou_prov_code(qa302a_code, qa302, provcd18):
    """
    Infer the two-digit hukou province code, such as 11, 34, or 44.

    Use a valid qa302a_code (>0), otherwise fall back to provcd18, which is
    already a two-digit province code in CFPS.
    """
    p = _to_prov_int(qa302a_code)
    if p is not None:
        return p
    # provcd18 is already a two-digit province code in CFPS.
    return _to_prov_int(provcd18)

def get_res_prov_code(ear201a, provcd18):
    """Infer the two-digit province code of residence."""
    p = _to_prov_int(ear201a)
    if p is not None:
        return p
    return _to_prov_int(provcd18)

def map_risk_cfps(qn101, qn102, qn103, qn104, qn105, qbb001):
    """
    Infer five-level risk preference from the qn101-qn105 decision tree.

      CE <= 50         -> 极度厌恶风险
      50 < CE <= 80    -> 略低风险偏好
      80 < CE <= 100   -> 平均风险偏好
      100 < CE <= 120  -> 略高风险偏好
      CE > 120         -> 高风险偏好
    Fall back to qbb001 lottery choices, collapsed from seven to five levels.
    """
    def si(v):
        if is_missing(v): return None
        try: return int(v)
        except: return None

    v1 = si(qn101)
    if v1 == 1:                          # Certain 100-yuan option: risk-averse branch.
        v2 = si(qn102)
        if v2 == 1:                      # Certain 80-yuan option.
            v3 = si(qn103)
            if v3 == 1: return "极度厌恶风险"   # CE ≤ 50
            if v3 == 5: return "略低风险偏好"   # 50 < CE ≤ 80
        elif v2 == 5:
            return "平均风险偏好"               # 80 < CE ≤ 100
    elif v1 == 5:                        # Lottery option: risk-seeking branch.
        v4 = si(qn104)
        if v4 == 1: return "略高风险偏好"       # 100 < CE ≤ 120
        if v4 == 5: return "高风险偏好"         # CE > 120, including both qn105 branches.

    # Fall back to qbb001.
    bb = si(qbb001)
    if bb is not None:
        return RISK_MAP_QBB.get(bb, "不清楚")
    return "不清楚"

def match_policy(province_name):
    if not isinstance(province_name, str) or province_name in ("", "不清楚"):
        return None
    pol = policy_dict.get(province_name)
    if pol is not None:
        return pol
    # Match province abbreviations against full names.
    ABBR = {
        "新疆": "新疆维吾尔自治区", "广西": "广西壮族自治区",
        "内蒙古": "内蒙古自治区",   "宁夏": "宁夏回族自治区",
        "西藏": "西藏自治区",
    }
    for abbr, full in ABBR.items():
        if abbr in province_name:
            pol = policy_dict.get(full)
            if pol is not None:
                return pol
    return None

def safe_pol_int(pol, key):
    v = pol.get(key)
    try:
        if pd.isna(v) or v is None: return "不清楚"
        return int(float(v))
    except:
        return "不清楚"

def safe_pol_float(pol, key):
    v = pol.get(key)
    try:
        if pd.isna(v) or v is None: return np.nan
        return float(v)
    except:
        return np.nan

# 7. Reconstruct sampled cases
log("\n[6] Reconstructing variables row by row...")

records     = []
warn_no_pol = set()

for _, row in df.iterrows():
    r = {}

    # Identifiers.
    r["家庭ID"] = str(int(row["fid18"]))
    r["个人ID"] = str(int(row["pid"]))

    # Individual characteristics.
    r["年龄"]     = int(row["age"]) if not is_missing(row.get("age")) else "不清楚"
    r["性别"]     = map_gender(row.get("gender"))
    r["文化程度"] = safe_map(row.get("cfps2018edu"), EDU_MAP)
    r["健康状况"] = safe_map(row.get("qp201"), HEALTH_MAP)

    # Hukou and migration.
    r["户口性质"] = safe_map(row.get("qa301"), HUKOU_MAP)
    hk_code  = get_hukou_prov_code(row.get("qa302a_code"), row.get("qa302"), row.get("provcd18"))
    res_code = get_res_prov_code(row.get("ear201a"), row.get("provcd18"))
    hk_prov  = PROV_CODE_TO_NAME.get(hk_code, "不清楚") if hk_code else "不清楚"
    res_prov = PROV_CODE_TO_NAME.get(res_code, "不清楚") if res_code else "不清楚"
    r["户口省份"] = hk_prov
    r["常住省份"] = res_prov
    if hk_code is None or res_code is None:
        r["是否流动"] = "不清楚"
    else:
        r["是否流动"] = "否" if hk_code == res_code else "是"

    # Employment.
    r["工作性质"] = map_work_cfps(row.get("jobclass_base"), row.get("qg5"))
    r["工作行业"] = safe_map(row.get("qg302code"), IND_MAP)
    r["工作职业"] = map_occ_cfps(row.get("qg303code"))
    r["单位类型"] = safe_map(row.get("qg2"), UNIT_MAP)

    # Individual income.
    emp_income_num = pd.to_numeric(row.get("emp_income"), errors="coerce")
    if pd.isna(emp_income_num) or emp_income_num <= 0:
        r["个人年收入"]  = "不清楚"
        emp_income_num   = np.nan
    else:
        r["个人年收入"] = f"{int(emp_income_num)}元"

    # Household finances.
    finc = pd.to_numeric(row.get("finc"), errors="coerce")
    fexp = pd.to_numeric(row.get("fexp"), errors="coerce")
    r["家庭总收入"] = fmt_yuan(finc)
    r["家庭总消费"] = fmt_yuan(fexp)

    # Total assets sum seven components, treating missing values as zero.
    total_asset = (
        safe_fill(row.get("houseasset_gross"))
        + safe_fill(row.get("fm401")) * 10000        # Ten thousand yuan to yuan.
        + safe_fill(row.get("fixed_asset"))
        + safe_fill(row.get("finance_asset"))
        + safe_fill(row.get("durables_asset"))
        + safe_fill(row.get("debit_other"))
        + safe_fill(row.get("land_asset"))
    )
    # Total debt sums five components, treating missing values as zero.
    total_debt = (
        safe_fill(row.get("house_debts"))
        + safe_fill(row.get("nonhousing_debts"))
        + safe_fill(row.get("ft501"))
        + safe_fill(row.get("ft601"))
        + safe_fill(row.get("ft602"))
    )
    # Treat all-zero asset components as unknown because they may indicate missing data.
    asset_cols_raw = [row.get(c) for c in
        ["houseasset_gross","fm401","fixed_asset","finance_asset",
         "durables_asset","debit_other","land_asset"]]
    all_asset_missing = all(pd.isna(v) for v in asset_cols_raw)
    r["家庭总资产"] = "不清楚" if all_asset_missing else f"{int(total_asset)}元"
    r["家庭总负债"] = f"{int(total_debt)}元"

    # Household composition.
    r["家庭人数"] = int(row["家庭人数"]) if not pd.isna(row.get("家庭人数")) else "不清楚"
    r["子女数"]   = int(row["子女数"])   if not pd.isna(row.get("子女数"))   else "不清楚"
    r["老人数"]   = int(row["老人数"])   if not pd.isna(row.get("老人数"))   else "不清楚"

    # Ground-truth pension status from task 1.
    ins_type = row["ins_type"]
    if ins_type == "不参保":
        r["参保决策"] = "不参保"
        r["参保账户"] = "不参保"
    else:
        r["参保决策"] = "参保"
        r["参保账户"] = ins_type  # Employee or resident scheme.

    # Historical enrollment fields are unavailable in CFPS.
    r["历史参保类型"] = "不清楚"
    r["累计缴纳年限"] = "不清楚"
    r["是否存在断缴"] = "不清楚"

    # Household enrollment, receipt, and pension amount.
    r["家庭参保人数"] = int(row["家庭参保人数"]) if not pd.isna(row.get("家庭参保人数")) else 0
    r["家庭领取人数"] = int(row["家庭领取人数"]) if not pd.isna(row.get("家庭领取人数")) else 0

    fn301_num = pd.to_numeric(row.get("fn301"), errors="coerce")
    if pd.isna(fn301_num) or fn301_num <= 0:
        pension_monthly = 0.0
        r["家庭月均养老金"] = "未领取"
    else:
        pension_monthly = fn301_num / 12.0
        r["家庭月均养老金"] = f"{int(pension_monthly)}元"

    # Behavioral variables.
    r["风险偏好"] = map_risk_cfps(
        row.get("qn101"), row.get("qn102"), row.get("qn103"),
        row.get("qn104"), row.get("qn105"), row.get("qbb001")
    )
    r["经济预期"] = safe_map(row.get("wv108"), EXPECT_MAP)

    # Region.
    r["城乡"] = {0: "农村", 1: "城镇"}.get(
        int(row["urban18"]) if not is_missing(row.get("urban18")) else -1, "不清楚"
    )
    r["地区"] = REGION_MAP.get(hk_code, "不清楚") if hk_code else "不清楚"

    # Income group.
    r["收入分组"] = income_tercile(emp_income_num)

    # Policy variables, prioritizing hukou province over residence province.
    pol = match_policy(hk_prov)
    if pol is None and res_prov != "不清楚":
        pol = match_policy(res_prov)
    if pol is None:
        if hk_prov != "不清楚":
            warn_no_pol.add(hk_prov)
        for k in ["社平工资", "缴费指数下限", "缴费指数上限",
                  "缴费基数下限", "缴费基数上限", "可选档次规则",
                  "缴费金额下限", "缴费金额上限", "缴费档次与补贴明细", "基础养老金"]:
            r[k] = "不清楚"
        min_jumin   = np.nan
        min_zhigong = np.nan
    else:
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
        min_jumin   = safe_pol_float(pol, "最低缴费金额（年）")
        min_zhigong = safe_pol_float(pol, "缴费金额（年，下限）")

    # Burden ratios use the largest positive candidate denominator.
    hhsize = r["家庭人数"] if isinstance(r["家庭人数"], int) and r["家庭人数"] > 0 else 1
    denom_personal   = emp_income_num if not pd.isna(emp_income_num) and emp_income_num > 0 else np.nan
    denom_fam_pc     = finc / hhsize if not pd.isna(finc) and finc > 0 else np.nan
    net_asset        = total_asset - total_debt
    denom_asset_pc   = net_asset / hhsize if (not all_asset_missing and net_asset > 0) else np.nan

    def best_denom():
        vals = [v for v in [denom_personal, denom_fam_pc, denom_asset_pc] if not math.isnan(v)]
        return max(vals) if vals else np.nan

    denom = best_denom()

    def burden_rate(fee):
        if math.isnan(fee) or math.isnan(denom) or denom <= 0: return "不清楚"
        return f"{fee / denom * 100:.2f}%"

    r["居民保负担率"] = burden_rate(min_jumin)
    r["职工保负担率"] = burden_rate(min_zhigong)

    # Household pension dependence.
    fn301_annual = fn301_num if not pd.isna(fn301_num) else 0.0
    if fn301_annual == 0:
        r["家庭养老金依赖度"] = "0.00%"
    elif not pd.isna(finc) and finc > 0:
        r["家庭养老金依赖度"] = f"{fn301_annual / finc * 100:.2f}%"
    else:
        r["家庭养老金依赖度"] = "不清楚"

    records.append(r)

log(f"  Reconstruction complete: {len(records)} records")
if warn_no_pol:
    log(f"  Warning: provinces not matched in the policy table: {warn_no_pol}")

# 8. Save reconstructed cases
log("\n[7] Saving results...")

FINAL_COLS = [
    "家庭ID", "个人ID", "年龄", "性别", "文化程度", "健康状况",
    "户口性质", "户口省份", "常住省份", "是否流动",
    "工作性质", "工作行业", "工作职业", "单位类型",
    "个人年收入", "家庭总收入", "家庭总消费", "家庭总资产", "家庭总负债",
    "家庭人数", "子女数", "老人数",
    "风险偏好", "经济预期",
    "居民保负担率", "职工保负担率", "家庭养老金依赖度",
    "城乡", "地区", "收入分组",
    "历史参保类型", "累计缴纳年限", "是否存在断缴",
    "家庭参保人数", "家庭领取人数", "家庭月均养老金",
    "缴费档次与补贴明细", "基础养老金", "社平工资",
    "缴费指数下限", "缴费指数上限", "可选档次规则",
    "缴费基数下限", "缴费基数上限", "缴费金额下限", "缴费金额上限",
    "参保决策", "参保账户",
]

out_df = pd.DataFrame(records, columns=FINAL_COLS)
out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
log(f"  Saved: {OUT_CSV}")
log(f"  Rows: {len(out_df)}  Columns: {len(out_df.columns)}")

# Summary statistics
log("\n[8] Summary statistics")
log(f"  Enrollment decision:\n{out_df['参保决策'].value_counts().to_string()}")
log(f"\n  Enrollment account:\n{out_df['参保账户'].value_counts().to_string()}")
log(f"\n  Urban-rural status:\n{out_df['城乡'].value_counts().to_string()}")
log(f"\n  Region:\n{out_df['地区'].value_counts().to_string()}")
log(f"\n  Income group:\n{out_df['收入分组'].value_counts().to_string()}")
log(f"\n  Employment type:\n{out_df['工作性质'].value_counts().to_string()}")
log(f"\n  Risk preference = '不清楚': {(out_df['风险偏好']=='不清楚').sum()}/{len(out_df)}")
log(f"  Economic expectations = '不清楚': {(out_df['经济预期']=='不清楚').sum()}/{len(out_df)}")
log(f"  Resident-scheme burden = '不清楚': {(out_df['居民保负担率']=='不清楚').sum()}/{len(out_df)}")
log(f"  Employee-scheme burden = '不清楚': {(out_df['职工保负担率']=='不清楚').sum()}/{len(out_df)}")
log(f"\n  Monthly household pension = '未领取': {(out_df['家庭月均养老金']=='未领取').sum()}/{len(out_df)}")

log("\nDone.")
_log.close()
