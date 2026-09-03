"""
04_reconstruct_variables.py
Reconstruct all variables required by the prompts for the 500 cases sampled in
task 2, and write reconstructed_500.csv. Mirror runtime logs to
04_reconstruct_log.txt.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import openpyxl

# Mirror logs to a file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from common.household_enrollment import count_other_enrolled_members

LOG_PATH = os.path.join(BASE_DIR, "04_reconstruct_log.txt")
_log_file = open(LOG_PATH, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log_file.write(msg + "\n")
    _log_file.flush()

log("=" * 70)
log("Task 3  Variable reconstruction")
log("=" * 70)

# Paths
CHIP_DIR   = str(PROJECT_ROOT / "data" / "raw" / "chip2018")
TASK1_DIR  = os.path.join(BASE_DIR, "../task1")
TASK2_DIR  = os.path.join(BASE_DIR, "../task2_sampling")
POLICY_XLS = str(PROJECT_ROOT / "data" / "raw" / "policy" / "province_policy_2018.xlsx")
OUT_CSV    = os.path.join(BASE_DIR, "reconstructed_500.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 0. Static mappings.
# ═══════════════════════════════════════════════════════════════════════════
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
# The policy workbook uses "新疆"; province mappings follow that form.

REGION_MAP = {
    # Eastern.
    11: "东部", 12: "东部", 13: "东部", 31: "东部", 32: "东部",
    33: "东部", 35: "东部", 37: "东部", 44: "东部", 46: "东部",
    # Central.
    14: "中部", 34: "中部", 36: "中部", 41: "中部", 42: "中部", 43: "中部",
    # Western.
    15: "西部", 45: "西部", 50: "西部", 51: "西部", 52: "西部",
    53: "西部", 54: "西部", 61: "西部", 62: "西部", 63: "西部",
    64: "西部", 65: "西部",
    # Northeastern.
    21: "东北", 22: "东北", 23: "东北",
}

EDU_MAP = {
    1: "未上过学", 2: "小学", 3: "初中", 4: "高中",
    5: "职高/技校", 6: "中专", 7: "大专", 8: "大学本科", 9: "研究生",
}
HEALTH_MAP = {1: "非常好", 2: "好", 3: "一般", 4: "不好", 5: "非常不好"}
HUKOU_MAP  = {1: "农业户口", 2: "非农业户口", 3: "居民户口", 4: "其他"}
WORK_MAP   = {1: "雇主", 2: "雇员", 3: "自营劳动者", 4: "家庭帮工"}
IND_MAP    = {
    1: "农林牧渔", 2: "采矿", 3: "制造", 4: "电燃水", 5: "建筑",
    6: "批发零售", 7: "交运邮政", 8: "宿餐", 9: "信息软件", 10: "金融",
    11: "房产", 12: "租赁商务", 13: "科研", 14: "水利环境", 15: "居民服务",
    16: "教育", 17: "卫生社会", 18: "文体娱乐", 19: "公共管理", 20: "国际组织",
}
OCC_MAP = {
    1: "负责人", 2: "专业技术人员", 3: "办事人员", 4: "商业服务业",
    5: "农林牧渔生产", 6: "生产运输操作", 7: "军人", 8: "其他",
}
# Employer-type codes differ between urban and rural files.
UNIT_URBAN = {
    1: "党政机关团体", 2: "事业单位", 3: "国有(控股)企业",
    4: "其他股份制企业", 5: "集体企业", 6: "中外合资/外商独资企业",
    7: "个体或私营企业", 8: "土地承包者", 9: "其他",
}
UNIT_RURAL = {
    1: "党政机关团体", 2: "事业单位", 3: "国有(控股)企业",
    4: "集体企业", 5: "中外合资/外商独资企业", 6: "个体或私营企业",
    7: "其他股份制企业", 8: "土地承包者", 9: "其他",
}
EXPECT_MAP = {1: "大幅度上升", 2: "小幅度上升", 3: "不变", 4: "下降"}

MISSING_VALS = {-88, -99, "-88", "-99"}

def is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    try:
        return int(v) in (-88, -99)
    except (ValueError, TypeError):
        return str(v).strip() in ("-88", "-99")

def safe_map(val, mapping):
    """Look up a value, returning '不清楚' when missing or unmatched."""
    if is_missing(val):
        return "不清楚"
    try:
        k = int(val)
    except (ValueError, TypeError):
        return "不清楚"
    return mapping.get(k, "不清楚")

def parse_multi(val):
    """Parse a multi-select field into integers; map -88, -99, or empty to None."""
    if is_missing(val):
        return None
    s = str(val).strip()
    if s in ("-88", "-99", ""):
        return None
    result = set()
    for part in s.split(","):
        part = part.strip()
        try:
            result.add(int(float(part)))
        except ValueError:
            pass
    return result if result else None

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load source data.
# ═══════════════════════════════════════════════════════════════════════════
log("\n[1] Loading data")

sampled = pd.read_csv(
    os.path.join(TASK2_DIR, "sampled_500.csv"), low_memory=False, dtype={"hhcode": str}
)
log(f"  sampled_500: {len(sampled)} rows")

filtered = pd.read_csv(
    os.path.join(TASK1_DIR, "filtered_flexible_workers.csv"),
    low_memory=False, dtype={"hhcode": str},
)
log(f"  filtered_flexible_workers: {len(filtered)} rows")

# Load full urban and rural person files and align their columns.
# E07 and E01_2 are urban-only; neither source includes urban_rural.
RURAL_COLS = ["hhcode", "idcode", "A04_1", "A23", "A20"]
URBAN_COLS = ["hhcode", "idcode", "A04_1", "A23", "A20", "E07", "E01_2"]
rural_p = pd.read_stata(os.path.join(CHIP_DIR, "chip2018_rural_person.dta"),
                        columns=RURAL_COLS)
urban_p = pd.read_stata(os.path.join(CHIP_DIR, "chip2018_urban_person.dta"),
                        columns=URBAN_COLS)
rural_p["urban_rural"] = 0
rural_p["E07"]   = np.nan   # Field absent from rural data.
rural_p["E01_2"] = np.nan
urban_p["urban_rural"] = 1
all_person = pd.concat([rural_p, urban_p], ignore_index=True)
all_person["hhcode"] = all_person["hhcode"].astype(str)
log(f"  Full individual data: {len(all_person)} rows ({len(rural_p)} rural, {len(urban_p)} urban)")

# Policy workbook.
wb = openpyxl.load_workbook(POLICY_XLS, data_only=True, read_only=True)
ws = wb["总表"]
all_rows = list(ws.iter_rows(values_only=True))
header_row = [str(c) if c is not None else f"_col{i}" for i, c in enumerate(all_rows[0])]
policy_df = pd.DataFrame(all_rows[1:], columns=header_row)
policy_df = policy_df[policy_df["省名"].notna()].copy()
# Index policy rows by province name.
policy_dict = {row["省名"]: row for _, row in policy_df.iterrows()}
log(f"  Policy table: {len(policy_dict)} provinces; columns: {list(policy_df.columns[:15])}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Income terciles over all flexible workers, one row per household.
# ═══════════════════════════════════════════════════════════════════════════
log("\n[2] Computing income-tercile thresholds")

_f = filtered.copy()
_f["n3701_num"] = pd.to_numeric(_f["n3701"], errors="coerce")
_f_valid = _f[~_f["n3701_num"].apply(is_missing) & _f["n3701_num"].notna()]
_f_dedup = (
    _f_valid.sort_values(["hhcode"])
    .drop_duplicates(subset="hhcode", keep="first")
)
t33 = _f_dedup["n3701_num"].quantile(1 / 3, interpolation="linear")
t67 = _f_dedup["n3701_num"].quantile(2 / 3, interpolation="linear")
log(f"  1/3 quantile: {t33:,.2f} | 2/3 quantile: {t67:,.2f}")
log(f"  Valid households: {len(_f_dedup)}")


def income_tercile(n3701_val):
    if is_missing(n3701_val) or pd.isna(n3701_val):
        return "不清楚"
    v = float(n3701_val)
    if v <= t33:
        return "低收入组"
    elif v <= t67:
        return "中收入组"
    else:
        return "高收入组"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Household statistics from the full person data.
# ═══════════════════════════════════════════════════════════════════════════
log("\n[3] Computing household statistics")

target_hhcodes = set(sampled["hhcode"].astype(str))
fam = all_person[all_person["hhcode"].isin(target_hhcodes)].copy()
log(f"  Household members: {len(fam)} people across {fam['hhcode'].nunique()} households")

# Age.
fam["age"] = 2018 - pd.to_numeric(fam["A04_1"], errors="coerce")

# Parse A23 option membership.
def a23_contains(val, opts):
    s = parse_multi(val)
    if s is None:
        return False
    return bool(s & opts)

fam["insured"] = fam["A23"].apply(lambda v: a23_contains(v, {1, 4, 5, 6}))

# Pension receipt indicators.
def is_receiving(row):
    if row["urban_rural"] == 1:  # Urban.
        e07 = pd.to_numeric(row["E07"], errors="coerce")
        e01_2 = row["E01_2"]
        e07_ok = (not pd.isna(e07)) and (e07 > 0)
        e01_ok = (not is_missing(e01_2)) and (not pd.isna(e01_2))
        return e07_ok or e01_ok
    else:  # Rural.
        a20 = pd.to_numeric(row["A20"], errors="coerce")
        if pd.isna(a20):
            return False
        a20 = int(a20)
        if a20 in (2, 3):
            return True
        if a20 == 1:
            age = row["age"]
            has_ins = a23_contains(row["A23"], {1, 4, 5, 6})
            return (not pd.isna(age)) and (age >= 60) and has_ins
        return False

fam["receiving"] = fam.apply(is_receiving, axis=1)

# Urban monthly pension, summed from E07.
fam["e07_num"] = pd.to_numeric(fam["E07"], errors="coerce")
fam["e07_valid"] = (fam["urban_rural"] == 1) & (fam["e07_num"] > 0)

# Aggregate by household code.
def agg_family(grp):
    size     = len(grp)
    children = int((grp["age"] < 16).sum())
    elderly  = int((grp["age"] >= 60).sum())
    recv     = int(grp["receiving"].sum())
    # Monthly pension is the sum of E07 among urban members.
    pension_sum = grp.loc[grp["e07_valid"], "e07_num"].sum()
    # Check whether the household has an urban member.
    has_urban = (grp["urban_rural"] == 1).any()
    return pd.Series({
        "家庭人数": size,
        "子女数":   children,
        "老人数":   elderly,
        "家庭领取人数":     recv,
        "家庭月均养老金_数值": pension_sum if has_urban else np.nan,
        "has_urban": has_urban,
    })

family_stats = fam.groupby("hhcode").apply(agg_family).reset_index()
log(f"  Household statistics complete: {len(family_stats)} households")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Reconstruct sampled cases.
# ═══════════════════════════════════════════════════════════════════════════
log("\n[4] Reconstructing variables row by row")

# Join household statistics.
sampled = sampled.merge(family_stats, on="hhcode", how="left")
sampled["家庭参保人数"] = count_other_enrolled_members(
    fam,
    sampled,
    household_col="hhcode",
    person_col="idcode",
    enrolled_col="insured",
)

records = []

warn_no_policy = set()   # Track provinces missing from the policy workbook.

for _, row in sampled.iterrows():
    r = {}

    # Identifiers.
    r["家庭ID"] = row["hhcode"]
    r["个人ID"] = int(row["idcode"]) if not is_missing(row["idcode"]) else "不清楚"

    # Individual characteristics.
    a04 = pd.to_numeric(row["A04_1"], errors="coerce")
    r["年龄"] = int(2018 - a04) if not pd.isna(a04) else "不清楚"
    r["性别"] = safe_map(row["A03"], {1: "男", 2: "女"})
    r["文化程度"]  = safe_map(row["A13_1"], EDU_MAP)
    r["健康状况"]  = safe_map(row["A16_1"], HEALTH_MAP)
    r["户口性质"]  = safe_map(row["A10"],   HUKOU_MAP)

    # Geographic variables.
    # Residence province from the first two hhcode digits.
    try:
        res_code = int(str(row["hhcode"])[:2])
        res_prov = PROV_CODE_TO_NAME.get(res_code, "不清楚")
    except Exception:
        res_prov = "不清楚"
    r["常住省份"] = res_prov

    # Hukou province.
    a09_1 = pd.to_numeric(row["A09_1"], errors="coerce")
    a09_2 = pd.to_numeric(row["A09_2"], errors="coerce")
    if pd.isna(a09_1) or is_missing(row["A09_1"]):
        hk_prov = "不清楚"
        hk_code = None
    elif int(a09_1) == 6:
        if not pd.isna(a09_2) and not is_missing(row["A09_2"]):
            hk_code = int(a09_2)
            hk_prov = PROV_CODE_TO_NAME.get(hk_code, "不清楚")
        else:
            hk_code = None
            hk_prov = "不清楚"
    elif int(a09_1) in (1, 2, 3, 4, 5):
        hk_prov = res_prov
        hk_code = res_code if res_prov != "不清楚" else None
    else:  # Code 7: other.
        hk_prov = "不清楚"
        hk_code = None
    r["户口省份"] = hk_prov

    # Migration status.
    if pd.isna(a09_1) or is_missing(row["A09_1"]):
        r["是否流动"] = "不清楚"
    elif int(a09_1) == 6:
        r["是否流动"] = "是"
    elif int(a09_1) in (1, 2, 3, 4, 5):
        r["是否流动"] = "否"
    else:
        r["是否流动"] = "不清楚"

    # Region from hukou province code.
    r["地区"] = REGION_MAP.get(hk_code, "不清楚") if hk_code else "不清楚"

    # Employment.
    r["工作性质"] = safe_map(row["C03_1"], WORK_MAP)
    r["工作行业"] = safe_map(row["C03_3"], IND_MAP)
    r["工作职业"] = safe_map(row["C03_4"], OCC_MAP)
    ur = int(row["urban_rural"]) if not is_missing(row["urban_rural"]) else 1
    r["单位类型"] = safe_map(row["C03_2"], UNIT_URBAN if ur == 1 else UNIT_RURAL)
    r["城乡"]    = "城镇" if ur == 1 else "农村"

    # Individual income.
    def to_num(v):
        if is_missing(v):
            return 0.0
        x = pd.to_numeric(v, errors="coerce")
        return 0.0 if pd.isna(x) else float(x)

    c05 = to_num(row["C05_1"])
    c09_4 = to_num(row["C09_4"]) if ur == 1 else 0.0
    c09_6 = to_num(row["C09_6"])
    c09_8 = to_num(row["C09_8"])
    new_econ = c09_6 * c09_8

    # Check whether all income sources are missing.
    all_missing = (
        is_missing(row["C05_1"]) and
        (ur != 1 or is_missing(row["C09_4"])) and
        (is_missing(row["C09_6"]) or is_missing(row["C09_8"]))
    )
    if all_missing:
        personal_income_str = "不清楚"
        personal_income_num = np.nan
    else:
        personal_income_num = c05 + c09_4 + new_econ
        personal_income_str = f"{int(personal_income_num)}元"
    r["个人年收入"] = personal_income_str

    # Household income and consumption.
    n3701 = pd.to_numeric(row["n3701"], errors="coerce")
    n4202 = pd.to_numeric(row["n4202"], errors="coerce")
    r["家庭总收入"] = (f"{int(n3701)}元" if (not pd.isna(n3701) and not is_missing(row["n3701"]))
                    else "不清楚")
    r["家庭总消费"] = (f"{int(n4202)}元" if (not pd.isna(n4202) and not is_missing(row["n4202"]))
                    else "不清楚")
    r["家庭总资产"] = "不清楚"
    r["家庭总负债"] = "不清楚"

    # Household composition.
    r["家庭人数"] = int(row["家庭人数"]) if not pd.isna(row["家庭人数"]) else "不清楚"
    r["子女数"]   = int(row["子女数"])   if not pd.isna(row["子女数"])   else "不清楚"
    r["老人数"]   = int(row["老人数"])   if not pd.isna(row["老人数"])   else "不清楚"

    # Household enrollment and pension-receipt counts.
    r["家庭参保人数"] = int(row["家庭参保人数"]) if not pd.isna(row["家庭参保人数"]) else "不清楚"
    r["家庭领取人数"] = int(row["家庭领取人数"]) if not pd.isna(row["家庭领取人数"]) else "不清楚"

    # Average monthly household pension.
    pension_num = row["家庭月均养老金_数值"]
    if ur == 0:  # Rural household.
        r["家庭月均养老金"] = "不清楚"
        pension_for_calc  = np.nan
    elif pd.isna(pension_num):
        r["家庭月均养老金"] = "0元/月"
        pension_for_calc  = 0.0
    else:
        r["家庭月均养老金"] = f"{int(pension_num)}元/月"
        pension_for_calc  = float(pension_num)

    # Fields unavailable in CHIP.
    r["家庭总资产"]   = "不清楚"
    r["家庭总负债"]   = "不清楚"
    r["风险偏好"]     = "不清楚"
    # Historical status is assigned after ground-truth action construction.
    r["累计缴纳年限"] = "不清楚"
    r["是否存在断缴"] = "不清楚"

    # Behavioral variables and expectations.
    r["经济预期"] = safe_map(row["P07_3"], EXPECT_MAP)

    # Income group.
    r["收入分组"] = income_tercile(row["n3701"])

    # Policy variables.
    pol = policy_dict.get(hk_prov)
    if pol is None:
        if hk_prov != "不清楚":
            warn_no_policy.add(hk_prov)
        for k in ["社平工资", "缴费指数下限", "缴费指数上限", "缴费基数下限",
                  "缴费基数上限", "可选档次规则", "缴费金额下限", "缴费金额上限",
                  "缴费档次与补贴明细", "基础养老金"]:
            r[k] = "不清楚"
        min_jumin = np.nan
        min_zhigong = np.nan
    else:
        r["社平工资"]       = pol["社平工资（年）"]
        r["缴费指数下限"]   = pol["缴费指数（下限）"]
        r["缴费指数上限"]   = pol["缴费指数（上限）"]
        r["缴费基数下限"]   = pol["缴费基数（年，下限）"]
        r["缴费基数上限"]   = pol["缴费基数（年，上限）"]
        r["可选档次规则"]   = pol["可选档次规则"]
        r["缴费金额下限"]   = pol["缴费金额（年，下限）"]
        r["缴费金额上限"]   = pol["缴费金额（年，上限）"]
        r["缴费档次与补贴明细"] = pol["缴费档次与补贴明细（年，分档次）"]
        r["基础养老金"]     = pol["基础养老金（年）"]
        try:
            min_jumin   = float(pol["最低缴费金额（年）"])
            min_zhigong = float(pol["缴费金额（年，下限）"])
        except (TypeError, ValueError):
            min_jumin   = np.nan
            min_zhigong = np.nan

    # Precomputed burden metrics.
    # Use the largest positive value among individual and household income as denominator.
    inc_vals = []
    if not np.isnan(personal_income_num) and personal_income_num > 0:
        inc_vals.append(personal_income_num)
    n3701_num = float(n3701) if (not pd.isna(n3701) and not is_missing(row["n3701"])) else np.nan
    if not np.isnan(n3701_num) and n3701_num > 0:
        inc_vals.append(n3701_num)
    denom = max(inc_vals) if inc_vals else np.nan

    if not np.isnan(min_jumin) and not np.isnan(denom):
        r["居民保负担率"] = f"{min_jumin / denom * 100:.2f}%"
    else:
        r["居民保负担率"] = "不清楚"

    if not np.isnan(min_zhigong) and not np.isnan(denom):
        r["职工保负担率"] = f"{min_zhigong / denom * 100:.2f}%"
    else:
        r["职工保负担率"] = "不清楚"

    if not np.isnan(pension_for_calc) and not np.isnan(n3701_num) and n3701_num > 0:
        r["家庭养老金依赖度"] = f"{pension_for_calc * 12 / n3701_num * 100:.2f}%"
    else:
        r["家庭养老金依赖度"] = "不清楚"

    # Ground-truth enrollment action and scheme.
    a23_set = parse_multi(row["A23"])
    if a23_set is None:
        r["参保决策"] = "不清楚"
        r["参保账户"] = "不清楚"
        r["历史参保类型"] = "不清楚"
    elif a23_set & {1, 4}:
        r["参保决策"] = "参保"
        r["参保账户"] = "城镇职工养老保险"
        r["历史参保类型"] = "不清楚"  # Participates, but historical status is unavailable.
    elif a23_set & {5, 6}:
        r["参保决策"] = "参保"
        r["参保账户"] = "城乡居民养老保险"
        r["历史参保类型"] = "不清楚"  # Participates, but historical status is unavailable.
    else:
        r["参保决策"] = "不参保"
        r["参保账户"] = "不参保"
        r["历史参保类型"] = "不参保"  # Historical non-participation is observed.

    records.append(r)

log(f"  Reconstruction complete: {len(records)} records")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Save reconstructed cases.
# ═══════════════════════════════════════════════════════════════════════════
log("\n[5] Saving results")

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

# ═══════════════════════════════════════════════════════════════════════════
# 6. Print summary statistics.
# ═══════════════════════════════════════════════════════════════════════════
log("\n[6] Summary statistics")

def count_missing(col):
    return (out_df[col] == "不清楚").sum()

key_cols = ["户口省份", "地区", "工作性质", "单位类型", "个人年收入",
            "收入分组", "居民保负担率", "职工保负担率", "家庭养老金依赖度",
            "参保决策", "参保账户"]
log(f"\n{'Variable':<16} {'Count':>8} {'Rate':>8}")
log("-" * 36)
for c in key_cols:
    n = count_missing(c)
    log(f"  {c:<14} {n:>8}  {n/len(out_df)*100:>7.1f}%")

log("\nEnrollment decision distribution")
log(out_df["参保决策"].value_counts().to_string())
log("\nEnrollment account distribution")
log(out_df["参保账户"].value_counts().to_string())
log("\nIncome-group distribution")
log(out_df["收入分组"].value_counts().to_string())
log("\nUrban-rural distribution")
log(out_df["城乡"].value_counts().to_string())

if warn_no_policy:
    log(f"\nWarning: unmatched policy-table provinces were set to '不清楚': {warn_no_policy}")

log("\n" + "=" * 70)
log("Reconstruction complete")
log("=" * 70)
_log_file.close()
