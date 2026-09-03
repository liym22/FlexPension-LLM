#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate DKI prompts and ground-truth records for CHFS 2019.

Compared with the baseline prompt, DKI adds resident and employee pension
burden ratios, household pension dependency, and a five-step decision trace.
The Chinese template is retained because it is the exact model input used in
the reported experiments.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 80)
print("Generate New-Era Prompts")
print("=" * 80)

# ==================== Path configuration ====================
_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

from config.paths import (
    ALL_SAMPLES_WITH_POLICY_FILE,
    BURDEN_DATA_FILE,
    PROMPTS_CHFS2019_DKI_DIR,
    RAW_POLICY_DIR,
)

SAMPLE_CSV = ALL_SAMPLES_WITH_POLICY_FILE
BURDEN_CSV = BURDEN_DATA_FILE
POLICY_XLSX = RAW_POLICY_DIR / "province_policy_2018.xlsx"
OUTPUT_DIR = PROMPTS_CHFS2019_DKI_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Region-to-province fallback ====================
REGION_TO_PROVINCE = {
    "东北": "辽宁省",
    "东部": "江苏省",
    "西部": "四川省",
    "中部": "湖北省",
}

# ==================== Load data ====================
print("\n[Step 1] Load data...")

sample_data = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
print(f"✓ Sample data: {len(sample_data)} records")

burden_data = pd.read_csv(BURDEN_CSV, encoding="utf-8-sig")
print(f"✓ Burden data: {len(burden_data)} records")

policy_data = pd.read_excel(POLICY_XLSX, sheet_name="总表")
print(f"✓ Policy data: {len(policy_data)} records")

# ==================== Merge affordability indicators ====================
# burden_data and sample_data share household and individual identifiers.
sample_data = sample_data.merge(
    burden_data[
        [
            "家庭ID",
            "个人ID",
            "居民保_个人收入负担",
            "居民保_家庭人均收入负担",
            "居民保_家庭人均净资产负担",
            "职工保_个人收入负担",
            "职工保_家庭人均收入负担",
            "职工保_家庭人均净资产负担",
        ]
    ],
    on=["家庭ID", "个人ID"],
    how="left",
)
print(f"✓ Columns after merge: {len(sample_data.columns)}")


# ==================== Parsing helpers ====================
def parse_yuan(val):
    try:
        return float(re.sub(r"[^\d.\-]", "", str(val)))
    except Exception:
        return np.nan


def safe_pct(numerator, denominator):
    """Return a percentage string, or N/A for an invalid denominator."""
    try:
        n = float(numerator)
        d = float(denominator)
        if np.isnan(n) or np.isnan(d) or d <= 0:
            return "N/A"
        return f"{n / d * 100:.2f}%"
    except Exception:
        return "N/A"


# ==================== Income terciles ====================
sample_data["_income_num"] = sample_data["个人年收入"].apply(parse_yuan)
valid_income = sample_data["_income_num"].dropna()
q33 = valid_income.quantile(1 / 3)
q67 = valid_income.quantile(2 / 3)


def income_group(val):
    if pd.isna(val):
        return "不清楚"
    if val <= q33:
        return "低收入"
    elif val <= q67:
        return "中收入"
    else:
        return "高收入"


sample_data["收入分组"] = sample_data["_income_num"].apply(income_group)
print(f"✓ Income terciles: low ≤ {q33:.0f} RMB, middle ≤ {q67:.0f} RMB, high > {q67:.0f} RMB")


# ==================== Precomputed indicators ====================
def calc_burden_rate(row, prefix):
    """Compute minimum contribution divided by the largest valid capacity base."""
    col_fee = (
        "居民保缴费额下限（年）" if prefix == "居民保" else "职工保缴费额下限（年）"
    )
    col_r1 = f"{prefix}_个人收入负担"
    col_r2 = f"{prefix}_家庭人均收入负担"
    col_r3 = f"{prefix}_家庭人均净资产负担"

    fee = pd.to_numeric(row.get(col_fee), errors="coerce")
    if pd.isna(fee) or fee <= 0:
        return "N/A"

    r1 = row.get(col_r1)
    r2 = row.get(col_r2)
    r3 = row.get(col_r3)

    # Recover the denominator from fee / burden ratio.
    def to_denom(ratio):
        try:
            r = float(ratio)
            if np.isnan(r) or r <= 0:
                return np.nan
            return fee / r
        except Exception:
            return np.nan

    denoms = [to_denom(r1), to_denom(r2), to_denom(r3)]
    pos_denoms = [d for d in denoms if not np.isnan(d) and d > 0]
    if not pos_denoms:
        return "N/A"
    best_denom = max(pos_denoms)
    pct = fee / best_denom * 100
    return f"{pct:.2f}%"


def calc_pension_dependency(row):
    """Compute annual household pension receipts divided by household income."""
    pension_month = parse_yuan(row.get("家庭月均养老金", ""))
    total_income = parse_yuan(row.get("家庭总收入", ""))
    if np.isnan(pension_month) or np.isnan(total_income) or total_income <= 0:
        return "N/A"
    if pension_month <= 0:
        return "0.00%"
    pct = pension_month * 12 / total_income * 100
    return f"{pct:.2f}%"


# ==================== DKI prompt template ====================
PROMPT_TEMPLATE = """# 角色
你是一位行为经济学与社会保障决策领域的专家。

# 任务背景

请你根据提供的灵活就业人员个人信息，分析在给定的政策情景下，该样本可能做出的养老保险参保决策，包括是否参保和参保种类（城乡居民基本养老保险/城镇职工基本养老保险）。

# 个人与家庭数据

## 个人基本信息

- 家庭ID: {家庭ID}
- 个人ID: {个人ID}
- 年龄: {年龄}岁, 性别: {性别}
- 文化程度: {文化程度}
- 健康状况: {健康状况}
- 户口: {户口性质}, 户口省: {户口省份}
- 常住省: {常住省份}, 是否流动: {是否流动}

## 就业与收入

- 工作性质: {工作性质}
- 工作行业: {工作行业}, 职业: {工作职业}, 单位类型: {单位类型}
- 个人年收入: {个人年收入}
- 家庭总收入: {家庭总收入}/年
- 家庭总消费: {家庭总消费}/年

## 家庭经济

- 总资产: {家庭总资产}
- 总负债: {家庭总负债}
- 家庭人数: {家庭人数}人
- 子女数(<16岁): {子女数}人
- 老人数(≥60岁): {老人数}人

{养老保险状态块}

## 预计算指标

- 居民保最优负担率: {居民保负担率}
- 职工保最优负担率: {职工保负担率}
- 家庭养老金依赖度: {家庭养老金依赖度}

## 心理特征

- 风险偏好: {风险偏好}
- 经济预期: {经济预期}

## 地区

- 城乡: {城乡}, 地区: {地区}
- 收入三分位: {收入分组}

# 政策情景（基于户籍地）

{政策情景块}

# 推理步骤

请严格按照以下步骤逐步分析：

## 步骤1：历史参保状态依赖

如果累计参保年限不等于 15 年（包括小于 15 年和大于 15 年）且未断缴或短期断缴，通常会延续缴费行为；刚好满 15 年或存在长期断缴等其他情况，需重新评估参保必要性。需注意若历史参保状态为不参保，也大概率维持现状。

## 步骤2：家庭影响

若家庭有 2 个及以上的人参保，正向示范效应会强烈激励该人员参保；但当家庭养老金依赖度极高且个人收入极低时，这种正向示范效应会被抑制。

## 步骤3：经济状况评估

使用预计算的负担率指标评价支付能力，若居民保负担率小于 1%，认为其有能力负担居民保；若职工保负担率小于 25%，认为其有能力负担职工保，初步评估其参保类型。

## 步骤4：行为与情境修正

结合健康、年龄、文化程度等因素评估回本预期，考虑人员流动状态带来的参保便利性，以及风险偏好与经济预期反映的制度信任度，微调参保决策。比如若历史为【城乡居民养老保险】且为【农业户口】或【初中及以下文化】，大概率维持现状。

## 步骤5：综合决策生成

根据历史参保状态、家庭影响、经济状况和行为经济学因素综合给出最终参保决策。

# 输出要求

**严格输出以下JSON格式，不要额外解释**：

```json
{{
  "household_id": "家庭ID",
  "individual_id": "个人ID",

  "decision_process": {{
    "step1": "历史参保状态依赖",
    "step2": "家庭影响",
    "step3": "经济状况评估",
    "step4": "行为与情境修正",
    "step5": "综合决策生成"
  }},

  "insurance_decision": {{
    "action": "不参保/参保",
    "insurance_type": "不参保/城乡居民养老保险/城镇职工养老保险",
    "annual_payment": 年缴费金额数字（不参保则为0）,
    "main_reason": "20字内核心原因"
  }}
}}
```

**关键强调**：

1. 你不是理财顾问，不要给出最优建议，而是模拟真实决策
2. 缴费能力评估应该基于提供的客观计算指标，不要凭感觉
"""

# Policy block for Hong Kong cases
HK_POLICY_BLOCK = "香港地区不支持缴纳中国大陆社会养老保险"

# Standard province-policy block
NORMAL_POLICY_BLOCK = """## 城乡居民基本养老保险
**缴费规则**：

- 缴费档次与补贴明细：{缴费档次与补贴明细}
- 缴费去向：全部缴费金额进入个人账户

**领取规则**：

- 缴费年限：满15年
- 领取年龄：60岁（男女均）
- 退休养老金 = 基础养老金 + 个人账户养老金
  - 基础养老金：{基础养老金}元/年（政府发放，定额）
  - 个人账户养老金/月 = 个人账户累计储存额 ÷ 139个月

## 城镇职工基本养老保险

**缴费规则**：

- 当地社平工资：{社平工资}元/年
- 可选缴费指数：{缴费指数下限}-{缴费指数上限}，{可选档次规则}
- 可选缴费基数(=社平工资×缴费指数)：{缴费基数下限}元/年 ～ {缴费基数上限}元/年
- 缴费比例：20%（其中8%进入个人账户，12%进入统筹）
- 年缴费金额(=缴费基数×缴费比例)：{缴费金额下限}元 ～ {缴费金额上限}元

**领取规则**：

- 缴费年限：满15年
- 领取年龄：男60岁/女55岁
- 退休养老金 = 基础养老金 + 个人账户养老金
  - 基础养老金 = 退休时社平工资 × (1 + 本人平均缴费指数) ÷ 2 × 缴费年限 × 1%
  - 个人账户养老金/月 = 个人账户储存额 ÷ 计发月数（60岁139个月，55岁170个月）"""

print("✓ Prompt template ready")


# ==================== Helper functions ====================
def build_insurance_section(row):
    hist_type = str(row.get("历史参保类型", "不清楚")).strip()
    family_part = (
        f"- 家庭参保人数: {row['家庭参保人数']}人\n"
        f"- 家庭领取养老金人数: {row['家庭领取人数']}人\n"
        f"- 家庭月均养老金: {row['家庭月均养老金']}/月"
    )
    if hist_type in ["不参保", "不清楚"]:
        return f"## 养老保险状态\n- 本人历史参保状态：{hist_type}\n" + family_part
    else:
        return (
            f"## 养老保险状态\n"
            f"- 本人历史参保类型：{hist_type}\n"
            f"- 累计缴纳年限：{row['累计缴纳年限']}\n"
            f"- 是否存在断缴：{row['是否存在断缴']}\n" + family_part
        )


def match_policy(province, policy_df):
    matched = policy_df[policy_df["省名"] == province]
    if len(matched) == 0:
        return None
    p = matched.iloc[0]

    def safe_int(col):
        v = p.get(col)
        return str(int(v)) if pd.notna(v) else "不清楚"

    return {
        "缴费档次与补贴明细": p.get("缴费档次与补贴明细（年，分档次）", "不清楚"),
        "基础养老金": safe_int("基础养老金（年）"),
        "社平工资": safe_int("社平工资（年）"),
        "缴费指数下限": str(p.get("缴费指数（下限）", "不清楚")),
        "缴费指数上限": str(p.get("缴费指数（上限）", "不清楚")),
        "可选档次规则": p.get("可选档次规则", "不清楚"),
        "缴费基数下限": safe_int("缴费基数（年，下限）"),
        "缴费基数上限": safe_int("缴费基数（年，上限）"),
        "缴费金额下限": safe_int("缴费金额（年，下限）"),
        "缴费金额上限": safe_int("缴费金额（年，上限）"),
    }


# ==================== Generate prompts ====================
print("\n[Step 2] Generate prompts...")

prompts = []
ground_truth = []
skip_count = 0

for idx, row in sample_data.iterrows():
    hhid = str(row["家庭ID"])
    pid = str(row["个人ID"])
    sample_id = f"{hhid}-{pid}"

    # --- Match province and policy ---
    province_raw = str(row["户口省份"]).strip() if pd.notna(row["户口省份"]) else ""
    is_hk = "香港" in province_raw
    is_empty = province_raw == "" or province_raw.lower() in ["nan", "不清楚"]

    if is_hk:
        province_display = province_raw
        policy_block = HK_POLICY_BLOCK

    elif is_empty:
        province_display = "不清楚"
        region = str(row.get("地区", "")).strip()
        proxy_province = REGION_TO_PROVINCE.get(region)
        if proxy_province is None:
            skip_count += 1
            continue
        policy_vars = match_policy(proxy_province, policy_data)
        if policy_vars is None:
            skip_count += 1
            continue
        policy_block = NORMAL_POLICY_BLOCK.format(**policy_vars)

    else:
        province_display = province_raw
        policy_vars = match_policy(province_raw, policy_data)
        if policy_vars is None:
            for suffix in [
                "省",
                "市",
                "自治区",
                "壮族自治区",
                "回族自治区",
                "维吾尔自治区",
                "藏族自治区",
            ]:
                short = province_raw.replace(suffix, "")
                policy_vars = match_policy(short, policy_data)
                if policy_vars:
                    break
        if policy_vars is None:
            skip_count += 1
            continue
        policy_block = NORMAL_POLICY_BLOCK.format(**policy_vars)

    # --- Compute DKI indicators ---
    jumin_rate = calc_burden_rate(row, "居民保")
    zhigong_rate = calc_burden_rate(row, "职工保")
    pension_dep = calc_pension_dependency(row)

    # --- Build the pension-history section ---
    insurance_section = build_insurance_section(row)

    sample_vars = {
        "家庭ID": hhid,
        "个人ID": pid,
        "年龄": str(row["年龄"]),
        "性别": row["性别"],
        "文化程度": row["文化程度"],
        "健康状况": row["健康状况"],
        "户口性质": row["户口性质"],
        "户口省份": province_display,
        "常住省份": row["常住省份"],
        "是否流动": row["是否流动"],
        "工作性质": row["工作性质"],
        "工作行业": row["工作行业"],
        "工作职业": row["工作职业"],
        "单位类型": row["单位类型"],
        "个人年收入": row["个人年收入"],
        "家庭总收入": row["家庭总收入"],
        "家庭总消费": row["家庭总消费"],
        "家庭总资产": row["家庭总资产"],
        "家庭总负债": row["家庭总负债"],
        "家庭人数": str(row["家庭人数"]),
        "子女数": str(row["子女数"]),
        "老人数": str(row["老人数"]),
        "养老保险状态块": insurance_section,
        "居民保负担率": jumin_rate,
        "职工保负担率": zhigong_rate,
        "家庭养老金依赖度": pension_dep,
        "风险偏好": row["风险偏好"],
        "经济预期": row["经济预期"],
        "城乡": row["城乡"],
        "地区": row["地区"],
        "收入分组": row["收入分组"],
        "政策情景块": policy_block,
    }

    try:
        prompt_text = PROMPT_TEMPLATE.format(**sample_vars)
    except KeyError as e:
        print(f"  ✗ Sample {sample_id}: template substitution failed: {e}; skipping")
        skip_count += 1
        continue

    prompts.append(
        {
            "id": sample_id,
            "household_id": hhid,
            "individual_id": pid,
            "province": province_display,
            "prompt": prompt_text,
        }
    )

    decision = row["2018年参保决策"]
    gt_type = "不参保" if decision == "不参保" else row["2019年参保账户"]
    ground_truth.append(
        {
            "id": sample_id,
            "household_id": hhid,
            "individual_id": pid,
            "decision": decision,
            "type": gt_type,
        }
    )

    if (idx + 1) % 1000 == 0:
        print(f"  ... Processed {idx + 1}/{len(sample_data)} records")

print(f"\n✓ Generated: {len(prompts)} | Skipped: {skip_count}")

# ==================== Save outputs ====================
print("\n[Step 3] Save results...")

prompts_path = OUTPUT_DIR / "all_prompts_newera.json"
gt_path = OUTPUT_DIR / "ground_truth_newera.json"

with open(prompts_path, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)
print(f"✓ Saved: {prompts_path}")

with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)
print(f"✓ Saved: {gt_path}")

# ==================== Print one example ====================
print("\n" + "=" * 80)
if prompts:
    print("Prompt example (first record, first 1,200 characters)")
    print("=" * 80)
    print(prompts[0]["prompt"][:1200])
    print("...")
print("=" * 80)
print(f"New-era prompt generation complete: {len(prompts)} records")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)
