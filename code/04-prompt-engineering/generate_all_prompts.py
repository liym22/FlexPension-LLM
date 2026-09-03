#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate baseline prompts for every CHFS 2019 sample.

Hong Kong cases receive a dedicated policy statement. When hukou province is
missing, the prompt keeps the value unknown and uses a representative province
from the observed region for policy matching.
"""

import pandas as pd
import json
import os
import re
from pathlib import Path

print("=" * 80)
print("Generate Prompts for All Samples")
print("=" * 80)

# ==================== Repository-relative paths ====================
# Resolve every input and output path from the script location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = PROJECT_ROOT / "data" / "processed" / "all_samples_with_policy.csv"
POLICY_XLSX = PROJECT_ROOT / "data" / "raw" / "policy" / "province_policy_2018.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prompts" / "chfs2019" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Region-to-province fallback used when hukou province is unavailable.
REGION_TO_PROVINCE = {
    "东北": "辽宁省",
    "东部": "江苏省",
    "西部": "四川省",
    "中部": "湖北省",
}

# ==================== Load data ====================
print("\n[Step 1] Load data...")
try:
    sample_data = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
    print(f"✓ Sample data: {len(sample_data)} records ({SAMPLE_CSV})")

    policy_data = pd.read_excel(POLICY_XLSX, sheet_name="总表")
    print(f"✓ Policy data: {len(policy_data)} provinces ({POLICY_XLSX})")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    exit(1)


# ==================== Compute income terciles ====================
def parse_income(val):
    """Parse an RMB-valued string and return NaN on failure."""
    try:
        return float(re.sub(r"[^\d.\-]", "", str(val)))
    except Exception:
        return float("nan")


sample_data["_income_num"] = sample_data["个人年收入"].apply(parse_income)
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

# ==================== Baseline prompt template ====================
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

## 步骤1：历史惯性评估
根据累计参保年限与是否存在断缴行为判断行为延续性。参保不满15年且未断缴者即使有流动性约束也通常延续缴费；长期未参保者、断缴者和缴费满15年者需评估参保必要性。

## 步骤2：流动性约束识别
结合家庭收支结余、资产负债和工作稳定性评估即时支付能力。收入波动大或债务负担重者可能因现金紧张而放弃参保，即使年收入看似充足。

## 步骤3：行为与情境修正
结合健康与年龄等因素评估回本预期，参考家庭养老金领取情况形成社会参照，考虑人员流动状态带来的参保便利性，以及风险偏好与经济预期反映的制度信任度，综合形成主观收益判断。

## 步骤4：综合决策生成
根据历史参保状态、流动性约束和行为经济学因素综合给出参保决策。

# 输出要求

**严格输出以下JSON格式，不要额外解释**：

```json
{{
  "household_id": "{家庭ID}",
  "individual_id": "{个人ID}",
  
  "decision_process": {{
    "step1": "历史参保状态的影响",
    "step2": "基于收支、资产与工作性质的缴费压力评估",
    "step3": "回本预期、家庭养老金参照、流动摩擦及心理特征的综合影响",
    "step4": "综合前三步的最终决策依据"
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
2. 缴费能力评估应该基于实际数字计算，不要凭感觉
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
  - 个人账户养老金/月 = 个人账户储存额 ÷ 计发月数（60岁139个月，55岁170个月）

## 关键政策对比
- 职工保具有抗通胀属性，养老金随着社会平均工资上涨而上涨，适用于收入稳定、能承受高缴费、看重长期保障和抗通胀的人群。
- 居民保属于定额福利，增长较慢，主要满足基本生存需求，适用于收入不稳定、负担重、更看重短期流动性的人群。"""

print("✓ Prompt template ready")


# ==================== Helper functions ====================
def build_insurance_section(row):
    """Build the pension-history section for one sample."""
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
    """Return policy fields for a province, or None when no row matches."""
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

    # ---------- Match hukou province to policy ----------
    province_raw = str(row["户口省份"]).strip() if pd.notna(row["户口省份"]) else ""

    is_hk = "香港" in province_raw
    is_empty = province_raw == "" or province_raw.lower() in ["nan", "不清楚"]

    if is_hk:
        province_display = province_raw
        policy_block = HK_POLICY_BLOCK
        policy_vars = {}  # No policy row is required for this case.

    elif is_empty:
        province_display = "不清楚"
        region = str(row.get("地区", "")).strip()
        proxy_province = REGION_TO_PROVINCE.get(region)
        if proxy_province is None:
            print(f"  ⚠ Sample {sample_id}: no representative province for region [{region}]; skipping")
            skip_count += 1
            continue
        policy_vars = match_policy(proxy_province, policy_data)
        if policy_vars is None:
            print(f"  ⚠ Sample {sample_id}: no policy data for representative province [{proxy_province}]; skipping")
            skip_count += 1
            continue
        policy_block = NORMAL_POLICY_BLOCK.format(**policy_vars)

    else:
        province_display = province_raw
        policy_vars = match_policy(province_raw, policy_data)
        if policy_vars is None:
            # Retry after removing province-level administrative suffixes.
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
            print(f"  ⚠ Sample {sample_id}: no policy data for province [{province_raw}]; skipping")
            skip_count += 1
            continue
        policy_block = NORMAL_POLICY_BLOCK.format(**policy_vars)

    # ---------- Prepare sample variables ----------
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

    if (idx + 1) % 500 == 0:
        print(f"  ... Processed {idx + 1}/{len(sample_data)} records")

print(f"\n✓ Generated: {len(prompts)} | Skipped: {skip_count}")

# ==================== Save results ====================
print("\n[Step 3] Save results...")

prompts_path = OUTPUT_DIR / "all_prompts.json"
gt_path = OUTPUT_DIR / "ground_truth.json"

with open(prompts_path, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)
print(f"✓ Saved: {prompts_path}")

with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)
print(f"✓ Saved: {gt_path}")

# ==================== Print one example ====================
print("\n" + "=" * 80)
if prompts:
    print("Prompt example (first record, first 1,000 characters)")
    print("=" * 80)
    print(prompts[0]["prompt"][:1000])
    print("...")

print("=" * 80)
print(f"Prompt generation complete: {len(prompts)} records")
print(f"Output directory: {OUTPUT_DIR}")
print(f"  - {prompts_path}")
print(f"  - {gt_path}")
print("=" * 80)
