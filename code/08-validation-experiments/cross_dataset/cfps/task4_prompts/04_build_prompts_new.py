#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_build_prompts_new.py
Build prompts and ground truth for the 500 CFPS 2018 cases reconstructed by
task 3.

New version:
  - Non-participants: hist_type = "不参保"
  - Participants: hist_type = "不清楚" because CFPS has no enrollment history

Outputs:
  data/prompts/generalization/cfps2018/prompts_cfps_new.json
  data/prompts/generalization/cfps2018/ground_truth_cfps_new.json
"""

import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR  = Path(__file__).resolve().parent
TASK3_CSV   = SCRIPT_DIR.parent / "task3_reconstructing" / "reconstructed_500.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
OUTPUT_DIR = PROJECT_ROOT / "data" / "prompts" / "generalization" / "cfps2018"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PROMPTS = OUTPUT_DIR / "prompts_cfps_new.json"
OUT_GT      = OUTPUT_DIR / "ground_truth_cfps_new.json"

print("=" * 70)
print("Task 4 New  Build prompts and ground truth (CFPS 2018)")
print("=" * 70)

# 1. Load reconstructed cases
print("\n[1] Loading data")
df = pd.read_csv(TASK3_CSV, dtype={"家庭ID": str, "个人ID": str})
print(f"  reconstructed_500: {len(df)} records")

# 2. Build the enrollment-history block
def build_insurance_section(row, hist_type: str) -> str:
    family_part = (
        f"- 家庭参保人数: {row['家庭参保人数']}人\n"
        f"- 家庭领取养老金人数: {row['家庭领取人数']}人\n"
        f"- 家庭月均养老金: {row['家庭月均养老金']}/月"
    )
    if hist_type in ("不参保", "不清楚"):
        return f"## 养老保险状态\n- 本人历史参保状态：{hist_type}\n" + family_part
    else:
        return (
            f"## 养老保险状态\n"
            f"- 本人历史参保类型：{hist_type}\n"
            f"- 累计缴纳年限：{row['累计缴纳年限']}\n"
            f"- 是否存在断缴：{row['是否存在断缴']}\n"
            + family_part
        )

# 3. Chinese prompt template aligned with CHFS 2017
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

print("Prompt template ready")

# 4. Generate one prompt per case
print("\n[2] Generating prompts (non-participants: hist_type='不参保'; participants: hist_type='不清楚')")

POLICY_COLS = [
    "缴费档次与补贴明细", "基础养老金", "社平工资",
    "缴费指数下限", "缴费指数上限", "可选档次规则",
    "缴费基数下限", "缴费基数上限", "缴费金额下限", "缴费金额上限",
]

prompts      = []
ground_truth = []
skip_count   = 0
hist_counts  = {"不参保": 0, "不清楚": 0}

def na_if_unclear(val) -> str:
    s = str(val).strip()
    return "N/A" if s in ("不清楚", "nan", "") else s

for _, row in df.iterrows():
    hhid = str(row["家庭ID"])
    pid  = str(row["个人ID"])
    sid  = f"{hhid}-{pid}"

    # Use "不参保" for historical non-participants and "不清楚" for participants.
    decision = str(row["参保决策"])
    hist_type = "不参保" if decision == "不参保" else "不清楚"
    hist_counts[hist_type] = hist_counts.get(hist_type, 0) + 1

    # Policy fields were matched during task 3.
    policy_vars = {col: str(row[col]) for col in POLICY_COLS}
    try:
        policy_block = NORMAL_POLICY_BLOCK.format(**policy_vars)
    except KeyError as e:
        print(f"  Skipped {sid}: failed to format policy block ({e})")
        skip_count += 1
        continue

    # Enrollment-history block.
    insurance_section = build_insurance_section(row, hist_type)

    # Normalize the income-group label.
    income_grp = str(row["收入分组"]).replace("组", "")

    # Fill the prompt template.
    sample_vars = {
        "家庭ID":       hhid,
        "个人ID":       pid,
        "年龄":         str(row["年龄"]),
        "性别":         row["性别"],
        "文化程度":     row["文化程度"],
        "健康状况":     row["健康状况"],
        "户口性质":     row["户口性质"],
        "户口省份":     row["户口省份"],
        "常住省份":     row["常住省份"],
        "是否流动":     row["是否流动"],
        "工作性质":     row["工作性质"],
        "工作行业":     row["工作行业"],
        "工作职业":     row["工作职业"],
        "单位类型":     row["单位类型"],
        "个人年收入":   row["个人年收入"],
        "家庭总收入":   row["家庭总收入"],
        "家庭总消费":   row["家庭总消费"],
        "家庭总资产":   row["家庭总资产"],
        "家庭总负债":   row["家庭总负债"],
        "家庭人数":     str(row["家庭人数"]),
        "子女数":       str(row["子女数"]),
        "老人数":       str(row["老人数"]),
        "养老保险状态块":   insurance_section,
        "居民保负担率":     na_if_unclear(row["居民保负担率"]),
        "职工保负担率":     na_if_unclear(row["职工保负担率"]),
        "家庭养老金依赖度": na_if_unclear(row["家庭养老金依赖度"]),
        "风险偏好":     row["风险偏好"],
        "经济预期":     row["经济预期"],
        "城乡":         row["城乡"],
        "地区":         row["地区"],
        "收入分组":     income_grp,
        "政策情景块":   policy_block,
    }

    try:
        prompt_text = PROMPT_TEMPLATE.format(**sample_vars)
    except KeyError as e:
        print(f"  Skipped {sid}: failed to fill prompt template ({e})")
        skip_count += 1
        continue

    prompts.append({
        "id":            sid,
        "household_id":  hhid,
        "individual_id": pid,
        "province":      str(row["户口省份"]),
        "prompt":        prompt_text,
    })

    # Ground truth
    gt_type = "不参保" if decision == "不参保" else str(row["参保账户"])
    ground_truth.append({
        "id":            sid,
        "household_id":  hhid,
        "individual_id": pid,
        "decision":      decision,
        "type":          gt_type,
    })

print(f"  Generated: {len(prompts)} | skipped: {skip_count}")
print(f"  hist_type distribution: 不参保={hist_counts.get('不参保',0)}, 不清楚={hist_counts.get('不清楚',0)}")

# 5. Save prompts and labels
print("\n[3] Saving results")

with open(OUT_PROMPTS, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)
print(f"  Saved: {OUT_PROMPTS}")

with open(OUT_GT, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)
print(f"  Saved: {OUT_GT}")

# 6. Print summary statistics
print("\n[4] Ground-truth distribution")
gt_df = pd.DataFrame(ground_truth)
print(gt_df["decision"].value_counts().to_string())
print()
print(gt_df["type"].value_counts().to_string())

print("\n[5] Prompt preview (first 600 characters of the first prompt)")
print("=" * 70)
if prompts:
    print(prompts[0]["prompt"][:600])
    print("...")
print("=" * 70)
print(f"Done. Generated {len(prompts)} prompts.")
