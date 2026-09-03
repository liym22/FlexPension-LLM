"""
05_build_prompts.py
Build prompts and ground truth for the 500 CHIP cases reconstructed by task 3.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR  = Path(__file__).resolve().parent
TASK3_CSV   = SCRIPT_DIR.parent / "task3_reconstructing" / "reconstructed_500.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
POLICY_XLSX = PROJECT_ROOT / "data" / "raw" / "policy" / "province_policy_2018.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prompts" / "generalization" / "chip2018"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PROMPTS = OUTPUT_DIR / "prompts_chip_new.json"
OUT_GT      = OUTPUT_DIR / "ground_truth_chip_new.json"

print("=" * 70)
print("Task 4  Build prompts and ground truth (CHIP)")
print("=" * 70)

# Region-to-province fallback when hukou province is unknown.
REGION_TO_PROVINCE = {
    "东北": "辽宁省",
    "东部": "江苏省",
    "西部": "四川省",
    "中部": "湖北省",
}

# ════════════════════════════════════════════════════════════════════════════
# 1. Load reconstructed cases and policy data.
# ════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading data")
df = pd.read_csv(TASK3_CSV, dtype={"家庭ID": str, "个人ID": str})
print(f"  reconstructed_500: {len(df)} records")

# Load cached policy formula values with data_only.
wb = openpyxl.load_workbook(POLICY_XLSX, data_only=True, read_only=True)
ws = wb["总表"]
all_rows   = list(ws.iter_rows(values_only=True))
header     = [str(c) if c is not None else f"_col{i}" for i, c in enumerate(all_rows[0])]
policy_df  = pd.DataFrame(all_rows[1:], columns=header)
policy_df  = policy_df[policy_df["省名"].notna()].reset_index(drop=True)
print(f"  Policy table: {len(policy_df)} provinces")

# ════════════════════════════════════════════════════════════════════════════
# 2. Policy matching.
# ════════════════════════════════════════════════════════════════════════════
def match_policy(province: str):
    """Return the exact province's policy values, or None if unmatched."""
    matched = policy_df[policy_df["省名"] == province]
    if len(matched) == 0:
        return None
    p = matched.iloc[0]

    def safe_int(col):
        v = p.get(col)
        try:
            return str(int(float(v))) if pd.notna(v) else "不清楚"
        except (ValueError, TypeError):
            return "不清楚"

    def fmt_index(col):
        """Format a contribution index without trailing zeros."""
        v = p.get(col)
        try:
            return f"{float(v):g}" if pd.notna(v) else "不清楚"
        except (ValueError, TypeError):
            return "不清楚"

    return {
        "缴费档次与补贴明细": p.get("缴费档次与补贴明细（年，分档次）", "不清楚"),
        "基础养老金":       safe_int("基础养老金（年）"),
        "社平工资":         safe_int("社平工资（年）"),
        "缴费指数下限":     fmt_index("缴费指数（下限）"),
        "缴费指数上限":     fmt_index("缴费指数（上限）"),
        "可选档次规则":     p.get("可选档次规则", "不清楚"),
        "缴费基数下限":     safe_int("缴费基数（年，下限）"),
        "缴费基数上限":     safe_int("缴费基数（年，上限）"),
        "缴费金额下限":     safe_int("缴费金额（年，下限）"),
        "缴费金额上限":     safe_int("缴费金额（年，上限）"),
    }

# ════════════════════════════════════════════════════════════════════════════
# 3. Build the enrollment-history block.
# ════════════════════════════════════════════════════════════════════════════
def build_insurance_section(row) -> str:
    # Use "不参保" for historical non-participants and "不清楚" for participants.
    hist_type = str(row.get("历史参保类型", "不清楚")).strip()

    # Remove the existing monthly suffix before template formatting.
    pension_raw = str(row.get("家庭月均养老金", "不清楚"))
    pension_val = pension_raw.removesuffix("/月")   # Python 3.9+

    family_part = (
        f"- 家庭参保人数: {row['家庭参保人数']}人\n"
        f"- 家庭领取养老金人数: {row['家庭领取人数']}人\n"
        f"- 家庭月均养老金: {pension_val}/月"
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

# ════════════════════════════════════════════════════════════════════════════
# 4. Chinese prompt template aligned with generate_prompts_newera.py.
# ════════════════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════════════════
# 5. Generate one prompt per case.
# ════════════════════════════════════════════════════════════════════════════
print("\n[2] Generating prompts row by row")

prompts      = []
ground_truth = []
skip_count   = 0
proxy_count  = 0   # Number of cases using a fallback province.

def na_if_unclear(val: str) -> str:
    """Map '不清楚' to 'N/A' for precomputed metric fields."""
    return "N/A" if str(val).strip() == "不清楚" else str(val)

for _, row in df.iterrows():
    hhid = str(row["家庭ID"])
    pid  = str(row["个人ID"])
    sid  = f"{hhid}-{pid}"

    # Resolve province and policy block.
    province_raw = str(row["户口省份"]).strip()
    is_unclear   = province_raw in ("", "不清楚", "nan")

    if is_unclear:
        province_display = "不清楚"
        region = str(row.get("地区", "")).strip()
        proxy  = REGION_TO_PROVINCE.get(region)
        if proxy is None:
            print(f"  Skipped {sid}: hukou province is '不清楚' and region {region} has no proxy province")
            skip_count += 1
            continue
        policy_vars = match_policy(proxy)
        if policy_vars is None:
            print(f"  Skipped {sid}: proxy province {proxy} has no policy-table match")
            skip_count += 1
            continue
        proxy_count += 1
        policy_block = NORMAL_POLICY_BLOCK.format(**policy_vars)
    else:
        province_display = province_raw
        policy_vars = match_policy(province_raw)
        if policy_vars is None:
            print(f"  Skipped {sid}: province {province_raw} has no policy-table match")
            skip_count += 1
            continue
        policy_block = NORMAL_POLICY_BLOCK.format(**policy_vars)

    # Normalize the income-group label.
    income_grp = str(row["收入分组"]).replace("组", "")

    # Enrollment-history block.
    insurance_section = build_insurance_section(row)

    # Fill template variables.
    sample_vars = {
        "家庭ID":     hhid,
        "个人ID":     pid,
        "年龄":       str(row["年龄"]),
        "性别":       row["性别"],
        "文化程度":   row["文化程度"],
        "健康状况":   row["健康状况"],
        "户口性质":   row["户口性质"],
        "户口省份":   province_display,
        "常住省份":   row["常住省份"],
        "是否流动":   row["是否流动"],
        "工作性质":   row["工作性质"],
        "工作行业":   row["工作行业"],
        "工作职业":   row["工作职业"],
        "单位类型":   row["单位类型"],
        "个人年收入": row["个人年收入"],
        "家庭总收入": row["家庭总收入"],
        "家庭总消费": row["家庭总消费"],
        "家庭总资产": row["家庭总资产"],
        "家庭总负债": row["家庭总负债"],
        "家庭人数":   str(row["家庭人数"]),
        "子女数":     str(row["子女数"]),
        "老人数":     str(row["老人数"]),
        "养老保险状态块":  insurance_section,
        "居民保负担率":    na_if_unclear(row["居民保负担率"]),
        "职工保负担率":    na_if_unclear(row["职工保负担率"]),
        "家庭养老金依赖度": na_if_unclear(row["家庭养老金依赖度"]),
        "风险偏好":   row["风险偏好"],
        "经济预期":   row["经济预期"],
        "城乡":       row["城乡"],
        "地区":       row["地区"],
        "收入分组":   income_grp,
        "政策情景块": policy_block,
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
        "province":      province_display,
        "prompt":        prompt_text,
    })

    ground_truth.append({
        "id":            sid,
        "household_id":  hhid,
        "individual_id": pid,
        "decision":      str(row["参保决策"]),
        "type":          str(row["参保账户"]),
    })

print(f"  Generated: {len(prompts)} | skipped: {skip_count} | proxy province used: {proxy_count}")

# ════════════════════════════════════════════════════════════════════════════
# 6. Save prompts and labels.
# ════════════════════════════════════════════════════════════════════════════
print("\n[3] Saving results")

with open(OUT_PROMPTS, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)
print(f"  Saved: {OUT_PROMPTS}")

with open(OUT_GT, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)
print(f"  Saved: {OUT_GT}")

# ════════════════════════════════════════════════════════════════════════════
# 7. Print summary statistics.
# ════════════════════════════════════════════════════════════════════════════
print("\n[4] Ground-truth distribution")
gt_df = pd.DataFrame(ground_truth)
print(gt_df["decision"].value_counts().to_string())
print()
print(gt_df["type"].value_counts().to_string())

print("\n[5] Prompt preview (first 800 characters of the first prompt)")
print("=" * 70)
if prompts:
    print(prompts[0]["prompt"][:800])
    print("...")
print("=" * 70)
print(f"Done. Generated {len(prompts)} prompts.")
