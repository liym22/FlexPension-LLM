#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: 变量转换
将 all_samples_with_policy.csv 中的可读变量转换为回归所需的标准格式
输出：regression_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("Step 2: Transform variables")
print("=" * 80)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
src_file = PROCESSED_DIR / 'all_samples_with_policy.csv'

df = pd.read_csv(src_file, dtype=str)  # Preserve survey codes while loading.
print(f"Loaded data: {len(df)} records, {len(df.columns)} columns")

reg = pd.DataFrame()

# Identifiers.
reg['hhid']  = df['家庭ID']
reg['pline'] = df['个人ID']

# ══════════════════════════════════════════════════════════════════════════════
# Predictors.
# ══════════════════════════════════════════════════════════════════════════════

# 1. Age.
reg['age'] = pd.to_numeric(df['年龄'], errors='coerce').astype(float)

# 2. Gender: male = 1, female = 0.
reg['gender'] = df['性别'].map({'男': 1, '女': 0}).astype(float)

# 3. Education as an ordinal variable.
edu_map = {'未上学': 0, '小学': 1, '初中': 2, '高中': 3, '中专/职高': 4,
           '大专': 5, '本科': 6, '硕士': 7, '博士': 8, '不清楚': 2}
reg['education'] = df['文化程度'].map(edu_map).astype(float)

# 4. Health status as an ordinal variable.
health_map = {'非常好': 4, '好': 3, '一般': 2, '不好': 1, '非常不好': 0, '不清楚': 3}
reg['health'] = df['健康状况'].map(health_map).astype(float)

# 5. Hukou-type indicators, with non-agricultural as reference.
reg['hukou_agri'] = (df['户口性质'] == '农业').astype(int)
reg['hukou_uni']  = (df['户口性质'] == '统一居民').astype(int)
reg['hukou_else'] = (~df['户口性质'].isin(['农业', '非农业', '统一居民'])).astype(int)

# 6. Province indicators, with Guangdong Province as reference.
PROV_DUMMIES = {
    '北京市':         'hukou_beijing',
    '天津市':         'hukou_tianjin',
    '河北省':         'hukou_hebei',
    '山西省':         'hukou_shanxi',
    '内蒙古自治区':   'hukou_neimenggu',
    '辽宁省':         'hukou_liaoning',
    '吉林省':         'hukou_jilin',
    '黑龙江省':       'hukou_heilongjiang',
    '上海市':         'hukou_shanghai',
    '江苏省':         'hukou_jiangsu',
    '浙江省':         'hukou_zhejiang',
    '安徽省':         'hukou_anhui',
    '福建省':         'hukou_fujian',
    '江西省':         'hukou_jiangxi',
    '山东省':         'hukou_shandong',
    '河南省':         'hukou_henan',
    '湖北省':         'hukou_hubei',
    '湖南省':         'hukou_hunan',
    # Do not create an indicator for the Guangdong reference group.
    '广西壮族自治区': 'hukou_guangxi',
    '海南省':         'hukou_hainan',
    '重庆市':         'hukou_chongqing',
    '四川省':         'hukou_sichuan',
    '贵州省':         'hukou_guizhou',
    '云南省':         'hukou_yunnan',
    '西藏自治区':     'hukou_xizang',
    '陕西省':         'hukou_shaanxi',
    '甘肃省':         'hukou_gansu',
    '青海省':         'hukou_qinghai',
    '宁夏回族自治区': 'hukou_ningxia',
    '新疆维吾尔自治区': 'hukou_xinjiang',
# Create a separate Hong Kong indicator; empty values remain in hukou_else.
    '香港':           'hukou_hongkong',
    '中国香港':       'hukou_hongkong',
}

# Initialize all province indicators to zero.
prov_col_names = sorted(set(PROV_DUMMIES.values()))
for col in prov_col_names:
    reg[col] = 0

for prov_name, col in PROV_DUMMIES.items():
    reg.loc[df['户口省份'] == prov_name, col] = 1

# 7. Migration status: yes = 1, no = 0.
reg['floating'] = df['是否流动'].map({'是': 1, '否': 0}).fillna(0).astype(int)

# 8. Employment-type indicators, with helpers/farmers as reference.
# Source categories cover temporary, employer, self-employed, helper, freelance, and farm work.
# Farm work is grouped into the helper reference category.
work_type = df['工作性质']
reg['job_linshigong']  = (work_type == '临时工').astype(int)
reg['job_guzhu']       = (work_type == '雇主').astype(int)
reg['job_ziying']      = (work_type == '自营').astype(int)
reg['job_ziyouzhiye']  = (work_type == '自由职业').astype(int)

# 9. Industry indicators for temporary workers, with agriculture as reference.
INDUSTRY_MAP = {
    '采矿制造':   'industry_mining',
    '建筑':       'industry_constr',
    '电力燃气水': 'industry_utility',
    '批发零售':   'industry_retail',
    '交通运输':   'industry_transport',
    '住宿餐饮':   'industry_hotel',
    '信息技术':   'industry_it',
    '金融':       'industry_finance',
    '房地产':     'industry_realestate',
    '科教文卫':   'industry_education',
    '居民服务':   'industry_service',
    '公共管理':   'industry_gov',
    '其他':       'industry_other',
    '不清楚':     'industry_unknown',
}
is_employee = (work_type == '临时工')
for col in INDUSTRY_MAP.values():
    reg[col] = 0
for chin, col in INDUSTRY_MAP.items():
    reg.loc[is_employee & (df['工作行业'] == chin), col] = 1

# 10. Occupation indicators for temporary workers, with managers as reference.
OCCU_MAP = {
    '专业技术': 'occu_professional',
    '办事人员': 'occu_clerk',
    '快递员':   'occu_delivery',
    '服务人员': 'occu_service',
    '生产制造': 'occu_production',
    '其他':     'occu_other',
    '不清楚':   'occu_unknown',
}
for col in OCCU_MAP.values():
    reg[col] = 0
for chin, col in OCCU_MAP.items():
    reg.loc[is_employee & (df['工作职业'] == chin), col] = 1

# 11. Employer-type indicators for temporary workers, with public institutions as reference.
EMPLOYER_MAP = {
    '国企':       'employer_soe',
    '个体户':     'employer_individual',
    '私企':       'employer_private',
    '外资/港澳台': 'employer_foreign',
    '其他':       'employer_other',
    '不清楚':     'employer_unknown',
}
for col in EMPLOYER_MAP.values():
    reg[col] = 0
for chin, col in EMPLOYER_MAP.items():
    reg.loc[is_employee & (df['单位类型'] == chin), col] = 1

# 12-16. Log-transform income, consumption, assets, and debt as ln(x + 1).
def parse_money(series):
    """Parse values such as '12345元'; map '不清楚' to NaN."""
    nums = series.str.replace('元', '', regex=False).str.strip()
    nums = pd.to_numeric(nums, errors='coerce')
    return nums

ind_income_raw  = parse_money(df['个人年收入'])
hh_income_raw   = parse_money(df['家庭总收入'])
hh_consump_raw  = parse_money(df['家庭总消费'])
hh_asset_raw    = parse_money(df['家庭总资产'])
hh_liabi_raw    = parse_money(df['家庭总负债'])

# Impute unknown values with the median.
for raw_series in [ind_income_raw, hh_income_raw, hh_consump_raw,
                   hh_asset_raw, hh_liabi_raw]:
    median_val = raw_series.median()
    raw_series.fillna(median_val, inplace=True)

# Clip at zero before log1p.
reg['ln_ind_income']  = np.log1p(ind_income_raw.clip(lower=0))
reg['ln_hh_income']   = np.log1p(hh_income_raw.clip(lower=0))
reg['ln_hh_consump']  = np.log1p(hh_consump_raw.clip(lower=0))
reg['ln_hh_asset']    = np.log1p(hh_asset_raw.clip(lower=0))
reg['ln_hh_liabi']    = np.log1p(hh_liabi_raw.clip(lower=0))

# 17-19. Household composition.
reg['hh_num']   = pd.to_numeric(df['家庭人数'],  errors='coerce').astype(float)
reg['hh_child'] = pd.to_numeric(df['子女数'],    errors='coerce').astype(float)
reg['hh_old']   = pd.to_numeric(df['老人数'],    errors='coerce').astype(float)

# 20. Cumulative contribution years.
# Source values use forms such as "5年", "5.5年", or "不清楚".
contrib_raw = df['累计缴纳年限'].str.replace('年', '', regex=False).str.strip()
contrib_raw = pd.to_numeric(contrib_raw, errors='coerce')

# Impute unknown values with the same-age nonzero median, then the overall nonzero median.
age_series = pd.to_numeric(df['年龄'], errors='coerce')
global_median = contrib_raw[contrib_raw > 0].median()

def fill_by_age(row_idx):
    if pd.notna(contrib_raw.iloc[row_idx]):
        return contrib_raw.iloc[row_idx]
    age = age_series.iloc[row_idx]
    if pd.isna(age):
        return global_median
    # Exclude zeros from same-age medians because zero denotes no contributions.
    same_age_mask = (age_series == age) & (contrib_raw > 0)
    if same_age_mask.sum() > 0:
        return contrib_raw[same_age_mask].median()
    return global_median

contrib_filled = pd.Series(
    [fill_by_age(i) for i in range(len(contrib_raw))],
    index=contrib_raw.index
)
reg['contribution_years'] = contrib_filled

# 21. Continuous contribution history: interruption = 0, no interruption = 1.
# Source values are "断缴N年", "不存在", or "不清楚".
def map_contribution_history(val):
    if pd.isna(val): return 0
    if '断缴' in str(val): return 0   # Contributions were interrupted.
    if '不存在' in str(val): return 1  # Contributions were continuous.
    return 0  # Unknown.

reg['has_contribution_history'] = df['是否存在断缴'].apply(map_contribution_history).astype(int)

# 22-23. Household counts for enrollment and pension receipt.
reg['hh_pay_num']     = pd.to_numeric(df['家庭参保人数'],  errors='coerce').astype(float)
reg['hh_receive_num'] = pd.to_numeric(df['家庭领取人数'],  errors='coerce').astype(float)

# 24. Average monthly household pension.
pension_raw = df['家庭月均养老金'].str.replace('元', '', regex=False).str.strip()
pension_raw = pension_raw.replace('未领取', '0')
reg['hh_pension'] = pd.to_numeric(pension_raw, errors='coerce').fillna(0).astype(int)

# 25. Risk preference.
risk_map = {'高风险偏好': 4, '略高风险偏好': 3, '平均风险偏好': 2,
            '略低风险偏好': 1, '极度厌恶风险': 0, '不知道': 2}
reg['risk_preference'] = df['风险偏好'].map(risk_map).astype(float)

# 26. Economic expectations.
expect_map = {'非常好': 4, '比较好': 3, '基本不变': 2, '比较差': 1, '非常差': 0, '不清楚': 2}
reg['econ_expectation'] = df['经济预期'].map(expect_map).astype(float)

# 27. Rural residence: urban = 0, rural = 1.
reg['rural'] = df['城乡'].map({'城镇': 0, '农村': 1}).astype(float)

# 28. Region indicators, with eastern China as reference.
reg['region_central']   = (df['地区'] == '中部').astype(int)
reg['region_west']      = (df['地区'] == '西部').astype(int)
reg['region_northeast'] = (df['地区'] == '东北').astype(int)

# 29. Log annual average wage.
avg_wage_raw = pd.to_numeric(df['社平工资（年）'], errors='coerce')
avg_wage_raw = avg_wage_raw.fillna(avg_wage_raw.median())
reg['ln_avg_wage'] = np.log1p(avg_wage_raw.clip(lower=0))

# 30. Employee-scheme minimum annual contribution divided by individual income.
# Clip income at one yuan and the resulting burden ratio at 10.
zhigong_payment = pd.to_numeric(df['职工保缴费额下限（年）'], errors='coerce').fillna(7200)
reg['zhigong_burden'] = (zhigong_payment / ind_income_raw.clip(lower=1)).clip(upper=10)

# 31. Resident-scheme minimum annual contribution divided by individual income.
jumin_payment = pd.to_numeric(df['居民保缴费额下限（年）'], errors='coerce').fillna(200)
reg['jumin_burden'] = (jumin_payment / ind_income_raw.clip(lower=1)).clip(upper=10)

# 32. Log annual basic pension.
base_pension_raw = pd.to_numeric(df['基础养老金（年）'], errors='coerce')
base_pension_raw = base_pension_raw.fillna(base_pension_raw.median())
reg['ln_pension'] = np.log1p(base_pension_raw.clip(lower=0))

# ══════════════════════════════════════════════════════════════════════════════
# Outcomes.
# ══════════════════════════════════════════════════════════════════════════════

# 1. 2018 enrollment decision: participation = 1, non-participation = 0.
reg['decision'] = df['2018年参保决策'].map({'参保': 1, '不参保': 0}).astype(float)

# 2. 2019 account: employee = 1, resident = 0; undefined for non-participants.
reg['type'] = df['2019年参保账户'].map(
    {'城镇职工养老保险': 1, '城乡居民养老保险': 0}
).astype(float)

# ══════════════════════════════════════════════════════════════════════════════
# Save transformed regression data.
# ══════════════════════════════════════════════════════════════════════════════
out_file = PROCESSED_DIR / 'regression_data.csv'
reg.to_csv(out_file, index=False, encoding='utf-8-sig')
print(f"Saved to: {out_file}")
print(f"  Shape: {reg.shape}")
print("\nColumn names:")
print(list(reg.columns))
print("\n" + "=" * 80)
print("Step 2 complete")
print("=" * 80)
