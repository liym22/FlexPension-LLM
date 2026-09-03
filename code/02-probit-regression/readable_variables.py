#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: 变量提取与可读化
- 仿照 reconstruct_variables.py，将 sampled_ids_filtered.csv 中所有样本的相关变量转换为可读方式
- 仿照 generate_prompts_v3.py，提取省级政策变量
- 输出：all_samples_with_policy.csv + variable_statistics_report.txt
"""

import sys

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path

print("=" * 80)
print("Step 1: Extract and reconstruct readable variables")
print("=" * 80)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from common.household_enrollment import count_other_enrolled_members

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: load source data.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 1] Loading data...")
try:
    # Sample ID table with hhid and pline columns.
    sampled_ids = pd.read_csv(PROCESSED_DIR / 'sampled_ids_filtered.csv',
                              dtype={'hhid': str, 'pline': str})
    print(f"Loaded sample IDs: {len(sampled_ids)} records; columns: {list(sampled_ids.columns)}")

    # Individual table.
    ind_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019' / 'chfs2019_ind_202112.dta'))
    ind_data['hhid']  = ind_data['hhid'].astype(str)
    ind_data['pline'] = ind_data['pline'].astype(str)
    print(f"Loaded individual data: {len(ind_data)} records")

    # Household table.
    hh_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019' / 'chfs2019_hh_202112.dta'))
    hh_data['hhid'] = hh_data['hhid'].astype(str)
    print(f"Loaded household data: {len(hh_data)} records")

    # Master table containing income, consumption, assets, debt, rural, and region fields.
    master_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019' / 'chfs2019_master_202112.dta'))
    master_data['hhid']  = master_data['hhid'].astype(str)
    master_data['pline'] = master_data['pline'].astype(str)
    print(f"Loaded master data: {len(master_data)} records")

    # Province-level policy data.
    policy_data = pd.read_excel(RAW_DIR / 'policy' / 'province_policy_2018.xlsx', sheet_name='总表')
    print(f"Loaded policy data: {len(policy_data)} rows")

except Exception as e:
    print(f"Failed to load data: {e}")
    raise

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: left-join source tables to sampled IDs.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2] Merging data...")

df = sampled_ids.merge(ind_data, on=['hhid', 'pline'], how='left')
print(f"After merging the individual table: {len(df)} records")

# Select required master-table columns to avoid name collisions.
master_cols = ['hhid', 'pline', 'total_income', 'total_consump',
               'total_asset', 'total_debt', 'rural', 'region']
master_cols = [c for c in master_cols if c in master_data.columns]
df = df.merge(master_data[master_cols], on=['hhid', 'pline'], how='left')
print(f"After merging the master table: {len(df)} records, {len(df.columns)} fields")

# Household risk, economic-expectation, and self-employment income fields.
hh_need_cols = ['hhid', 'h3104', 'h3601',
                'b2002b_imp', 'b2002', 'b2054', 'b2055_imp', 'b2003e_imp']
hh_need_cols = [c for c in hh_need_cols if c in hh_data.columns]
df = df.merge(hh_data[hh_need_cols], on='hhid', how='left')
print(f"After merging the household table: {len(df)} records, {len(df.columns)} fields")

# Age.
df['age'] = 2019 - df['a2005']

# Annual individual income, following reconstruct_variables.py.
def calculate_personal_income(row):
    work_type = row.get('a3132d', np.nan)
    income = 0.0
    if work_type == 2:  # Temporary worker.
        income += row.get('a3136_imp',  0) if pd.notna(row.get('a3136_imp'))  else 0
        income += row.get('a3136a_imp', 0) if pd.notna(row.get('a3136a_imp')) else 0
        for i in [1, 2, 3, 4, 5, 6, 7777]:
            v = row.get(f'a3136bb_{i}_imp', np.nan)
            income += v if pd.notna(v) else 0
            v = row.get(f'a3136bd_{i}_imp', np.nan)
            income += v if pd.notna(v) else 0
        v = row.get('a3138c', np.nan)
        income += v if pd.notna(v) else 0
    elif work_type == 6:  # Freelancer.
        v = row.get('a3136aa_imp', np.nan)
        income += v if pd.notna(v) else 0
    elif work_type in [3, 4]:  # Employer or self-employed worker.
        v = row.get('b2002b_imp', np.nan)
        income += v if pd.notna(v) else 0
        b2002 = row.get('b2002', np.nan)
        if pd.notna(b2002):
            if b2002 > 1:
                v = row.get('b2003e_imp', np.nan)
                income += v if pd.notna(v) else 0
            elif b2002 == 1:
                b2054 = row.get('b2054', np.nan)
                if pd.notna(b2054):
                    v = row.get('b2055_imp', np.nan)
                    if pd.notna(v):
                        income += v if b2054 == 1 else (-v if b2054 == 2 else 0)
    # Additional income for all workers.
    v = row.get('a3171_imp', np.nan)
    income += v if pd.notna(v) else 0
    return income

df['personal_income'] = df.apply(calculate_personal_income, axis=1)

# Enrollment-history variables aligned with reconstruct_variables.py.
def calc_insurance_vars(row):
    """
    一次性计算所有参保衍生变量，返回 dict：
      decision_2018   : 2018年参保决策
      account_2019    : 2019年参保账户
      hist_years      : 2018年前累计缴纳年限（int 或 None=不清楚）
      hist_type       : 2018年前历史参保类型
      hist_start_year : 2018年前开始缴纳年份（int 或 None）
      hist_gap        : 2018年前是否存在断缴
    """
    f1001a = row.get('f1001a', np.nan)
    f1008  = row.get('f1008_imp', np.nan)    # Employee-scheme contribution amount.
    f1008a = row.get('f1008a_imp', np.nan)   # Resident-scheme contribution amount.
    f1009  = row.get('f1009', np.nan)        # Contribution start year.
    f1009a = row.get('f1009a', np.nan)       # Cumulative contribution years.

    # 2018 enrollment decision.
    uninsured_2018 = (
        f1001a == 7788
        or (f1001a == 2 and pd.notna(f1008) and f1008 == 0)
        or (f1001a in [3, 4, 5] and pd.notna(f1008a) and f1008a == 0)
    )
    decision_2018 = '不参保' if uninsured_2018 else '参保'

    # 2019 pension account.
    if f1001a == 2:
        account_2019 = '城镇职工养老保险'
    elif f1001a in [3, 4, 5]:
        account_2019 = '城乡居民养老保险'
    else:
        account_2019 = '不参保'

    # Cumulative contribution years before 2018.
    if f1001a == 7788:
        hist_years = 0
    elif f1001a == 2 and pd.notna(f1008) and f1008 > 0 and pd.notna(f1009a):
        hist_years = max(0, int(np.floor(f1009a)) - 1)
    elif f1001a in [3, 4, 5] and pd.notna(f1008a) and f1008a > 0 and pd.notna(f1009a):
        hist_years = max(0, int(np.floor(f1009a)) - 1)
    elif pd.notna(f1001a) and f1001a != 7788 and pd.notna(f1009a):
        hist_years = int(np.floor(f1009a))
    else:  # f1001a differs from 7788 and f1009a is missing.
        hist_years = None  # Unknown.

    # Historical enrollment type before 2018.
    if pd.isna(f1001a) or f1001a == 7788:
        hist_type = '不参保'
    else:
        if pd.isna(f1009) and pd.isna(f1009a):
            hist_type = '不清楚'
        elif (pd.notna(f1009) and f1009 >= 2018) or hist_years == 0:
            hist_type = '不参保'
        else:
            hist_type = account_2019

    # Contribution start year before 2018.
    if hist_type in ['不参保', '不清楚']:
        hist_start_year = None
    elif pd.isna(f1009):
        hist_start_year = None
    elif f1009 < 2018:
        hist_start_year = int(f1009)
    else:
        hist_start_year = None

    # Whether a contribution interruption occurred before 2018.
    if hist_type == '不参保':
        hist_gap = '不存在'
    elif hist_type == '不清楚':
        hist_gap = '不清楚'
    else:
        if hist_years is not None and hist_start_year is not None:
            gap_year = 2018 - (hist_start_year + hist_years)
            hist_gap = f'断缴{int(gap_year)}年' if gap_year > 0 else '不存在'
        else:
            hist_gap = '不清楚'

    return {
        'decision_2018':    decision_2018,
        'account_2019':     account_2019,
        'hist_years':       hist_years,
        'hist_type':        hist_type,
        'hist_start_year':  hist_start_year,
        'hist_gap':         hist_gap,
    }

_ins = df.apply(calc_insurance_vars, axis=1, result_type='expand')
df['decision_2018']   = _ins['decision_2018']
df['account_2019']    = _ins['account_2019']
df['hist_years']      = _ins['hist_years']       # Integer or None.
df['hist_type']       = _ins['hist_type']
df['hist_start_year'] = _ins['hist_start_year']  # Integer or None.
df['hist_gap']        = _ins['hist_gap']

# Household composition statistics require the full individual table.
print("  Computing household structure (this may take a moment)...")
ind_data['_insured_for_household_count'] = (
    (ind_data['f1001a'] != 7788) & ind_data['f1001a'].notna()
).astype(int)
family_stats = []
for hhid, grp in ind_data.groupby('hhid'):
    ages = 2019 - grp['a2005']
    # Pension receipt is indicated by f1003 == 1.
    receiving = (grp['f1003'] == 1) if 'f1003' in grp.columns else pd.Series(False, index=grp.index)
    pension_col = 'f1005_imp' if 'f1005_imp' in grp.columns else None
    family_pension = grp.loc[receiving, pension_col].sum() if pension_col else 0

    family_stats.append({
        'hhid':            hhid,
        'household_size':  len(grp),
        'children_num':    int((ages < 16).sum()),
        'elderly_num':     int((ages >= 60).sum()),
        'family_receiving': int(receiving.sum()),
        'family_pension':  float(family_pension),
    })

family_df = pd.DataFrame(family_stats)
df = df.merge(family_df, on='hhid', how='left')
df['family_insured'] = count_other_enrolled_members(
    ind_data,
    df,
    household_col='hhid',
    person_col='pline',
    enrolled_col='_insured_for_household_count',
)
print("Derived-variable computation complete")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: define readable-value mappings.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 4] Defining mapping functions...")

def map_gender(val):
    return {1: '男', 2: '女'}.get(val, '不清楚')

def map_education(val):
    if pd.isna(val): return '不清楚'
    return {1:'未上学', 2:'小学', 3:'初中', 4:'高中', 5:'中专/职高',
            6:'大专', 7:'本科', 8:'硕士', 9:'博士'}.get(val, '不清楚')

def map_health(val):
    return {1:'非常好', 2:'好', 3:'一般', 4:'不好', 5:'非常不好'}.get(val, '不清楚')

def map_hukou(val):
    return {1:'农业', 2:'非农业', 3:'统一居民', 4:'无户口', 7777:'其他'}.get(val, '不清楚')

def get_residence_prov(row):
    """Use hukou province when co-resident; otherwise use a2016b_prov."""
    if row.get('a2000c') == 1:
        return row.get('a2019_prov', '')
    elif row.get('a2000c') == 2:
        res = row.get('a2016b_prov', '')
        return res if pd.notna(res) and res != '' else row.get('a2019_prov', '')
    return row.get('a2019_prov', '')

def map_work_type(val):
    return {2:'临时工', 3:'雇主', 4:'自营', 5:'帮工',
            6:'自由职业', 7:'务农'}.get(val, '不清楚')

def map_industry(val):
    if pd.isna(val): return '不清楚'
    return {1:'农林牧渔', 2:'采矿制造', 3:'建筑', 4:'电力燃气水', 5:'批发零售',
            6:'交通运输', 7:'住宿餐饮', 8:'信息技术', 9:'金融', 10:'房地产',
            11:'科教文卫', 12:'居民服务', 13:'公共管理', 7777:'其他'}.get(val, '不清楚')

def map_occupation(val):
    if pd.isna(val): return '不清楚'
    return {1:'负责人', 2:'专业技术', 3:'办事人员', 4:'快递员',
            5:'服务人员', 6:'生产制造', 7777:'其他'}.get(val, '不清楚')

def map_employer(val):
    if pd.isna(val): return '不清楚'
    return {1:'机关事业', 2:'国企', 4:'个体户', 5:'私企',
            6:'外资/港澳台', 7777:'其他'}.get(val, '不清楚')

def format_hist_years(val):
    """Format pre-2018 contribution years; map missing values to '不清楚'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '不清楚'
    return f'{int(val)}年'

def format_hist_start_year(val):
    """Format the contribution start year; map missing values to an empty string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ''
    return f'{int(val)}年'

def map_risk_preference(val):
    if pd.isna(val): return '不知道'
    return {1:'高风险偏好', 2:'略高风险偏好', 3:'平均风险偏好',
            4:'略低风险偏好', 5:'极度厌恶风险', 6:'不知道'}.get(val, '不知道')

def map_expectation(val):
    if pd.isna(val): return '不清楚'
    return {1:'非常好', 2:'比较好', 3:'基本不变',
            4:'比较差', 5:'非常差'}.get(val, '不清楚')

def map_rural(val):
    return '城镇' if val == 0 else '农村'

def map_region(val):
    return {1: '东部', 2: '中部', 3: '西部', 4: '东北'}.get(val, '不清楚')

def format_income(val):
    if pd.isna(val): return '不清楚'
    return f'{int(val)}元'

def format_years(val):
    if pd.isna(val): return '不清楚'
    return f'{val:.1f}年'.replace('.0年', '年')

print("Mapping functions defined")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: build the readable sample table.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 5] Building the readable data frame...")

rec = pd.DataFrame()

# Identifiers.
rec['家庭ID'] = df['hhid']
rec['个人ID'] = df['pline']

# Individual characteristics.
rec['年龄']     = df['age']
rec['性别']     = df['a2003'].apply(map_gender)
rec['文化程度'] = df['a2012'].apply(map_education)
rec['健康状况'] = df['a2025b'].apply(map_health)

# Hukou and migration.
rec['户口性质'] = df['a2022'].apply(map_hukou)
rec['户口省份'] = df['a2019_prov']                               # Already a province-name string.
rec['常住省份'] = df.apply(get_residence_prov, axis=1)
rec['是否流动'] = df['a2000c'].map({1: '否', 2: '是'}).fillna('否')

# Employment.
rec['工作性质'] = df['a3132d'].apply(map_work_type)
rec['工作行业'] = df['a3132f'].apply(map_industry)
rec['工作职业'] = df['a3132g'].apply(map_occupation)
rec['单位类型'] = df['a3132c'].apply(map_employer)

# Income and assets, retaining the original yuan unit.
rec['个人年收入'] = df['personal_income'].apply(format_income)
rec['家庭总收入'] = df['total_income'].apply(format_income)   if 'total_income'   in df.columns else '不清楚'
rec['家庭总消费'] = df['total_consump'].apply(format_income)  if 'total_consump'  in df.columns else '不清楚'
rec['家庭总资产'] = df['total_asset'].apply(format_income)    if 'total_asset'    in df.columns else '不清楚'
rec['家庭总负债'] = df['total_debt'].apply(format_income)     if 'total_debt'     in df.columns else '不清楚'

# Household composition.
rec['家庭人数'] = df['household_size'].astype(int)
rec['子女数']   = df['children_num'].astype(int)
rec['老人数']   = df['elderly_num'].astype(int)

# Pension enrollment.
rec['2018年参保决策'] = df['decision_2018']
rec['2019年参保账户'] = df['account_2019']
rec['历史参保类型']   = df['hist_type']
rec['开始缴纳年份']   = df['hist_start_year'].apply(format_hist_start_year)
rec['累计缴纳年限']   = df['hist_years'].apply(format_hist_years)
rec['是否存在断缴']   = df['hist_gap']
rec['家庭参保人数'] = df['family_insured'].astype(int)
rec['家庭领取人数'] = df['family_receiving'].astype(int)
rec['家庭月均养老金'] = df['family_pension'].apply(
    lambda x: f'{int(x)}元' if pd.notna(x) and x > 0 else '未领取')

# Behavioral variables.
rec['风险偏好'] = df['h3104'].apply(map_risk_preference)
rec['经济预期'] = df['h3601'].apply(map_expectation)

# Region.
rec['城乡'] = df['rural'].apply(map_rural)    if 'rural'   in df.columns else '不清楚'
rec['地区'] = df['region'].apply(map_region)  if 'region'  in df.columns else '不清楚'

print(f"Reconstruction complete: {len(rec.columns)} fields, {len(rec)} records")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: match province-level policy variables.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 6] Matching province-level policy variables...")

def match_policy(province_name, policy_df):
    # Map Hong Kong variants and empty values to Guangdong Province.
    if not isinstance(province_name, str) or province_name.strip() in ('', '香港', '中国香港'):
        province_name = '广东省'
    row = policy_df[policy_df['省名'] == province_name]
    if len(row) == 0:
        # Default policy values.
        return {
            '社平工资（年）':        60000,
            '职工保缴费额下限（年）': 7200,
            '基础养老金（年）':       1200,
            '居民保缴费额下限（年）': 200,
        }
    row = row.iloc[0]

    def safe_int(key, default):
        val = row.get(key, default)
        try:
            if isinstance(val, str):
                val = val.replace('%', '')
            return int(float(val))
        except Exception:
            return default

    return {
        '社平工资（年）':        safe_int('社平工资（年）', 60000),
        '职工保缴费额下限（年）': safe_int('缴费金额（年，下限）', 7200),
        '基础养老金（年）':       safe_int('基础养老金（年）', 1200),
        '居民保缴费额下限（年）': safe_int('最低缴费金额（年）', 200),
    }

# Match each unique province to policy values before merging.
unique_provs = rec['户口省份'].unique()
prov_policy_map = {p: match_policy(p, policy_data) for p in unique_provs}
missing_provs = [p for p, v in prov_policy_map.items()
                 if v['基础养老金（年）'] == 1200 and v['社平工资（年）'] == 60000]
if missing_provs:
    print(f"  Provinces missing from the policy table; defaults used: {missing_provs}")

policy_records = rec['户口省份'].map(prov_policy_map)
policy_df_matched = pd.DataFrame(list(policy_records))
final_data = pd.concat([rec.reset_index(drop=True),
                        policy_df_matched.reset_index(drop=True)], axis=1)
print(f"Policy matching complete; final field count: {len(final_data.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: save the readable dataset.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 7] Saving results...")
output_file = PROCESSED_DIR / 'all_samples_with_policy.csv'
final_data.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Saved to: {output_file}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: generate the variable summary report.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 8] Generating the statistical report...")

report = []
report.append("=" * 80)
report.append("变量统计报告")
report.append("=" * 80)
report.append(f"\n总样本数: {len(final_data)}")

for col in final_data.columns:
    report.append("\n" + "=" * 80)
    report.append(f"变量: {col}")
    report.append("-" * 40)

    series = final_data[col]
    if series.dtype in ['int64', 'float64', 'int32', 'float32']:
        desc = series.describe()
        report.append("  类型: 连续变量")
        report.append(f"  样本数  : {int(desc['count'])}")
        report.append(f"  均值    : {desc['mean']:.2f}")
        report.append(f"  中位数  : {series.median():.2f}")
        report.append(f"  标准差  : {desc['std']:.2f}")
        report.append(f"  最小值  : {desc['min']:.2f}")
        report.append(f"  最大值  : {desc['max']:.2f}")
        report.append(f"  缺失值  : {series.isna().sum()} ({series.isna().sum()/len(series)*100:.1f}%)")
    else:
        vc = series.value_counts(dropna=False)
        total = len(series)
        report.append("  类型: 分类变量")
        report.append("  取值分布:")
        for val, cnt in vc.items():
            report.append(f"    {val}: {cnt} ({cnt/total*100:.1f}%)")
        report.append(f"  缺失值  : {series.isna().sum()} ({series.isna().sum()/total*100:.1f}%)")

report_text = "\n".join(report)
report_file = PROCESSED_DIR / 'variable_statistics_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"Statistical report saved to: {report_file}")

print("\n" + "=" * 80)
print("Step 1 complete")
print("=" * 80)
print("\nOutput files:")
print(f"  1. {output_file}")
print(f"  2. {report_file}")
