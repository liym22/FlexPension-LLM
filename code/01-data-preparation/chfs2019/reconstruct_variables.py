#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reconstruct CHFS 2019 variables for readable prompt generation."""

import sys

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from common.household_enrollment import count_other_enrolled_members

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chfs2019"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CHFS 2019 variable reconstruction")
print("=" * 80)

# 1. Load source data.
print("\n[Step 1] Load data...")
try:
    # Load the 30 screening-sample IDs.
    sample_30 = pd.read_csv(PROCESSED_DIR / 'sample_30.csv', dtype={'hhid': str})
    print(f"✓ Loaded {len(sample_30)} sampled respondent IDs")
    
    # Load the full individual and household tables.
    ind_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019_ind_202112.dta'))
    ind_data['hhid'] = ind_data['hhid'].astype(str)
    print("✓ Loaded the individual table")
    
    hh_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019_hh_202112.dta'))
    hh_data['hhid'] = hh_data['hhid'].astype(str)
    print("✓ Loaded the household table")
    
    master_data, _ = pyreadstat.read_dta(str(RAW_DIR / 'chfs2019_master_202112.dta'))
    master_data['hhid'] = master_data['hhid'].astype(str)
    print("✓ Loaded the master table")
    
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    exit(1)

# 2. Join source tables.
print("\n[Step 2] Merge required tables...")
# Join individual characteristics.
sample_30_full = sample_30.merge(ind_data, on=['hhid', 'pline'], how='left')
print(f"✓ Respondents after merging the individual table: {len(sample_30_full)}")

# Join household-level fields by hhid.
sample_30_full = sample_30_full.merge(
    hh_data[['hhid', 'h3104', 'h3601', 'b2002b_imp', 'b2002', 'b2054', 'b2055_imp', 'b2003e_imp']], 
    on='hhid', how='left'
)
print(f"✓ Respondents after merging the household table: {len(sample_30_full)}")

# Join person-level derived fields from the master table by hhid and pline.
sample_30_full = sample_30_full.merge(
    master_data[['hhid', 'pline', 'total_income', 'total_consump', 'total_asset', 'total_debt', 'rural', 'region']], 
    on=['hhid', 'pline'], how='left'
)
print(f"✓ Respondents after merging the master table: {len(sample_30_full)}")

sample_30 = sample_30_full
print(f"✓ Merge complete: {len(sample_30)} respondents and {len(sample_30.columns)} fields")

# 3. Construct derived variables.
print("\n[Step 3] Calculate derived variables...")

# 3.1 Age.
sample_30['age'] = 2019 - sample_30['a2005']

# 3.2 Individual income by employment type.
def calculate_personal_income(row):
    work_type = row['a3132d']
    income = 0
    
    # Temporary-worker income.
    if work_type == 2:
        # Wages.
        income += row['a3136_imp'] if pd.notna(row['a3136_imp']) else 0
        # Bonuses.
        income += row['a3136a_imp'] if pd.notna(row['a3136a_imp']) else 0
        # Cash benefits.
        for i in [1, 2, 3, 4, 5, 6, 7777]:
            col_name = f'a3136bb_{i}_imp'
            if col_name in row.index:
                income += row[col_name] if pd.notna(row[col_name]) else 0
        # In-kind benefits.
        for i in [1, 2, 3, 4, 5, 6, 7777]:
            col_name = f'a3136bd_{i}_imp'
            if col_name in row.index:
                income += row[col_name] if pd.notna(row[col_name]) else 0
        # Reimbursements.
        income += row['a3138c'] if pd.notna(row['a3138c']) else 0
    
    # Freelancer income.
    elif work_type == 6:
        income += row['a3136aa_imp'] if pd.notna(row['a3136aa_imp']) else 0
    
    # Employer or self-employed income.
    elif work_type in [3, 4]:
        # Part A: closed business projects.
        income += row['b2002b_imp'] if pd.notna(row['b2002b_imp']) else 0
        
        # Part B: active business projects.
        b2002 = row['b2002']
        if pd.notna(b2002):
            if b2002 > 1:  # Multiple projects.
                income += row['b2003e_imp'] if pd.notna(row['b2003e_imp']) else 0
            elif b2002 == 1:  # Single project.
                b2054 = row['b2054']
                if pd.notna(b2054):
                    if b2054 == 1:  # Profit.
                        income += row['b2055_imp'] if pd.notna(row['b2055_imp']) else 0
                    elif b2054 == 2:  # Loss.
                        income -= row['b2055_imp'] if pd.notna(row['b2055_imp']) else 0
                    # b2054 == 3 denotes break-even.
    
    # Family helper.
    elif work_type == 5:
        income += 0
    
    # Additional income for all employment types.
    income += row['a3171_imp'] if pd.notna(row['a3171_imp']) else 0
    
    return income

sample_30['personal_income'] = sample_30.apply(calculate_personal_income, axis=1)

# 3.3 Construct income terciles.
# Recompute cutoffs from the filtered sample and master table to match stratified_sampling.py.
_filtered_ids = pd.read_csv(PROCESSED_DIR / 'sampled_ids_filtered.csv', dtype={'hhid': str})
_filtered_with_income = _filtered_ids.merge(
    master_data[['hhid', 'pline', 'total_income']], on=['hhid', 'pline'], how='left'
)
_filtered_unique = (
    _filtered_with_income
    .sort_values(['hhid', 'pline'], kind='mergesort')
    .drop_duplicates(subset='hhid', keep='first')
    .dropna(subset=['total_income'])
)
income_tercile_33 = _filtered_unique['total_income'].quantile(1/3, interpolation='linear')
income_tercile_67 = _filtered_unique['total_income'].quantile(2/3, interpolation='linear')
print("  Income-tercile thresholds from sampled_ids_filtered.csv: "
      f"{income_tercile_33:,.2f} / {income_tercile_67:,.2f} CNY")

def income_group(income):
    if pd.isna(income):
        return None
    elif income <= income_tercile_33:
        return 'low'
    elif income <= income_tercile_67:
        return 'mid'
    else:
        return 'high'

sample_30['income_tercile'] = sample_30['total_income'].apply(income_group)

# 3.4 Construct enrollment-history variables.
def calc_insurance_vars(row):
    """Calculate all derived pension-enrollment variables for one respondent."""
    f1001a = row['f1001a']
    f1008  = row['f1008_imp']    # Employee-scheme contribution amount.
    f1008a = row['f1008a_imp']   # Resident-scheme contribution amount.
    f1009  = row['f1009']        # Contribution start year.
    f1009a = row['f1009a']       # Cumulative contribution years.

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
    elif f1001a != 7788 and pd.notna(f1009a):
        hist_years = int(np.floor(f1009a))
    else:  # f1001a differs from 7788 and f1009a is missing.
        hist_years = None  # Unknown.

    # Historical enrollment type before 2018.
    if f1001a == 7788:
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

_ins = sample_30.apply(calc_insurance_vars, axis=1, result_type='expand')
sample_30['decision_2018']   = _ins['decision_2018']
sample_30['account_2019']    = _ins['account_2019']
sample_30['hist_years']      = _ins['hist_years']       # Integer or None.
sample_30['hist_type']       = _ins['hist_type']
sample_30['hist_start_year'] = _ins['hist_start_year']  # Integer or None.
sample_30['hist_gap']        = _ins['hist_gap']

# 3.5 Household composition statistics.
ind_data['_insured_for_household_count'] = (
    (ind_data['f1001a'] != 7788) & ind_data['f1001a'].notna()
).astype(int)
family_stats = []
for hhid in sample_30['hhid'].unique():
    family_members = ind_data[ind_data['hhid'] == hhid]
    
    household_size = len(family_members)
    children_num = ((2019 - family_members['a2005']) < 16).sum()
    elderly_num = ((2019 - family_members['a2005']) >= 60).sum()
    
    # Pension receipt counts.
    family_receiving = (family_members['f1003'] == 1).sum()
    
    # Average household pension.
    receiving_members = family_members[family_members['f1003'] == 1]
    family_pension = receiving_members['f1005_imp'].sum() if len(receiving_members) > 0 else 0
    
    family_stats.append({
        'hhid': hhid,
        'household_size': household_size,
        'children_num': children_num,
        'elderly_num': elderly_num,
        'family_receiving': family_receiving,
        'family_pension': family_pension
    })

family_stats_df = pd.DataFrame(family_stats)
sample_30 = sample_30.merge(family_stats_df, on='hhid', how='left')
sample_30['family_insured'] = count_other_enrolled_members(
    ind_data,
    sample_30,
    household_col='hhid',
    person_col='pline',
    enrolled_col='_insured_for_household_count',
)

print("✓ Derived-variable calculation complete")

# 4. Map survey codes to readable values.
print("\n[Step 4] Map survey codes to readable values...")

def map_gender(val):
    mapping = {1: '男', 2: '女'}
    return mapping.get(val, f'未知({val})')

def map_education(val):
    if pd.isna(val):
        return '不清楚'
    mapping = {
        1: '未上学', 2: '小学', 3: '初中', 4: '高中',
        5: '中专/职高', 6: '大专', 7: '本科', 8: '硕士', 9: '博士'
    }
    return mapping.get(val, f'其他({int(val)})')

def map_health(val):
    mapping = {1: '非常好', 2: '好', 3: '一般', 4: '不好', 5: '非常不好'}
    return mapping.get(val, f'未知({val})')

def map_hukou(val):
    mapping = {1: '农业', 2: '非农业', 3: '统一居民', 4: '无户口', 7777: '其他'}
    return mapping.get(val, f'未知({val})')

def map_work_type(val):
    mapping = {
        2: '临时工', 3: '雇主', 4: '自营', 5: '帮工', 6: '自由职业', 7: '务农'
    }
    return mapping.get(val, f'未知({val})')

def map_industry(val):
    if pd.isna(val):
        return '不清楚'
    mapping = {
        1: '农林牧渔', 2: '采矿制造', 3: '建筑', 4: '电力燃气水',
        5: '批发零售', 6: '交通运输', 7: '住宿餐饮', 8: '信息技术',
        9: '金融', 10: '房地产', 11: '科教文卫', 12: '居民服务', 
        13: '公共管理', 7777: '其他'
    }
    return mapping.get(val, f'未知({int(val)})')

def map_occupation(val):
    if pd.isna(val):
        return '不清楚'
    mapping = {
        1: '负责人', 2: '专业技术', 3: '办事人员', 4: '快递员',
        5: '服务人员', 6: '生产制造', 7777: '其他'
    }
    return mapping.get(val, f'未知({int(val)})')

def map_employer(val):
    if pd.isna(val):
        return '不清楚'
    mapping = {
        1: '机关事业', 2: '国企', 4: '个体户', 5: '私企', 
        6: '外资/港澳台', 7777: '其他'
    }
    return mapping.get(val, f'未知({int(val)})')

def map_insurance(val):
    if pd.isna(val) or val == 7788:
        return '未参保'
    return '已参保'

def get_insurance_detail(val):
    """Return the pension scheme for an enrolled respondent."""
    if pd.isna(val) or val == 7788:
        return ''
    # Collapse enrolled accounts into the two target schemes.
    if val in [1, 2]:
        return '城镇职工养老保险'
    elif val in [3, 4, 5]:
        return '城乡居民养老保险'
    else:
        return '其他'  # Catch-all for unrecognized enrolled account codes.

def map_risk_preference(val):
    if pd.isna(val):
        return '不清楚'
    mapping = {
        1: '高风险偏好', 2: '略高风险偏好', 3: '平均风险偏好',
        4: '略低风险偏好', 5: '极度厌恶风险', 6: '不知道'
    }
    return mapping.get(val, f'未知({int(val)})')

def map_expectation(val):
    if pd.isna(val):
        return '不清楚'
    mapping = {
        1: '非常好', 2: '比较好', 3: '基本不变', 4: '比较差', 5: '非常差'
    }
    return mapping.get(val, f'未知({int(val)})')

def map_rural(val):
    return '城镇' if val == 0 else '农村'

def map_region(val):
    mapping = {1: '东部', 2: '中部', 3: '西部', 4: '东北'}
    return mapping.get(val, f'未知({val})')

def map_income_tercile(val):
    mapping = {'low': '低收入组', 'mid': '中收入组', 'high': '高收入组'}
    return mapping.get(val, val)

def format_year(val):
    """Format a calendar year for the Chinese prompt."""
    if pd.isna(val):
        return '不清楚'
    return f'{int(val)}年'

def format_years(val):
    """Format a duration, which may contain a fractional year."""
    if pd.isna(val):
        return '不清楚'
    # Retain one decimal place and remove a redundant trailing zero.
    return f'{val:.1f}年'.replace('.0年', '年')

def format_hist_years(val):
    """Format cumulative contribution years before 2018."""
    if val is None or pd.isna(val):
        return '不清楚'
    return f'{int(val)}年'

def format_hist_start_year(val):
    """Format the historical contribution start year for the Chinese prompt."""
    if val is None or pd.isna(val):
        return ''
    return f'{int(val)}年'

# 5. Build the reconstructed sample table.
print("\n[Step 5] Build the reconstructed table...")

reconstructed = pd.DataFrame()

# Identifiers.
reconstructed['家庭ID'] = sample_30['hhid']
reconstructed['个人ID'] = sample_30['pline']

# Individual characteristics.
reconstructed['年龄'] = sample_30['age']
reconstructed['性别'] = sample_30['a2003'].apply(map_gender)
reconstructed['文化程度'] = sample_30['a2012'].apply(map_education)
reconstructed['健康状况'] = sample_30['a2025b'].apply(map_health)
reconstructed['户口性质'] = sample_30['a2022'].apply(map_hukou)
reconstructed['户口省份'] = sample_30['a2019_prov']

# Residence province based on whether hukou and residence coincide.
def get_residence_prov(row):
    if row['a2000c'] == 1:  # Same province.
        return row['a2019_prov']
    elif row['a2000c'] == 2:  # Different province.
        # Validate a2016b_prov.
        residence = row['a2016b_prov']
        if pd.notna(residence) and residence != '':
            return residence
        else:
            return row['a2019_prov']  # Fall back to hukou province.
    else:
        return row['a2019_prov']  # Default to hukou province.

reconstructed['常住省份'] = sample_30.apply(get_residence_prov, axis=1)
reconstructed['是否流动'] = (sample_30['a2000c'] == 2).map({True: '是', False: '否'})

# Employment and income.
reconstructed['工作性质'] = sample_30['a3132d'].apply(map_work_type)
reconstructed['工作行业'] = sample_30['a3132f'].apply(map_industry)
reconstructed['工作职业'] = sample_30['a3132g'].apply(map_occupation)
reconstructed['单位类型'] = sample_30['a3132c'].apply(map_employer)

# Preserve zero income and display missing income as unknown.
def format_income(val):
    if pd.isna(val):
        return '不清楚'
    val = int(val)
    if val == 0:
        return '0元'
    return f'{val}元'

def format_large_amount(val, threshold=10000):
    """Format a large monetary value for the Chinese prompt."""
    if pd.isna(val):
        return '不清楚'
    val = int(val)
    return f'{val}元'

reconstructed['个人年收入'] = sample_30['personal_income'].apply(format_income)
reconstructed['家庭总收入'] = sample_30['total_income'].apply(format_large_amount)
reconstructed['家庭总消费'] = sample_30['total_consump'].apply(format_large_amount)

# Household finances.
reconstructed['家庭总资产'] = sample_30['total_asset'].apply(format_large_amount)
reconstructed['家庭总负债'] = sample_30['total_debt'].apply(format_large_amount)
reconstructed['家庭人数'] = sample_30['household_size'].astype(int)
reconstructed['子女数'] = sample_30['children_num'].astype(int)
reconstructed['老人数'] = sample_30['elderly_num'].astype(int)

# Pension enrollment.
reconstructed['2018年参保决策'] = sample_30['decision_2018']
reconstructed['2019年参保账户'] = sample_30['account_2019']
reconstructed['历史参保类型']   = sample_30['hist_type']
reconstructed['开始缴纳年份']   = sample_30['hist_start_year'].apply(format_hist_start_year)
reconstructed['累计缴纳年限']   = sample_30['hist_years'].apply(format_hist_years)
reconstructed['是否存在断缴']   = sample_30['hist_gap']
reconstructed['家庭参保人数'] = sample_30['family_insured'].astype(int)
reconstructed['家庭领取人数'] = sample_30['family_receiving'].astype(int)
reconstructed['家庭月均养老金'] = sample_30['family_pension'].apply(
    lambda x: f'{int(x)}元' if x > 0 else '未领取'
)

# Behavioral variables.
reconstructed['风险偏好'] = sample_30['h3104'].apply(map_risk_preference)
reconstructed['经济预期'] = sample_30['h3601'].apply(map_expectation)

# Region.
reconstructed['城乡'] = sample_30['rural'].apply(map_rural)
reconstructed['地区'] = sample_30['region'].apply(map_region)
reconstructed['收入分组'] = sample_30['income_tercile'].apply(map_income_tercile)

print(f"✓ Reconstruction complete with {len(reconstructed.columns)} fields")

# 5.1 Enrollment fields already follow the decision-to-history sequence used by prompts.

# 6. Save reconstructed data.
print("\n[Step 6] Save results...")

# Save CSV output.
reconstructed.to_csv(PROCESSED_DIR / 'sample_30_reconstructed.csv', index=False, encoding='utf-8-sig')
print("✓ Saved sample_30_reconstructed.csv")

# Save an Excel copy for local inspection.
try:
    reconstructed.to_excel(PROCESSED_DIR / 'sample_30_reconstructed.xlsx', index=False, engine='openpyxl')
    print("✓ Saved sample_30_reconstructed.xlsx")
except Exception as e:
    print(f"⚠ Failed to save the Excel file: {e}")

# 7. Print examples.
print("\n" + "=" * 80)
print("Reconstructed data preview (first 3 rows)")
print("=" * 80)
print(reconstructed.head(3).to_string())

# 8. Print summary statistics.
print("\n" + "=" * 80)
print("Variable Summary")
print("=" * 80)

print("\nGender distribution")
print(reconstructed['性别'].value_counts())

print("\nEmployment-type distribution")
print(reconstructed['工作性质'].value_counts())

print("\nPension-enrollment distribution")
print(reconstructed['2018年参保决策'].value_counts())

print("\nRisk-preference distribution")
print(reconstructed['风险偏好'].value_counts())

print("\nIncome summary")
# Parse numeric income values from formatted strings.
personal_income_values = sample_30['personal_income'].fillna(0)
family_income_values = sample_30['total_income'].fillna(0)

print(f"Personal annual income - mean: {personal_income_values.mean():,.0f} CNY, "
      f"median: {personal_income_values.median():,.0f} CNY, "
      f"range: {personal_income_values.min():,.0f}--{personal_income_values.max():,.0f} CNY")
print(f"Total household income - mean: {family_income_values.mean():,.0f} CNY, "
      f"median: {family_income_values.median():,.0f} CNY")

print("\n" + "=" * 80)
print("Variable reconstruction complete")
print("=" * 80)
