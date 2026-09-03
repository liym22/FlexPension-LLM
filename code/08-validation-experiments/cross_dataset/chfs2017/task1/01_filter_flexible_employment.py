#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_filter_flexible_employment.py
从 CHFS2017 个人数据中筛选灵活就业样本。

筛选条件：
  条件1  劳动年龄：男 16-59 岁，女 16-54 岁（age = 2017 - a2005）
  条件2  工作性质为灵活就业：a3132a ∈ {2, 4, 5, 6}
           2=临时工  4=自营  5=自由职业  6=其他
           （排除 3=务农：农业收入难以衡量，且缴费行为与其他灵活就业差异大）
  条件3  养老保险回答有效：f1001a 不为空 且 f1001a ∉ {1, 7777}
           排除 f1001a=1（机关事业，属强制参保，非研究对象）
           排除 f1001a=7777（其他险种，无法归入职工保/居民保任一类，不可用作 ground truth）

输出:
  task1/filtered_flexible_workers.csv
  task1/01_filter_log.txt
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
IND_DTA    = PROJECT_ROOT / "data" / "raw" / "chfs2017" / "chfs2017_ind_202206.dta"
OUT_CSV    = SCRIPT_DIR / "filtered_flexible_workers.csv"
LOG_PATH   = SCRIPT_DIR / "01_filter_log.txt"

# Logging
_log = open(LOG_PATH, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log.write(msg + "\n")
    _log.flush()

log("=" * 70)
log("Task 1: Filter the CHFS 2017 flexible-worker sample")
log("=" * 70)

# Load required DTA columns
log("\n[1] Loading selected columns from chfs2017_ind_202206.dta...")

USECOLS = [
    "hhid", "pline",
    "a2003",        # Gender: 1 = male, 2 = female.
    "a2005",        # Birth year.
    "a3132a",       # Employment type.
    "f1001a",       # Pension scheme used as ground truth.
    "f1008_imp",    # Monthly contribution used in task 2 stratification.
]

ind, meta = pyreadstat.read_dta(str(IND_DTA), usecols=USECOLS)
ind["hhid"]  = ind["hhid"].astype(str)
ind["pline"] = ind["pline"].astype(str)
log(f"  Raw individual records: {len(ind):,}")

# Build inclusion criteria
log("\n[2] Building filter conditions...")

# Criterion 1: working age.
ind["_age"] = 2017 - ind["a2005"]
cond1 = (
    ((ind["a2003"] == 1) & (ind["_age"] >= 16) & (ind["_age"] < 60)) |
    ((ind["a2003"] == 2) & (ind["_age"] >= 16) & (ind["_age"] < 55))
)

# Criterion 2: flexible employment excluding farm work.
cond2 = ind["a3132a"].isin([2, 4, 5, 6])  # Temporary, self-employed, freelance, or other.

# Criterion 3: f1001a maps to a target ground-truth class.
# Exclude government pension code 1 and unclassified code 7777.
cond3 = ind["f1001a"].notna() & (~ind["f1001a"].isin([1, 7777]))

total_cond = cond1 & cond2 & cond3

log("\n  Filter sequence (sample counts):")
log(f"  Raw sample                                      : {len(ind):>8,}")
log(f"  Condition 1: working age                        : {cond1.sum():>8,}")
log(f"  + Condition 2: flexible work (a3132a in {{2,4,5,6}}): {(cond1 & cond2).sum():>8,}")
log(f"  + Condition 3: classifiable f1001a              : {total_cond.sum():>8,}")

# Apply filters
filtered = ind[total_cond].copy()
filtered.drop(columns=["_age"], inplace=True)

log(f"\n  Final sample size: {len(filtered):,}")
log("\n  a3132a distribution (employment type):")
wt_map = {2: "临时工", 3: "务农", 4: "自营", 5: "自由职业", 6: "其他"}
for code, cnt in filtered["a3132a"].value_counts().sort_index().items():
    log(f"    {int(code)}={wt_map.get(int(code), '?')}: {cnt:,}")

log("\n  f1001a distribution (pension channel):")
ins_map = {2: "城镇职工", 3: "新农保", 4: "城镇居民", 5: "城乡统一居民",
           7788: "未参保"}  # Codes 1 and 7777 were excluded by criterion 3.
for code, cnt in filtered["f1001a"].value_counts().sort_index().items():
    log(f"    {int(code)}={ins_map.get(int(code), '?')}: {cnt:,}")

# Save artifacts
log("\n[3] Saving...")
filtered.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
log(f"  Saved: {OUT_CSV}")
log(f"  Rows: {len(filtered):,}; columns: {len(filtered.columns)}")
log(f"  Column names: {list(filtered.columns)}")
log("\nComplete")
_log.close()
