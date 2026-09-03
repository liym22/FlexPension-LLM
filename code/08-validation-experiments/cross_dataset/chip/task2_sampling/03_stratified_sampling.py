"""
03_stratified_sampling.py
Draw a proportional stratified random sample of 500 flexible workers selected
by task 1, using "城镇保/居民保/不参保" as strata.

Stratum rules, in priority order:
  城镇保: A23 contains 1 or 4
  居民保: A23 contains neither 1 nor 4, but contains 5 or 6
  不参保: otherwise (contains none of 1, 4, 5, or 6)

Reproducibility: random_state=42
"""

import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IN_PATH  = SCRIPT_DIR.parent / "task1" / "filtered_flexible_workers.csv"
OUT_PATH = SCRIPT_DIR / "sampled_500.csv"

TOTAL_N     = 500
RANDOM_SEED = 42

# ============================================================
# 1. Load filtered data.
# ============================================================
df = pd.read_csv(IN_PATH, low_memory=False)
print(f"Input sample size: {len(df):,}")

# ============================================================
# 2. Parse A23 and assign priority-ordered strata.
# ============================================================
def classify_stratum(val):
    """
    Classify with priority 城镇保 > 居民保 > 不参保.
    """
    s = str(val).strip()
# Task 1 removes A23 special missing codes -88 and -99.
    if s in ('-88', '-99'):
        return '不参保'
    opts = set(s.split(','))
    if '1' in opts or '4' in opts:
        return '城镇保'
    elif '5' in opts or '6' in opts:
        return '居民保'
    else:
        return '不参保'

df['stratum'] = df['A23'].apply(classify_stratum)

# ============================================================
# 3. Compute proportional stratum allocations.
# ============================================================
stratum_order = ['城镇保', '居民保', '不参保']
counts = df['stratum'].value_counts()
total  = len(df)

# Round proportional allocations and then adjust to 500 total cases.
raw_alloc = {s: counts[s] / total * TOTAL_N for s in stratum_order}
alloc     = {s: int(round(v)) for s, v in raw_alloc.items()}

# Adjust the rounded allocation to exactly 500.
diff = TOTAL_N - sum(alloc.values())
if diff != 0:
# Add or remove cases according to fractional remainders.
    remainders = sorted(stratum_order,
                        key=lambda s: raw_alloc[s] - int(raw_alloc[s]),
                        reverse=(diff > 0))
    for i in range(abs(diff)):
        alloc[remainders[i]] += 1 if diff > 0 else -1

print(f"\n{'─'*55}")
print(f"{'Stratum':<8} {'Total':>8} {'Share':>7} {'Sample':>7}")
print(f"{'─'*55}")
for s in stratum_order:
    n = counts[s]
    print(f"  {s:<6} {n:>8,}  {n/total*100:>6.1f}%  {alloc[s]:>7}")
print(f"{'─'*55}")
print(f"  {'Total':<6} {total:>8,}  {'100.0%':>7}  {sum(alloc.values()):>7}")

# ============================================================
# 4. Draw the fixed-seed stratified sample.
# ============================================================
sampled_parts = []
for s in stratum_order:
    stratum_df = df[df['stratum'] == s]
    n_sample   = alloc[s]
    sampled    = stratum_df.sample(n=n_sample, random_state=RANDOM_SEED)
    sampled_parts.append(sampled)

sampled_df = pd.concat(sampled_parts, ignore_index=True)

# ============================================================
# 5. Validate and save artifacts.
# ============================================================
assert len(sampled_df) == TOTAL_N, f"Unexpected sample size: {len(sampled_df)}"

print(f"\nSample validation:")
for s in stratum_order:
    n = (sampled_df['stratum'] == s).sum()
    print(f"  {s}: {n} people")
print(f"  Total: {len(sampled_df)} people")
print(f"  Random seed: {RANDOM_SEED} (reproducible)")

sampled_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUT_PATH}")
