#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_stratified_sampling.py
Draw a stratified random sample of 500 flexible workers selected by task 1,
using pension scheme as the stratum.

Strata (using ins_type calculated by task 1):
  城镇职工养老保险 -> stratum 1
  城乡居民养老保险 -> stratum 2
  不参保           -> stratum 3

Reproducibility: random_state=42

Outputs:
  task2_sampling/sampled_500.csv
  task2_sampling/02_sampling_log.txt
"""

from pathlib import Path
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
IN_CSV     = SCRIPT_DIR.parent / "task1" / "filtered_flexible_workers.csv"
OUT_CSV    = SCRIPT_DIR / "sampled_500.csv"
LOG_PATH   = SCRIPT_DIR / "02_sampling_log.txt"

TOTAL_N     = 500
RANDOM_SEED = 42

_log = open(LOG_PATH, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log.write(msg + "\n")
    _log.flush()

log("=" * 70)
log("Task 2  CFPS 2018 stratified random sampling (N=500)")
log("=" * 70)

# Load filtered data
log(f"\n[1] Loading {IN_CSV.name}...")
df = pd.read_csv(IN_CSV)
log(f"  Input sample size: {len(df):,}")

# Allocate strata
log("\n[2] Allocating pension-scheme strata...")

STRATUM_MAP = {
    "城镇职工养老保险": "城镇职工保",
    "城乡居民养老保险": "城乡居民保",
    "不参保":           "不参保",
}
stratum_order = ["城镇职工保", "城乡居民保", "不参保"]

df["stratum"] = df["ins_type"].map(STRATUM_MAP)

counts = df["stratum"].value_counts()
total  = len(df)

# Use largest-remainder allocation to preserve TOTAL_N.
raw_alloc = {s: counts.get(s, 0) / total * TOTAL_N for s in stratum_order}
alloc     = {s: int(round(v)) for s, v in raw_alloc.items()}

diff = TOTAL_N - sum(alloc.values())
if diff != 0:
    remainders = sorted(
        stratum_order,
        key=lambda s: raw_alloc[s] - int(raw_alloc[s]),
        reverse=(diff > 0),
    )
    for i in range(abs(diff)):
        alloc[remainders[i]] += 1 if diff > 0 else -1

log(f"\n  {'Stratum':<12} {'Total':>8} {'Share':>7} {'Sample':>7}")
log(f"  {'─'*42}")
for s in stratum_order:
    n = counts.get(s, 0)
    log(f"  {s:<12} {n:>8,}  {n/total*100:>6.1f}%  {alloc[s]:>7}")
log(f"  {'─'*42}")
log(f"  {'Total':<12} {total:>8,}  {'100.0%':>7}  {sum(alloc.values()):>7}")

# Draw stratified sample
log(f"\n[3] Drawing stratified random sample (seed={RANDOM_SEED})...")

sampled_parts = []
for s in stratum_order:
    sub  = df[df["stratum"] == s]
    n    = alloc[s]
    samp = sub.sample(n=n, random_state=RANDOM_SEED)
    sampled_parts.append(samp)

sampled_df = pd.concat(sampled_parts, ignore_index=True)
assert len(sampled_df) == TOTAL_N

log(f"\n  Sample validation:")
for s in stratum_order:
    n = (sampled_df["stratum"] == s).sum()
    log(f"    {s}: {n} people")
log(f"    Total: {len(sampled_df)} people (seed={RANDOM_SEED}, reproducible)")

# Save artifacts
log(f"\n[4] Saving...")
sampled_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
log(f"  Saved: {OUT_CSV}")
log(f"  Rows: {len(sampled_df)}  Columns: {len(sampled_df.columns)}")
log("\nDone.")
_log.close()
