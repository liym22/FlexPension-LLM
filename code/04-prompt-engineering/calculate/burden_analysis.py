#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute pension contribution burden indicators.

For resident and employee pensions, the minimum annual contribution is divided
by personal income, household income per capita, and household net assets per
capita. Invalid denominators map to NaN, and the representative burden is the
minimum positive ratio across the three capacity measures.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---- Font configuration ----
rcParams["font.family"] = [
    "Heiti TC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "sans-serif",
]
rcParams["axes.unicode_minus"] = False

# ==================== Paths ====================
_CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_DIR))

from config.paths import ALL_SAMPLES_WITH_POLICY_FILE, BURDEN_DATA_FILE

DATA_CSV = ALL_SAMPLES_WITH_POLICY_FILE
OUTPUT_DIR = BURDEN_DATA_FILE.parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Pension Contribution Burden Analysis")
print("=" * 70)

# ==================== Load data ====================
print("\n[Step 1] Load data...")
df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
print(f"  Total samples: {len(df)}")


# ==================== Parse RMB-valued income and asset fields ====================
def parse_yuan(series):
    """Parse RMB-valued strings as floats and return NaN on failure."""

    def _parse(v):
        try:
            return float(re.sub(r"[^\d.\-]", "", str(v)))
        except Exception:
            return np.nan

    return series.apply(_parse)


print("\n[Step 2] Parse numeric fields...")
df["_个人年收入"] = parse_yuan(df["个人年收入"])
df["_家庭总收入"] = parse_yuan(df["家庭总收入"])
df["_家庭总资产"] = parse_yuan(df["家庭总资产"])
df["_家庭总负债"] = parse_yuan(df["家庭总负债"])
df["_家庭人数"] = pd.to_numeric(df["家庭人数"], errors="coerce")
df["_居民保最低缴费"] = pd.to_numeric(df["居民保缴费额下限（年）"], errors="coerce")
df["_职工保最低缴费"] = pd.to_numeric(df["职工保缴费额下限（年）"], errors="coerce")

# Household income and net assets per capita
df["_家庭人均收入"] = df.apply(
    lambda r: r["_家庭总收入"] / r["_家庭人数"] if r["_家庭人数"] > 0 else np.nan,
    axis=1,
)
df["_家庭净资产"] = df["_家庭总资产"] - df["_家庭总负债"]
df["_家庭人均净资产"] = df.apply(
    lambda r: r["_家庭净资产"] / r["_家庭人数"] if r["_家庭人数"] > 0 else np.nan,
    axis=1,
)

# ==================== Compute the three affordability denominators ====================
print("\n[Step 3] Calculate three burden measures...")


def burden(fee_col, denom_col):
    """Compute fee divided by denominator, returning NaN when invalid."""
    fee = df[fee_col]
    denom = df[denom_col]
    result = np.where(denom > 0, fee / denom, np.nan)
    return pd.Series(result, index=df.index)


# Resident pension burden
df["居民保_个人收入负担"] = burden("_居民保最低缴费", "_个人年收入")
df["居民保_家庭人均收入负担"] = burden("_居民保最低缴费", "_家庭人均收入")
df["居民保_家庭人均净资产负担"] = burden("_居民保最低缴费", "_家庭人均净资产")

# Employee pension burden
df["职工保_个人收入负担"] = burden("_职工保最低缴费", "_个人年收入")
df["职工保_家庭人均收入负担"] = burden("_职工保最低缴费", "_家庭人均收入")
df["职工保_家庭人均净资产负担"] = burden("_职工保最低缴费", "_家庭人均净资产")

# ==================== Representative burden: minimum positive ratio ====================
print("\n[Step 4] Calculate representative burden (minimum positive measure)...")


def min_positive(*cols):
    """Return the row-wise minimum positive value across columns."""
    mat = np.stack([df[c].values for c in cols], axis=1).astype(float)
    mat[mat <= 0] = np.nan
    with np.errstate(all="ignore"):
        result = np.nanmin(mat, axis=1)
    result[np.isnan(result).all(axis=0) if mat.ndim > 1 else np.isnan(result)] = np.nan
    return pd.Series(result, index=df.index)


df["居民保负担"] = min_positive(
    "居民保_个人收入负担", "居民保_家庭人均收入负担", "居民保_家庭人均净资产负担"
)
df["职工保负担"] = min_positive(
    "职工保_个人收入负担", "职工保_家庭人均收入负担", "职工保_家庭人均净资产负担"
)

# ==================== Group definitions ====================
# Combine the 2018 participation decision with the 2019 account type.
mask_jumin = (df["2018年参保决策"] == "参保") & (df["2019年参保账户"] == "城乡居民养老保险")
mask_zhigong = (df["2018年参保决策"] == "参保") & (df["2019年参保账户"] == "城镇职工养老保险")
mask_no = df["2018年参保决策"] == "不参保"

grp_jumin = df[mask_jumin]
grp_zhigong = df[mask_zhigong]
grp_no = df[mask_no]

print(f"\n  Resident pension participants: {len(grp_jumin)}")
print(f"  Employee pension participants: {len(grp_zhigong)}")
print(f"  Nonparticipants: {len(grp_no)}")

# ==================== Descriptive statistics ====================
print("\n[Step 5] Generate descriptive statistics...")


def describe_burden(series, label):
    s = series.dropna()
    # Restrict descriptive summaries to [0, 5] to remove extreme outliers.
    s_trim = s[(s > 0) & (s <= 5)]
    lines = [
        f"  {label}",
        f"    Valid values: {len(s)}  Within (0, 5]: {len(s_trim)}",
        f"    Median: {s_trim.median():.4f}  Mean: {s_trim.mean():.4f}",
        f"    25%: {s_trim.quantile(0.25):.4f}  75%: {s_trim.quantile(0.75):.4f}",
        f"    Minimum: {s_trim.min():.4f}  Maximum: {s_trim.max():.4f}",
    ]
    return "\n".join(lines)


summary_parts = [
    "=" * 65,
    "Pension Contribution Burden Analysis Summary",
    "=" * 65,
    "",
    "[Resident pension participants in 2018: resident pension burden]",
    describe_burden(grp_jumin["居民保负担"], "Resident pension burden"),
    "",
    "[Employee pension participants in 2018: employee pension burden]",
    describe_burden(grp_zhigong["职工保负担"], "Employee pension burden"),
    "",
    "[Nonparticipants in 2018: resident pension burden]",
    describe_burden(grp_no["居民保负担"], "Resident pension burden among nonparticipants"),
    "",
    "Note: Representative burden is the minimum positive value across personal income, per-capita household income, and per-capita household net-asset burdens.",
    "      Statistics are trimmed to (0, 5] to exclude extreme outliers.",
]
summary_text = "\n".join(summary_parts)
print(summary_text)

summary_path = OUTPUT_DIR / "burden_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)
print(f"\n✓ Summary saved: {summary_path}")

# ==================== Plotting ====================
print("\n[Step 6] Plot burden distributions...")


def clip_for_plot(series, upper=2.0):
    """Restrict a series to the positive plotting interval."""
    s = series.dropna()
    return s[(s > 0) & (s <= upper)]


# Color configuration
COLOR_JUMIN = "#2196F3"  # Blue
COLOR_ZHIGONG = "#FF5722"  # Orange-red
COLOR_NO = "#4CAF50"  # Green

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Pension Contribution Burden Distribution (representative burden = minimum positive measure)",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

UPPER = 2.0  # Upper clipping bound and x-axis maximum
BINS = 40

panels = [
    (
        axes[0],
        clip_for_plot(grp_jumin["居民保负担"], UPPER),
        COLOR_JUMIN,
        f"Resident pension participants (n={len(grp_jumin)})\nResident pension burden",
        "Resident pension burden (minimum contribution / capacity base)",
    ),
    (
        axes[1],
        clip_for_plot(grp_zhigong["职工保负担"], UPPER),
        COLOR_ZHIGONG,
        f"Employee pension participants (n={len(grp_zhigong)})\nEmployee pension burden",
        "Employee pension burden (minimum contribution / capacity base)",
    ),
    (
        axes[2],
        clip_for_plot(grp_no["居民保负担"], UPPER),
        COLOR_NO,
        f"Nonparticipants (n={len(grp_no)})\nResident pension burden",
        "Resident pension burden (minimum contribution / capacity base)",
    ),
]

for ax, data, color, title, xlabel in panels:
    ax.hist(data, bins=BINS, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
    # Median line
    med = data.median()
    ax.axvline(
        med, color="black", linestyle="--", linewidth=1.4, label=f"Median={med:.3f}"
    )
    # Mean line
    mean = data.mean()
    ax.axvline(
        mean, color="dimgray", linestyle=":", linewidth=1.4, label=f"Mean={mean:.3f}"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel("Sample count", fontsize=9.5)
    ax.set_xlim(0, UPPER)
    ax.legend(fontsize=8.5)
    ax.text(
        0.97,
        0.95,
        f"Valid samples: {len(data)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555",
    )

plt.tight_layout()
fig_path_png = OUTPUT_DIR / "burden_distribution.png"
fig_path_svg = OUTPUT_DIR / "burden_distribution.svg"
plt.savefig(fig_path_png, dpi=180, bbox_inches="tight")
plt.savefig(fig_path_svg, format='svg', bbox_inches="tight")
plt.close()
print(f"✓ Burden distribution plot saved: {fig_path_png}")
print(f"✓ SVG saved: {fig_path_svg}")

# ==================== Save wide-format data ====================
print("\n[Step 7] Save wide-format data...")

keep_cols = list(df.columns[:39]) + [
    "居民保_个人收入负担",
    "居民保_家庭人均收入负担",
    "居民保_家庭人均净资产负担",
    "职工保_个人收入负担",
    "职工保_家庭人均收入负担",
    "职工保_家庭人均净资产负担",
    "居民保负担",
    "职工保负担",
]
out_df = df[[c for c in keep_cols if c in df.columns]]
csv_path = BURDEN_DATA_FILE
out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✓ Wide-format data saved: {csv_path} ({len(out_df)} rows × {len(out_df.columns)} columns)")

# ==================== Completion summary ====================
print("\n" + "=" * 70)
print("Analysis complete. Output files:")
print(f"  1. Statistical summary: {summary_path}")
print(f"  2. Distribution plot: {fig_path}")
print(f"  3. Wide-format data: {csv_path}")
print("=" * 70)
