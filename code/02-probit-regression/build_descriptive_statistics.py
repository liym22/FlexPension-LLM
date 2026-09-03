#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 CHFS2019 主分析样本的描述性统计表。

输出：
- probit_reg/output/descriptive_statistics.csv
- probit_reg/output/descriptive_statistics.xlsx
- probit_reg/output/descriptive_statistics.tex
- probit_reg/output/descriptive_statistics_variable_map.csv
- probit_reg/output/descriptive_statistics_refined.csv
- probit_reg/output/descriptive_statistics_refined.xlsx
- probit_reg/output/descriptive_statistics_refined.tex
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_CSV = BASE_DIR / "data" / "processed" / "all_samples_with_policy.csv"
BURDEN_CSV = BASE_DIR / "code" / "04-prompt-engineering" / "calculate" / "burden_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "probit" / "descriptive_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONEY_VARS = {"个人年收入", "家庭总收入", "家庭总消费", "家庭总资产", "家庭总负债"}
RATE_VARS = {
    "女性",
    "初中及以下",
    "高中",
    "大专及以上",
    "农业户口",
    "是否流动",
    "是否参保",
    "参保类型：城乡居民养老保险",
    "参保类型：城镇职工养老保险",
    "是否断缴",
    "居民保负担率",
    "职工保负担率",
    "家庭养老金依赖度",
}


def parse_amount(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "不清楚", "未知", "nan", "None"}:
        return np.nan
    if text == "未领取":
        return 0.0
    cleaned = re.sub(r"[^\d.\-]", "", text)
    if cleaned == "":
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def parse_years(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "不清楚", "nan", "None"}:
        return np.nan
    cleaned = re.sub(r"[^\d.\-]", "", text)
    if cleaned == "":
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def parse_gap_binary(value):
    text = str(value).strip()
    if text.startswith("断缴"):
        return 1.0
    if text in {"不存在", "未断缴"}:
        return 0.0
    return np.nan


def compute_dependency(monthly_pension, total_income):
    if pd.isna(monthly_pension) or pd.isna(total_income) or total_income <= 0:
        return np.nan
    if monthly_pension <= 0:
        return 0.0
    return monthly_pension * 12 / total_income


def finalize_health_measure(series: pd.Series) -> tuple[pd.Series, str]:
    valid = set(series.dropna().astype(str).unique())
    ordered_map = {"非常不好": 1, "不好": 2, "一般": 3, "好": 4, "非常好": 5}
    if valid and valid.issubset(set(ordered_map) | {"不清楚"}):
        return series.map(ordered_map), "健康状况（1=非常不好, 5=非常好）"
    binary = series.map(
        lambda x: 1.0
        if str(x).strip() in {"好", "非常好"}
        else (0.0 if str(x).strip() in {"一般", "不好", "非常不好"} else np.nan)
    )
    return binary, "健康较好"


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    lower_bound = valid.quantile(lower)
    upper_bound = valid.quantile(upper)
    return numeric.clip(lower=lower_bound, upper=upper_bound)


def build_analysis_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    burden_df = pd.read_csv(BURDEN_CSV, encoding="utf-8-sig")
    burden_keep = burden_df[
        [
            "家庭ID",
            "个人ID",
            "居民保_个人收入负担",
            "居民保_家庭人均收入负担",
            "居民保_家庭人均净资产负担",
            "职工保_个人收入负担",
            "职工保_家庭人均收入负担",
            "职工保_家庭人均净资产负担",
        ]
    ].copy()
    df = df.merge(burden_keep, on=["家庭ID", "个人ID"], how="left", suffixes=("", "_burden_src"))

    analysis = pd.DataFrame(index=df.index)

    analysis["年龄"] = pd.to_numeric(df["年龄"], errors="coerce")
    analysis["女性"] = df["性别"].map({"女": 1, "男": 0}).astype(float)

    edu = df["文化程度"].fillna("不清楚")
    analysis["初中及以下"] = edu.isin(["未上学", "小学", "初中", "不清楚"]).astype(float)
    analysis["高中"] = edu.isin(["高中", "中专/职高"]).astype(float)
    analysis["大专及以上"] = edu.isin(["大专", "本科", "硕士", "博士"]).astype(float)

    health_series, health_label = finalize_health_measure(df["健康状况"])
    analysis[health_label] = health_series.astype(float)

    analysis["农业户口"] = (df["户口性质"] == "农业").astype(float)
    analysis["是否流动"] = df["是否流动"].map({"是": 1, "否": 0}).astype(float)

    analysis["个人年收入"] = df["个人年收入"].apply(parse_amount)
    analysis["家庭总收入"] = df["家庭总收入"].apply(parse_amount)
    analysis["家庭总消费"] = df["家庭总消费"].apply(parse_amount)
    analysis["家庭总资产"] = df["家庭总资产"].apply(parse_amount)
    analysis["家庭总负债"] = df["家庭总负债"].apply(parse_amount)

    analysis["家庭人数"] = pd.to_numeric(df["家庭人数"], errors="coerce")
    analysis["子女数"] = pd.to_numeric(df["子女数"], errors="coerce")
    analysis["老人数"] = pd.to_numeric(df["老人数"], errors="coerce")

    analysis["是否参保"] = df["2018年参保决策"].map({"参保": 1, "不参保": 0}).astype(float)
# Compute scheme shares among 2018 participants so the two scheme shares sum to participation.
    analysis["参保类型：城乡居民养老保险"] = (
        (df["2018年参保决策"] == "参保") & (df["2019年参保账户"] == "城乡居民养老保险")
    ).astype(float)
    analysis["参保类型：城镇职工养老保险"] = (
        (df["2018年参保决策"] == "参保") & (df["2019年参保账户"] == "城镇职工养老保险")
    ).astype(float)

    analysis["历史参保年限"] = df["累计缴纳年限"].apply(parse_years)
    analysis["是否断缴"] = df["是否存在断缴"].apply(parse_gap_binary)
    analysis["家庭参保人数"] = pd.to_numeric(df["家庭参保人数"], errors="coerce")

    monthly_pension = df["家庭月均养老金"].apply(parse_amount)
    total_income = analysis["家庭总收入"]
    analysis["家庭养老金依赖度"] = pd.Series(
        [compute_dependency(p, inc) for p, inc in zip(monthly_pension, total_income)],
        index=df.index,
        dtype=float,
    )

    def best_burden(prefix: str) -> pd.Series:
        candidates = pd.concat(
            [
                pd.to_numeric(df[f"{prefix}_个人收入负担"], errors="coerce"),
                pd.to_numeric(df[f"{prefix}_家庭人均收入负担"], errors="coerce"),
                pd.to_numeric(df[f"{prefix}_家庭人均净资产负担"], errors="coerce"),
            ],
            axis=1,
        )
        positive = candidates.where(candidates > 0)
        return positive.min(axis=1, skipna=True)

    analysis["居民保负担率"] = best_burden("居民保")
    analysis["职工保负担率"] = best_burden("职工保")

    variable_map = pd.DataFrame(
        [
            ("年龄", "年龄", "直接使用主样本字段"),
            ("女性", "性别", "女=1, 男=0"),
            ("初中及以下", "文化程度", "未上学/小学/初中/不清楚 = 1"),
            ("高中", "文化程度", "高中/中专职高 = 1"),
            ("大专及以上", "文化程度", "大专/本科/硕士/博士 = 1"),
            (health_label, "健康状况", "先检查编码；当前采用稳定有序映射或后备二元变量"),
            ("农业户口", "户口性质", "农业=1"),
            ("是否流动", "是否流动", "是=1, 否=0"),
            ("个人年收入", "个人年收入", "去除‘元’并转数值"),
            ("家庭总收入", "家庭总收入", "去除‘元’并转数值"),
            ("家庭总消费", "家庭总消费", "去除‘元’并转数值"),
            ("家庭总资产", "家庭总资产", "去除‘元’并转数值"),
            ("家庭总负债", "家庭总负债", "去除‘元’并转数值"),
            ("家庭人数", "家庭人数", "直接转数值"),
            ("子女数", "子女数", "直接转数值"),
            ("老人数", "老人数", "直接转数值"),
            ("是否参保", "2018年参保决策", "参保=1, 不参保=0"),
            ("参保类型：城乡居民养老保险", "2018年参保决策 + 2019年参保账户", "2018决策参保 且 2019账户为城乡居民=1"),
            ("参保类型：城镇职工养老保险", "2018年参保决策 + 2019年参保账户", "2018决策参保 且 2019账户为城镇职工=1"),
            ("历史参保年限", "累计缴纳年限", "去除‘年’并转数值"),
            ("是否断缴", "是否存在断缴", "断缴X年=1, 不存在=0, 不清楚缺失"),
            ("家庭参保人数", "家庭参保人数", "直接转数值"),
            ("居民保负担率", "burden_data 三类居民保负担", "取正值中的最小值作为代表负担率"),
            ("职工保负担率", "burden_data 三类职工保负担", "取正值中的最小值作为代表负担率"),
        ],
        columns=["变量名", "源字段", "构造说明"],
    )

    return analysis, variable_map


def summarize_frame(analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in analysis.columns:
        series = pd.to_numeric(analysis[col], errors="coerce")
        rows.append(
            {
                "变量名": col,
                "N": int(series.notna().sum()),
                "均值": series.mean(),
                "标准差": series.std(),
                "最小值": series.min(),
                "最大值": series.max(),
            }
        )
    summary = pd.DataFrame(rows)
    for numeric_col in ["均值", "标准差", "最小值", "最大值"]:
        summary[numeric_col] = summary[numeric_col].round(4)
    return summary


def build_refined_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in analysis.columns:
        series = pd.to_numeric(analysis[col], errors="coerce")
        unit = "原值"
        treatment = "原始值"

        if col in MONEY_VARS:
            series = winsorize_series(series)
            treatment = "1%缩尾"
            unit = "元"
        elif col in RATE_VARS:
            series = series * 100
            unit = "%"
            treatment = "比例转百分比"
            if col in {"居民保负担率", "职工保负担率", "家庭养老金依赖度"}:
                series = winsorize_series(series)
                treatment = "比例转百分比后1%缩尾"
        elif col in {"年龄", "历史参保年限", "家庭人数", "子女数", "老人数"}:
            unit = "原值"
            treatment = "原始值"
        elif "健康状况" in col or col == "健康较好":
            unit = "得分"
            treatment = "原始值"

        rows.append(
            {
                "变量名": col,
                "N": int(series.notna().sum()),
                "均值": round(series.mean(), 2),
                "标准差": round(series.std(), 2),
                "最小值": round(series.min(), 2),
                "最大值": round(series.max(), 2),
                "展示单位": unit,
                "处理方式": treatment,
            }
        )

    return pd.DataFrame(rows)


def render_latex_table(latex_df: pd.DataFrame, refined: bool = False) -> str:
    caption = "CHFS 2019主分析样本描述性统计（精修版）" if refined else "CHFS 2019主分析样本描述性统计"
    label = "tab:descriptive_statistics_refined" if refined else "tab:descriptive_statistics"
    note = (
        "注：金额变量按1%和99%分位缩尾；比例变量以百分数展示，负担率与养老金依赖度在百分比口径下再做1%缩尾。"
        if refined
        else None
    )
    header = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize" if refined else "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "变量名 & N & 均值 & 标准差 & 最小值 & 最大值 \\\\",
        "\\hline",
    ]
    body = []
    for _, row in latex_df.iterrows():
        body.append(
            f"{row['变量名']} & {row['N']} & {row['均值']} & {row['标准差']} & {row['最小值']} & {row['最大值']} \\\\",
        )
    footer = [
        "\\hline",
        "\\end{tabular}",
    ]
    if note:
        footer.append(
            f"\\begin{{minipage}}{{0.92\\textwidth}}\\vspace{{2pt}}\\footnotesize {note}\\end{{minipage}}"
        )
    footer.append("\\end{table}")
    return "\n".join(header + body + footer) + "\n"


def write_outputs(summary: pd.DataFrame, variable_map: pd.DataFrame, refined_summary: pd.DataFrame) -> None:
    csv_path = OUTPUT_DIR / "descriptive_statistics.csv"
    xlsx_path = OUTPUT_DIR / "descriptive_statistics.xlsx"
    tex_path = OUTPUT_DIR / "descriptive_statistics.tex"
    map_path = OUTPUT_DIR / "descriptive_statistics_variable_map.csv"
    refined_csv_path = OUTPUT_DIR / "descriptive_statistics_refined.csv"
    refined_xlsx_path = OUTPUT_DIR / "descriptive_statistics_refined.xlsx"
    refined_tex_path = OUTPUT_DIR / "descriptive_statistics_refined.tex"

    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    variable_map.to_csv(map_path, index=False, encoding="utf-8-sig")
    refined_summary.to_csv(refined_csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="descriptive_stats")
        variable_map.to_excel(writer, index=False, sheet_name="variable_map")

    with pd.ExcelWriter(refined_xlsx_path, engine="openpyxl") as writer:
        refined_summary.to_excel(writer, index=False, sheet_name="descriptive_stats_refined")
        variable_map.to_excel(writer, index=False, sheet_name="variable_map")

    latex_df = summary.copy()
    latex_df["N"] = latex_df["N"].map(lambda x: f"{int(x):,}")
    for col in ["均值", "标准差", "最小值", "最大值"]:
        latex_df[col] = latex_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    refined_latex_df = refined_summary[["变量名", "N", "均值", "标准差", "最小值", "最大值"]].copy()
    refined_latex_df["N"] = refined_latex_df["N"].map(lambda x: f"{int(x):,}")
    for col in ["均值", "标准差", "最小值", "最大值"]:
        refined_latex_df[col] = refined_latex_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    tex_path.write_text(render_latex_table(latex_df), encoding="utf-8")
    refined_tex_path.write_text(render_latex_table(refined_latex_df, refined=True), encoding="utf-8")


def main() -> None:
    analysis, variable_map = build_analysis_frame()
    summary = summarize_frame(analysis)
    refined_summary = build_refined_summary(analysis)
    write_outputs(summary, variable_map, refined_summary)

    print("=" * 70)
    print("Descriptive-statistics tables generated")
    print("=" * 70)
    print(f"Sample size: {len(analysis)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(summary.to_string(index=False))
    print("-" * 70)
    print("Refined descriptive-statistics tables")
    print(refined_summary.to_string(index=False))


if __name__ == "__main__":
    main()
