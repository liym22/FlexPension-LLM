#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY_DIR = ROOT / "outputs" / "bootstrap_evaluation_binary"
DEFAULT_MACRO_DIR = ROOT / "outputs" / "bootstrap_evaluation_macro"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "bootstrap_type_f1_mode_comparison"

METRIC_FIELDS = [
    "dataset",
    "model",
    "parse_success_rate",
    "action_f1",
    "type_f1_binary",
    "type_f1_macro",
    "type_f1_delta_macro_minus_binary",
    "composite_f1_binary",
    "composite_f1_macro",
    "composite_f1_delta_macro_minus_binary",
]

EXTERNAL_FIELDS = [
    "dataset",
    "model",
    "n_datasets",
    "datasets",
    "parse_success_rate",
    "action_f1",
    "type_f1_binary",
    "type_f1_macro",
    "type_f1_delta_macro_minus_binary",
    "composite_f1_binary",
    "composite_f1_macro",
    "composite_f1_delta_macro_minus_binary",
]

SELECTED_MODELS = {
    "flexpension",
    "correct_only",
    "claude_sonnet_4_5",
    "claude_sonnet_4_6",
    "claude_opus_4_6",
    "gemini_3_1_pro_preview",
    "qwen3_7_plus",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def join_metric_rows(
    binary_rows: list[dict[str, str]], macro_rows: list[dict[str, str]], *, include_external_meta: bool = False
) -> list[dict[str, object]]:
    macro_by_key = {(row["dataset"], row["model"]): row for row in macro_rows}
    joined: list[dict[str, object]] = []
    for binary in binary_rows:
        key = (binary["dataset"], binary["model"])
        macro = macro_by_key.get(key)
        if macro is None:
            raise ValueError(f"Missing macro row for dataset={key[0]} model={key[1]}")

        type_binary = as_float(binary, "type_f1")
        type_macro = as_float(macro, "type_f1")
        composite_binary = as_float(binary, "composite_f1")
        composite_macro = as_float(macro, "composite_f1")
        row: dict[str, object] = {
            "dataset": binary["dataset"],
            "model": binary["model"],
            "parse_success_rate": as_float(binary, "parse_success_rate"),
            "action_f1": as_float(binary, "action_f1"),
            "type_f1_binary": type_binary,
            "type_f1_macro": type_macro,
            "type_f1_delta_macro_minus_binary": type_macro - type_binary,
            "composite_f1_binary": composite_binary,
            "composite_f1_macro": composite_macro,
            "composite_f1_delta_macro_minus_binary": composite_macro - composite_binary,
        }
        if include_external_meta:
            row["n_datasets"] = int(binary["n_datasets"])
            row["datasets"] = binary["datasets"]
        joined.append(row)
    return sorted(joined, key=lambda row: (str(row["dataset"]), -float(row["composite_f1_macro"]), str(row["model"])))


def fmt(value: object) -> str:
    return f"{float(value):.4f}"


def write_readme(out_dir: Path, all_rows: list[dict[str, object]], external_rows: list[dict[str, object]]) -> None:
    by_key = {(row["dataset"], row["model"]): row for row in all_rows}
    external_by_model = {row["model"]: row for row in external_rows}
    flex_blind = by_key[("blind", "flexpension")]
    flex_external = external_by_model["flexpension"]
    parse_all_ok = all(float(row["parse_success_rate"]) == 1.0 for row in all_rows)

    lines = [
        "# Binary vs Macro Type F1 Comparison",
        "",
        f"Generated: {date.today().isoformat()}.",
        "",
        "Both result sets use `B=10000, seed=42`, identical samples and models, the same CLDS row-level repair, and the same CFPS teacher latest-per-ID repair. Only the `type_f1` definition differs.",
        "",
        "- Binary outputs: `outputs/bootstrap_evaluation_binary/`",
        "- Macro outputs: `outputs/bootstrap_evaluation_macro/`",
        "- All-model comparison: `all_model_metrics_binary_vs_macro.csv`",
        "- Four-survey external-average comparison: `external_average_binary_vs_macro.csv`",
        "- Selected-model comparison: `selected_model_metrics_binary_vs_macro.csv`",
        "",
        "## Key Results",
        "",
        "| Type F1 mode | FlexPension blind Composite | FlexPension external avg Composite |",
        "| --- | ---: | ---: |",
        f"| binary Type F1 | {fmt(flex_blind['composite_f1_binary'])} | {fmt(flex_external['composite_f1_binary'])} |",
        f"| macro Type F1 | {fmt(flex_blind['composite_f1_macro'])} | {fmt(flex_external['composite_f1_macro'])} |",
        "",
        "Top five systems by macro-mode average across the four external surveys:",
        "",
        "| rank | model | binary Composite | macro Composite | delta |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(sorted(external_rows, key=lambda item: float(item["composite_f1_macro"]), reverse=True)[:5], start=1):
        lines.append(
            f"| {rank} | {row['model']} | {fmt(row['composite_f1_binary'])} | "
            f"{fmt(row['composite_f1_macro'])} | {float(row['composite_f1_delta_macro_minus_binary']):+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Across all {len(all_rows)} `model x dataset` combinations, parse success rates "
            f"{'are 1.0 in both outputs' if parse_all_ok else 'include non-1.0 values; inspect the CSV'}.",
            "",
        ]
    )
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(binary_dir: Path, macro_dir: Path, out_dir: Path) -> None:
    all_rows = join_metric_rows(
        read_csv(binary_dir / "all_model_metrics.csv"),
        read_csv(macro_dir / "all_model_metrics.csv"),
    )
    external_rows = join_metric_rows(
        read_csv(binary_dir / "external_macro_average_metrics.csv"),
        read_csv(macro_dir / "external_macro_average_metrics.csv"),
        include_external_meta=True,
    )
    selected_rows = [row for row in all_rows if row["model"] in SELECTED_MODELS]

    write_csv(out_dir / "all_model_metrics_binary_vs_macro.csv", all_rows, METRIC_FIELDS)
    write_csv(out_dir / "external_average_binary_vs_macro.csv", external_rows, EXTERNAL_FIELDS)
    write_csv(out_dir / "selected_model_metrics_binary_vs_macro.csv", selected_rows, METRIC_FIELDS)
    write_readme(out_dir, all_rows, external_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary-dir", type=Path, default=DEFAULT_BINARY_DIR)
    parser.add_argument("--macro-dir", type=Path, default=DEFAULT_MACRO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    write_outputs(args.binary_dir, args.macro_dir, args.output_dir)
    print(f"Wrote binary/macro comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
