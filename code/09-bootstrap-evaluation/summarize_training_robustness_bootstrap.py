from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


GROUPS = ("data_scale", "checkpoint_dynamics", "temperature")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        path = root / group / f"{group}_model_metrics.csv"
        for row in read_csv(path):
            rows.append({"group": group, **row})
    return rows


def delta_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        path = root / group / f"{group}_bootstrap_deltas.csv"
        for row in read_csv(path):
            rows.append({"group": group, **row})
    return rows


def sort_key(row: dict[str, Any]) -> tuple[int, float]:
    group = str(row["group"])
    model = str(row["model"])
    if group == "data_scale":
        return (0, float(model.replace("data", "")))
    if group == "checkpoint_dynamics":
        return (1, float(model.replace("ckpt", "")))
    if group == "temperature":
        return (2, float(model.replace("t", "").replace("_", ".")))
    return (9, 0.0)


def format_metric_table(rows: list[dict[str, Any]], group: str) -> list[str]:
    selected = [row for row in rows if row["group"] == group]
    selected.sort(key=sort_key)
    lines = [
        f"## {group}",
        "",
        "| model | action_f1 | type_f1 | composite_f1 | parse_success_rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            "| {model} | {action:.4f} | {type_f1:.4f} | {composite:.4f} | {parse_rate:.4f} |".format(
                model=row["model"],
                action=float(row["action_f1"]),
                type_f1=float(row["type_f1"]),
                composite=float(row["composite_f1"]),
                parse_rate=float(row["parse_success_rate"]),
            )
        )
    lines.append("")
    return lines


def format_delta_table(rows: list[dict[str, Any]], group: str) -> list[str]:
    selected = [
        row
        for row in rows
        if row["group"] == group and row["metric"] == "composite_f1"
    ]
    lines = [
        f"## {group} Composite F1 paired deltas",
        "",
        "| comparison | delta | 95% CI | p(delta <= 0) |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in selected:
        lines.append(
            "| {a} - {b} | {delta:.4f} | [{low:.4f}, {high:.4f}] | {p:.4f} |".format(
                a=row["model_a"],
                b=row["model_b"],
                delta=float(row["observed_delta"]),
                low=float(row["ci_low"]),
                high=float(row["ci_high"]),
                p=float(row["p_delta_le_0"]),
            )
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    metrics = metric_rows(args.root)
    deltas = delta_rows(args.root)
    metrics.sort(key=sort_key)

    write_csv(args.root / "combined_model_metrics.csv", metrics)
    write_csv(args.root / "combined_bootstrap_deltas.csv", deltas)

    lines = [
        "# Training and Robustness Bootstrap Summary",
        "",
        f"Configuration: binary Type F1, B={args.bootstrap}, seed={args.seed}.",
        "",
        "These results support Figure 3 (training efficiency and robustness) and do not replace the main benchmark table.",
        "",
    ]
    for group in GROUPS:
        lines.extend(format_metric_table(metrics, group))
    for group in GROUPS:
        lines.extend(format_delta_table(deltas, group))

    (args.root / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
