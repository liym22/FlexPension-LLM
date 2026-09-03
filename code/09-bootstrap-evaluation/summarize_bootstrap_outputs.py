#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


METRIC_FIELDS = [
    "parse_success_rate",
    "action_f1",
    "type_f1",
    "composite_f1",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def collect(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metric_rows: list[dict[str, str]] = []
    delta_rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*_model_metrics.csv")):
        if path.parent == root:
            continue
        metric_rows.extend(read_csv(path))
    for path in sorted(root.rglob("*_bootstrap_deltas.csv")):
        if path.parent == root:
            continue
        delta_rows.extend(read_csv(path))
    return metric_rows, delta_rows


def read_bootstrap_config(root: Path) -> tuple[str, str, str]:
    for path in sorted(root.rglob("*_bootstrap_summary.json")):
        if path.parent == root:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            str(payload.get("bootstrap", "")),
            str(payload.get("seed", "")),
            str(payload.get("type_f1_mode", "")),
        )
    return "", "", ""


def external_macro_rows(metric_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        dataset = row["dataset"]
        if dataset == "blind":
            continue
        grouped[row["model"]].append(row)

    rows: list[dict[str, object]] = []
    for model, model_rows in sorted(grouped.items()):
        datasets = sorted(row["dataset"] for row in model_rows)
        out: dict[str, object] = {
            "dataset": "external_macro_average",
            "model": model,
            "n_datasets": len(model_rows),
            "datasets": ";".join(datasets),
        }
        for field in METRIC_FIELDS:
            values = [as_float(row, field) for row in model_rows]
            out[field] = sum(values) / len(values)
        rows.append(out)
    rows.sort(key=lambda row: float(row["composite_f1"]), reverse=True)
    return rows


def best_rows(metric_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        by_dataset[row["dataset"]].append(row)

    rows: list[dict[str, object]] = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        ordered = sorted(dataset_rows, key=lambda row: as_float(row, "composite_f1"), reverse=True)
        for rank, row in enumerate(ordered, start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "rank": rank,
                    "model": row["model"],
                    "action_f1": row["action_f1"],
                    "type_f1": row["type_f1"],
                    "composite_f1": row["composite_f1"],
                    "parse_success_rate": row["parse_success_rate"],
                }
            )
    return rows


def write_markdown_summary(
    path: Path,
    metric_rows: list[dict[str, str]],
    macro_rows: list[dict[str, object]],
    delta_rows: list[dict[str, str]],
    bootstrap: str,
    seed: str,
    type_f1_mode: str,
) -> None:
    lines: list[str] = []
    lines.append("# Bootstrap Evaluation Summary")
    lines.append("")
    lines.append(
        f"Configuration: B={bootstrap}, seed={seed}, "
        f"type_f1_mode={type_f1_mode or 'unknown'}. "
        "Deltas are `flexpension - baseline`."
    )
    lines.append("")

    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        by_dataset[row["dataset"]].append(row)

    for dataset in sorted(by_dataset):
        rows = sorted(by_dataset[dataset], key=lambda row: as_float(row, "composite_f1"), reverse=True)
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| rank | model | action_f1 | type_f1 | composite_f1 | parse_success |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {row['model']} | {float(row['action_f1']):.4f} | "
                f"{float(row['type_f1']):.4f} | {float(row['composite_f1']):.4f} | "
                f"{float(row['parse_success_rate']):.4f} |"
            )
        lines.append("")

    lines.append("## external_macro_average")
    lines.append("")
    lines.append("| rank | model | action_f1 | type_f1 | composite_f1 | n_datasets |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for rank, row in enumerate(macro_rows, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {float(row['action_f1']):.4f} | "
            f"{float(row['type_f1']):.4f} | {float(row['composite_f1']):.4f} | "
            f"{row['n_datasets']} |"
        )
    lines.append("")

    flex_deltas = [
        row
        for row in delta_rows
        if row["model_a"] == "flexpension" and row["metric"] == "composite_f1"
    ]
    lines.append("## flexpension_delta_composite_f1")
    lines.append("")
    lines.append("| dataset | baseline | delta | 95% CI |")
    lines.append("| --- | --- | ---: | --- |")
    for row in sorted(flex_deltas, key=lambda row: (row["dataset"], row["model_b"])):
        lines.append(
            f"| {row['dataset']} | {row['model_b']} | {float(row['observed_delta']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/bootstrap_evaluation")
    metric_rows, delta_rows = collect(root)
    if not metric_rows:
        raise SystemExit(f"No metric rows found under {root}")
    bootstrap, seed, type_f1_mode = read_bootstrap_config(root)

    write_csv(root / "all_model_metrics.csv", metric_rows)
    write_csv(root / "all_bootstrap_deltas.csv", delta_rows)

    macro_rows = external_macro_rows(metric_rows)
    write_csv(
        root / "external_macro_average_metrics.csv",
        macro_rows,
        ["dataset", "model", "n_datasets", "datasets", *METRIC_FIELDS],
    )
    write_csv(
        root / "ranked_model_metrics.csv",
        best_rows(metric_rows),
        ["dataset", "rank", "model", "action_f1", "type_f1", "composite_f1", "parse_success_rate"],
    )
    write_markdown_summary(
        root / "bootstrap_summary.md",
        metric_rows,
        macro_rows,
        delta_rows,
        bootstrap,
        seed,
        type_f1_mode,
    )
    print(f"Wrote summaries to {root}")


if __name__ == "__main__":
    main()
