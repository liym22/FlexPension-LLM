from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_metrics(metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["metrics_file"] = str(path)
        rows.append(data)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model_short_name",
        "model_id",
        "dataset",
        "n_total",
        "api_success_rate",
        "parse_success_rate",
        "action_f1",
        "type_f1",
        "composite_f1",
        "total_cost_rmb",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in cols})


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model_short_name",
        "dataset",
        "n_total",
        "api_success_rate",
        "parse_success_rate",
        "action_f1",
        "type_f1",
        "composite_f1",
        "total_cost_rmb",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    rows = sorted(load_metrics(args.metrics_dir), key=lambda row: row.get("composite_f1", 0.0), reverse=True)
    write_csv(rows, args.output_prefix.with_suffix(".csv"))
    write_markdown(rows, args.output_prefix.with_suffix(".md"))
    print(f"rows={len(rows)}")
    print(f"csv={args.output_prefix.with_suffix('.csv')}")
    print(f"md={args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()

