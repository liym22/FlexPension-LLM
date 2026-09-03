#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BOOTSTRAP_MODULE = SCRIPT_DIR / "bootstrap_evaluation.py"
DATASETS: dict[str, dict[str, str]] = {}

METRICS = ("action_f1", "type_f1", "composite_f1")


def load_bootstrap_module():
    spec = importlib.util.spec_from_file_location("bootstrap_evaluation", BOOTSTRAP_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def interval_summary(values: np.ndarray, *, observed: float) -> dict[str, float]:
    return {
        "observed": float(observed),
        "ci_low": percentile(values, 0.025),
        "ci_high": percentile(values, 0.975),
        "bootstrap_mean": float(np.mean(values)),
    }


def average_vectors(vectors_by_dataset: dict[str, np.ndarray]) -> np.ndarray:
    if not vectors_by_dataset:
        raise ValueError("No dataset vectors to average")
    lengths = {len(values) for values in vectors_by_dataset.values()}
    if len(lengths) != 1:
        raise ValueError(f"Dataset bootstrap vectors have inconsistent lengths: {sorted(lengths)}")
    return np.mean(np.stack([vectors_by_dataset[name] for name in sorted(vectors_by_dataset)]), axis=0)


def average_delta_vectors(
    model_a_vectors: dict[str, np.ndarray], model_b_vectors: dict[str, np.ndarray]
) -> np.ndarray:
    dataset_names = set(model_a_vectors) & set(model_b_vectors)
    if dataset_names != set(model_a_vectors) or dataset_names != set(model_b_vectors):
        raise ValueError("Delta vectors require matching datasets")
    deltas = {name: model_a_vectors[name] - model_b_vectors[name] for name in dataset_names}
    return average_vectors(deltas)


def dataset_path(bootstrap_root: Path, dataset_name: str, key: str) -> Path:
    value = DATASETS[dataset_name][key]
    if isinstance(value, Path):
        return value
    return bootstrap_root / value


def load_dataset_model_specs(bootstrap_root: Path, dataset_name: str) -> list[dict[str, Any]]:
    summary_path = bootstrap_root / "external" / dataset_name / f"{dataset_name}_bootstrap_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return list(payload["models"])


def bootstrap_dataset_vectors(
    *,
    bootstrap_root: Path,
    dataset_name: str,
    type_f1_mode: str,
    b: int,
    seed: int,
    dataset_offset: int,
    module: Any,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
    prompts = dataset_path(bootstrap_root, dataset_name, "prompts")
    truth = dataset_path(bootstrap_root, dataset_name, "ground_truth")
    records = module.load_external_records(prompts, truth)
    labels = [record.terminal_label for record in records]
    indices = module.stratified_bootstrap_index_matrix(labels, b, seed + dataset_offset)

    vectors_by_model: dict[str, dict[str, np.ndarray]] = {}
    observed_by_model: dict[str, dict[str, float]] = {}
    for spec in load_dataset_model_specs(bootstrap_root, dataset_name):
        predictions = module.load_predictions(Path(spec["path"]), spec["kind"], row_level=True)
        contributions = module.prediction_contributions(records, predictions)
        matrix = module.contribution_matrix(contributions)
        counts = matrix[indices].sum(axis=1)
        vectors_by_model[spec["model"]] = {
            metric: module.metric_vector_from_count_matrix(counts, metric, type_f1_mode=type_f1_mode)
            for metric in METRICS
        }
        observed = module.metrics_from_contributions(contributions, type_f1_mode=type_f1_mode)
        observed_by_model[spec["model"]] = {metric: float(observed[metric]) for metric in METRICS}
    return vectors_by_model, observed_by_model


def run_external_average_bootstrap(
    *,
    bootstrap_root: Path,
    output_dir: Path,
    type_f1_mode: str,
    b: int,
    seed: int,
    focal_model: str = "flexpension",
) -> None:
    module = load_bootstrap_module()
    dataset_names = list(DATASETS)
    vectors: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    observed: dict[str, dict[str, dict[str, float]]] = {}
    for offset, dataset_name in enumerate(dataset_names):
        dataset_vectors, dataset_observed = bootstrap_dataset_vectors(
            bootstrap_root=bootstrap_root,
            dataset_name=dataset_name,
            type_f1_mode=type_f1_mode,
            b=b,
            seed=seed,
            dataset_offset=1009 * offset,
            module=module,
        )
        vectors[dataset_name] = dataset_vectors
        observed[dataset_name] = dataset_observed

    model_names = sorted(set.intersection(*(set(vectors[name]) for name in dataset_names)))
    model_rows: list[dict[str, Any]] = []
    model_average_vectors: dict[str, dict[str, np.ndarray]] = {model: {} for model in model_names}
    model_observed: dict[str, dict[str, float]] = {model: {} for model in model_names}
    for model in model_names:
        for metric in METRICS:
            per_dataset_vectors = {
                dataset: vectors[dataset][model][metric]
                for dataset in dataset_names
            }
            averaged = average_vectors(per_dataset_vectors)
            observed_average = sum(observed[dataset][model][metric] for dataset in dataset_names) / len(dataset_names)
            model_average_vectors[model][metric] = averaged
            model_observed[model][metric] = observed_average
            summary = interval_summary(averaged, observed=observed_average)
            model_rows.append(
                {
                    "type_f1_mode": type_f1_mode,
                    "dataset": "external_macro_average",
                    "model": model,
                    "metric": metric,
                    "n_datasets": len(dataset_names),
                    "datasets": ";".join(dataset_names),
                    "n_bootstrap": b,
                    "seed": seed,
                    **summary,
                }
            )

    delta_rows: list[dict[str, Any]] = []
    if focal_model not in model_average_vectors:
        raise ValueError(f"Missing focal model: {focal_model}")
    for baseline in model_names:
        if baseline == focal_model:
            continue
        for metric in METRICS:
            delta_vector = average_delta_vectors(
                {dataset: vectors[dataset][focal_model][metric] for dataset in dataset_names},
                {dataset: vectors[dataset][baseline][metric] for dataset in dataset_names},
            )
            observed_delta = model_observed[focal_model][metric] - model_observed[baseline][metric]
            summary = interval_summary(delta_vector, observed=observed_delta)
            delta_rows.append(
                {
                    "type_f1_mode": type_f1_mode,
                    "dataset": "external_macro_average",
                    "model_a": focal_model,
                    "model_b": baseline,
                    "metric": metric,
                    "n_datasets": len(dataset_names),
                    "datasets": ";".join(dataset_names),
                    "n_bootstrap": b,
                    "seed": seed,
                    "p_delta_le_0": float(np.mean(delta_vector <= 0)),
                    "p_delta_ge_0": float(np.mean(delta_vector >= 0)),
                    **summary,
                }
            )

    write_csv(
        output_dir / "external_average_model_cis.csv",
        model_rows,
        [
            "type_f1_mode",
            "dataset",
            "model",
            "metric",
            "n_datasets",
            "datasets",
            "n_bootstrap",
            "seed",
            "observed",
            "ci_low",
            "ci_high",
            "bootstrap_mean",
        ],
    )
    write_csv(
        output_dir / "external_average_bootstrap_deltas.csv",
        delta_rows,
        [
            "type_f1_mode",
            "dataset",
            "model_a",
            "model_b",
            "metric",
            "n_datasets",
            "datasets",
            "n_bootstrap",
            "seed",
            "observed",
            "ci_low",
            "ci_high",
            "bootstrap_mean",
            "p_delta_le_0",
            "p_delta_ge_0",
        ],
    )
    write_readme(output_dir, model_rows, delta_rows, type_f1_mode, b, seed)


def write_readme(
    output_dir: Path,
    model_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    type_f1_mode: str,
    b: int,
    seed: int,
) -> None:
    composite_rows = [row for row in model_rows if row["metric"] == "composite_f1"]
    composite_rows.sort(key=lambda row: float(row["observed"]), reverse=True)
    focal_rows = [
        row
        for row in delta_rows
        if row["model_a"] == "flexpension" and row["metric"] == "composite_f1"
    ]
    focal_rows.sort(key=lambda row: float(row["observed"]), reverse=True)
    selected_baselines = {
        "correct_only",
        "claude_opus_4_6",
        "gemini_3_1_pro_preview",
        "claude_sonnet_4_6",
        "claude_sonnet_4_5",
        "qwen3_7_plus",
        "qwen_zs",
    }

    lines = [
        "# External Average Bootstrap CI",
        "",
        f"配置：type_f1_mode={type_f1_mode}，B={b}，seed={seed}。",
        "",
        "方法：在每个 external 数据集内按真实终端标签分层 bootstrap，模型之间共享同一组重抽样索引；每次重抽后先计算各数据集指标，再对四个数据集等权平均。",
        "",
        "## Composite F1 model CI",
        "",
        "| rank | model | observed | 95% CI |",
        "| ---: | --- | ---: | --- |",
    ]
    for rank, row in enumerate(composite_rows, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {float(row['observed']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |"
        )
    lines.extend(
        [
            "",
            "## FlexPension Composite F1 deltas",
            "",
            "| baseline | delta | 95% CI | p(delta <= 0) |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for row in focal_rows:
        if row["model_b"] not in selected_baselines:
            continue
        lines.append(
            f"| {row['model_b']} | {float(row['observed']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | "
            f"{float(row['p_delta_le_0']):.4f} |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_dataset_manifest(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset manifest must be a JSON object keyed by dataset name")
    required = {"prompts", "ground_truth"}
    for dataset, spec in payload.items():
        if not isinstance(spec, dict) or not required.issubset(spec):
            raise ValueError(f"Dataset {dataset} must define prompts and ground_truth")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CI for external four-dataset average.")
    parser.add_argument("--bootstrap-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--type-f1-mode", choices=["binary", "macro"], required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DATASETS
    DATASETS = load_dataset_manifest(args.dataset_manifest)
    run_external_average_bootstrap(
        bootstrap_root=args.bootstrap_root,
        output_dir=args.output_dir,
        type_f1_mode=args.type_f1_mode,
        b=args.bootstrap,
        seed=args.seed,
    )
    print(f"Wrote external average bootstrap CI to {args.output_dir}")


if __name__ == "__main__":
    main()
