from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACTION_POSITIVE = "参保"
TYPE_RESIDENT = "城乡居民养老保险"
TYPE_POSITIVE = "城镇职工养老保险"
TYPE_LABELS = (TYPE_RESIDENT, TYPE_POSITIVE)
TYPE_F1_MODES = ("binary", "macro")
METRICS = ("action_f1", "type_f1", "composite_f1")
PredictionData = dict[str, dict[str, Any]] | list[dict[str, Any]]


@dataclass(frozen=True)
class EvalRecord:
    sample_id: str
    action: str
    insurance_type: str

    @property
    def terminal_label(self) -> str:
        if self.action != ACTION_POSITIVE:
            return "不参保"
        return self.insurance_type or "__UNKNOWN__"


@dataclass(frozen=True)
class MetricContribution:
    missing: int
    parse_success: int
    action_tp: int
    action_fp: int
    action_fn: int
    type_eval: int
    type_tp: int
    type_fp: int
    type_fn: int
    type_resident_tp: int
    type_resident_fp: int
    type_resident_fn: int


def _safe_json_loads(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None

    fenced = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    for block in reversed(fenced):
        parsed = _safe_json_loads(block)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def decision_from_payload(payload: dict[str, Any] | None) -> tuple[str, str]:
    if not payload:
        return "不参保", "不参保"
    decision = payload.get("insurance_decision") or payload.get("insuranceDecision") or {}
    action = decision.get("action") or "不参保"
    insurance_type = decision.get("insurance_type") or decision.get("insuranceType") or "不参保"
    return str(action), str(insurance_type)


def sample_id_from_payload(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    household_id = payload.get("household_id") or payload.get("householdId")
    individual_id = payload.get("individual_id") or payload.get("individualId")
    if household_id is None or individual_id is None:
        return None
    return f"{household_id}-{individual_id}"


def load_distill_records(path: Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            assistant = next((m for m in messages if m.get("role") == "assistant"), None)
            if not assistant:
                raise ValueError(f"{path}:{line_no} missing assistant truth message")
            payload = _safe_json_loads(assistant.get("content", ""))
            sample_id = sample_id_from_payload(payload)
            if not sample_id:
                raise ValueError(f"{path}:{line_no} missing household_id/individual_id")
            action, insurance_type = decision_from_payload(payload)
            records.append(EvalRecord(sample_id, action, insurance_type))
    return records


def load_external_records(prompts_path: Path, ground_truth_path: Path) -> list[EvalRecord]:
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    truths = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    truth_by_id = {str(row["id"]): row for row in truths}
    records: list[EvalRecord] = []
    for prompt_row in prompts:
        sample_id = str(prompt_row["id"])
        truth = truth_by_id[sample_id]
        action = truth.get("decision") or "不参保"
        insurance_type = truth.get("type") or "不参保"
        records.append(EvalRecord(sample_id, action, insurance_type))
    return records


def load_closeai_predictions(path: Path, *, row_level: bool = False) -> PredictionData:
    predictions: PredictionData = [] if row_level else {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            prediction = {
                "success": bool(row.get("success", False)),
                "parse_ok": bool(row.get("parse_success", row.get("parse_ok", False))),
                "action": row.get("predicted_action") or row.get("action"),
                "insurance_type": row.get("predicted_insurance_type") or row.get("insurance_type"),
            }
            if row_level:
                predictions.append({"sample_id": sample_id, **prediction})  # type: ignore[union-attr]
            else:
                predictions[sample_id] = prediction  # type: ignore[index]
    return predictions


def load_lora_predictions(path: Path, *, row_level: bool = False) -> PredictionData:
    predictions: PredictionData = [] if row_level else {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            label_payload = _safe_json_loads(row.get("labels"))
            sample_id = sample_id_from_payload(label_payload)
            if not sample_id:
                response_payload = extract_json_object(row.get("response"))
                sample_id = sample_id_from_payload(response_payload)
            if not sample_id:
                raise ValueError(f"{path}:{line_no} missing sample id")

            parsed = extract_json_object(row.get("response"))
            action, insurance_type = decision_from_payload(parsed)
            prediction = {
                "success": True,
                "parse_ok": parsed is not None,
                "action": action,
                "insurance_type": insurance_type,
            }
            if row_level:
                predictions.append({"sample_id": sample_id, **prediction})  # type: ignore[union-attr]
            else:
                predictions[sample_id] = prediction  # type: ignore[index]
    return predictions


def load_predictions(path: Path, kind: str, *, row_level: bool = False) -> PredictionData:
    if kind == "closeai":
        return load_closeai_predictions(path, row_level=row_level)
    if kind == "lora":
        return load_lora_predictions(path, row_level=row_level)
    raise ValueError(f"Unsupported prediction kind: {kind}")


def _binary_f1(y_true: list[str], y_pred: list[str], positive_label: str) -> float:
    tp = sum(t == positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fp = sum(t != positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fn = sum(t == positive_label and p != positive_label for t, p in zip(y_true, y_pred))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def align_row_level_predictions(
    records: list[EvalRecord], predictions: list[dict[str, Any]]
) -> list[dict[str, Any] | None]:
    needed = Counter(record.sample_id for record in records)
    queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for pred in predictions:
        sample_id = pred.get("sample_id")
        if sample_id is not None:
            queues[str(sample_id)].append(pred)

    aligned: list[dict[str, Any] | None] = []
    for record in records:
        queue = queues.get(record.sample_id)
        if not queue:
            aligned.append(None)
            continue

        if len(queue) >= needed[record.sample_id]:
            pred = queue.popleft()
        elif len(queue) > 1:
            pred = queue.popleft()
        else:
            pred = queue[0]

        needed[record.sample_id] -= 1
        aligned.append(pred)
    return aligned


def prediction_contributions(
    records: Iterable[EvalRecord], predictions: PredictionData
) -> list[MetricContribution]:
    contributions: list[MetricContribution] = []
    records_list = list(records)
    row_level = isinstance(predictions, list)
    aligned_predictions = align_row_level_predictions(records_list, predictions) if row_level else []
    for idx, record in enumerate(records_list):
        if row_level:
            pred = aligned_predictions[idx]
        else:
            pred = predictions.get(record.sample_id)  # type: ignore[union-attr]
        missing = int(pred is None)
        if pred is None:
            pred = {"success": False, "parse_ok": False}

        parse_ok = bool(pred.get("parse_ok", pred.get("parse_success", False)))
        pred_action = (pred.get("action") or pred.get("predicted_action")) if parse_ok else "不参保"
        pred_type = (pred.get("insurance_type") or pred.get("predicted_insurance_type")) if parse_ok else "不参保"
        pred_action = pred_action or "不参保"
        pred_type = pred_type or "__INVALID__"

        action_tp = int(record.action == ACTION_POSITIVE and pred_action == ACTION_POSITIVE)
        action_fp = int(record.action != ACTION_POSITIVE and pred_action == ACTION_POSITIVE)
        action_fn = int(record.action == ACTION_POSITIVE and pred_action != ACTION_POSITIVE)

        include_type = parse_ok and record.action == ACTION_POSITIVE and pred_action == ACTION_POSITIVE
        type_tp = int(include_type and record.insurance_type == TYPE_POSITIVE and pred_type == TYPE_POSITIVE)
        type_fp = int(include_type and record.insurance_type != TYPE_POSITIVE and pred_type == TYPE_POSITIVE)
        type_fn = int(include_type and record.insurance_type == TYPE_POSITIVE and pred_type != TYPE_POSITIVE)
        type_resident_tp = int(include_type and record.insurance_type == TYPE_RESIDENT and pred_type == TYPE_RESIDENT)
        type_resident_fp = int(include_type and record.insurance_type != TYPE_RESIDENT and pred_type == TYPE_RESIDENT)
        type_resident_fn = int(include_type and record.insurance_type == TYPE_RESIDENT and pred_type != TYPE_RESIDENT)

        contributions.append(
            MetricContribution(
                missing=missing,
                parse_success=int(parse_ok),
                action_tp=action_tp,
                action_fp=action_fp,
                action_fn=action_fn,
                type_eval=int(include_type),
                type_tp=type_tp,
                type_fp=type_fp,
                type_fn=type_fn,
                type_resident_tp=type_resident_tp,
                type_resident_fp=type_resident_fp,
                type_resident_fn=type_resident_fn,
            )
        )
    return contributions


def metrics_from_contributions(
    contributions: list[MetricContribution],
    indices: list[int] | None = None,
    *,
    type_f1_mode: str = "binary",
) -> dict[str, Any]:
    if type_f1_mode not in TYPE_F1_MODES:
        raise ValueError(f"Unsupported type_f1_mode: {type_f1_mode}")
    rows = contributions if indices is None else [contributions[i] for i in indices]
    n_total = len(rows)
    action_tp = sum(row.action_tp for row in rows)
    action_fp = sum(row.action_fp for row in rows)
    action_fn = sum(row.action_fn for row in rows)
    type_tp = sum(row.type_tp for row in rows)
    type_fp = sum(row.type_fp for row in rows)
    type_fn = sum(row.type_fn for row in rows)
    type_resident_tp = sum(row.type_resident_tp for row in rows)
    type_resident_fp = sum(row.type_resident_fp for row in rows)
    type_resident_fn = sum(row.type_resident_fn for row in rows)

    action_f1 = _f1_from_counts(action_tp, action_fp, action_fn)
    type_employee_f1 = _f1_from_counts(type_tp, type_fp, type_fn)
    if type_f1_mode == "macro":
        type_resident_f1 = _f1_from_counts(type_resident_tp, type_resident_fp, type_resident_fn)
        type_f1 = sum((type_resident_f1, type_employee_f1)) / len(TYPE_LABELS)
    else:
        type_f1 = type_employee_f1
    composite_f1 = 0.6 * action_f1 + 0.4 * type_f1
    parse_success = sum(row.parse_success for row in rows)
    return {
        "n_total": n_total,
        "n_missing_predictions": sum(row.missing for row in rows),
        "n_parse_success": parse_success,
        "parse_success_rate": parse_success / n_total if n_total else 0.0,
        "n_type_eval": sum(row.type_eval for row in rows),
        "action_f1": action_f1,
        "type_f1": type_f1,
        "composite_f1": composite_f1,
    }


def contribution_matrix(contributions: list[MetricContribution]) -> np.ndarray:
    return np.asarray(
        [
            [
                row.missing,
                row.parse_success,
                row.action_tp,
                row.action_fp,
                row.action_fn,
                row.type_eval,
                row.type_tp,
                row.type_fp,
                row.type_fn,
                row.type_resident_tp,
                row.type_resident_fp,
                row.type_resident_fn,
            ]
            for row in contributions
        ],
        dtype=np.int16,
    )


def _f1_vector(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)


def metric_vector_from_count_matrix(counts: np.ndarray, metric: str, *, type_f1_mode: str = "binary") -> np.ndarray:
    if type_f1_mode not in TYPE_F1_MODES:
        raise ValueError(f"Unsupported type_f1_mode: {type_f1_mode}")
    action_f1 = _f1_vector(counts[:, 2], counts[:, 3], counts[:, 4])
    if metric == "action_f1":
        return action_f1

    type_employee_f1 = _f1_vector(counts[:, 6], counts[:, 7], counts[:, 8])
    if type_f1_mode == "macro":
        type_resident_f1 = _f1_vector(counts[:, 9], counts[:, 10], counts[:, 11])
        type_f1 = (type_resident_f1 + type_employee_f1) / len(TYPE_LABELS)
    else:
        type_f1 = type_employee_f1
    if metric == "type_f1":
        return type_f1
    if metric == "composite_f1":
        return 0.6 * action_f1 + 0.4 * type_f1
    raise ValueError(f"Unsupported metric: {metric}")


def evaluate(
    records: Iterable[EvalRecord],
    predictions: PredictionData,
    *,
    type_f1_mode: str = "binary",
) -> dict[str, Any]:
    return metrics_from_contributions(
        prediction_contributions(records, predictions),
        type_f1_mode=type_f1_mode,
    )


def select_records(records: list[EvalRecord], indices: list[int]) -> list[EvalRecord]:
    return [records[i] for i in indices]


def stratified_bootstrap_indices(labels: list[str], seed: int) -> list[int]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(idx)

    sampled: list[int] = []
    for label in sorted(groups):
        group = groups[label]
        sampled.extend(rng.choice(group) for _ in group)
    rng.shuffle(sampled)
    return sampled


def stratified_bootstrap_index_matrix(labels: list[str], b: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(idx)

    sampled_groups: list[np.ndarray] = []
    for label in sorted(groups):
        group = np.asarray(groups[label], dtype=np.int32)
        sampled_groups.append(rng.choice(group, size=(b, len(group)), replace=True))
    return np.concatenate(sampled_groups, axis=1)


def paired_delta(
    records: list[EvalRecord],
    model_a: PredictionData,
    model_b: PredictionData,
    metric: str,
    *,
    type_f1_mode: str = "binary",
) -> float:
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    return float(
        evaluate(records, model_a, type_f1_mode=type_f1_mode)[metric]
        - evaluate(records, model_b, type_f1_mode=type_f1_mode)[metric]
    )


def paired_delta_from_contributions(
    contrib_a: list[MetricContribution],
    contrib_b: list[MetricContribution],
    metric: str,
    indices: list[int] | None = None,
    *,
    type_f1_mode: str = "binary",
) -> float:
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    return float(
        metrics_from_contributions(contrib_a, indices, type_f1_mode=type_f1_mode)[metric]
        - metrics_from_contributions(contrib_b, indices, type_f1_mode=type_f1_mode)[metric]
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_delta(
    records: list[EvalRecord],
    model_a: PredictionData,
    model_b: PredictionData,
    *,
    metric: str,
    b: int,
    seed: int,
    type_f1_mode: str = "binary",
    bootstrap_indices: np.ndarray | None = None,
    chunk_size: int = 1000,
) -> dict[str, Any]:
    labels = [record.terminal_label for record in records]
    contrib_a = prediction_contributions(records, model_a)
    contrib_b = prediction_contributions(records, model_b)
    matrix_a = contribution_matrix(contrib_a)
    matrix_b = contribution_matrix(contrib_b)
    if bootstrap_indices is None:
        bootstrap_indices = stratified_bootstrap_index_matrix(labels, b, seed)

    delta_chunks: list[np.ndarray] = []
    for start in range(0, b, chunk_size):
        indices = bootstrap_indices[start : start + chunk_size]
        counts_a = matrix_a[indices].sum(axis=1)
        counts_b = matrix_b[indices].sum(axis=1)
        delta_chunks.append(
            metric_vector_from_count_matrix(counts_a, metric, type_f1_mode=type_f1_mode)
            - metric_vector_from_count_matrix(counts_b, metric, type_f1_mode=type_f1_mode)
        )
    deltas_array = np.concatenate(delta_chunks)
    deltas = deltas_array.tolist()

    observed = paired_delta_from_contributions(contrib_a, contrib_b, metric, type_f1_mode=type_f1_mode)
    p_delta_le_0 = float(np.mean(deltas_array <= 0))
    p_delta_ge_0 = float(np.mean(deltas_array >= 0))
    return {
        "metric": metric,
        "observed_delta": observed,
        "ci_low": percentile(deltas, 0.025),
        "ci_high": percentile(deltas, 0.975),
        "bootstrap_mean_delta": sum(deltas) / len(deltas),
        "p_delta_le_0": p_delta_le_0,
        "p_delta_ge_0": p_delta_ge_0,
        "n_bootstrap": b,
        "seed": seed,
    }


def parse_model_arg(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("model must use name=kind=path")
    name, kind, path = parts
    if kind not in {"closeai", "lora"}:
        raise argparse.ArgumentTypeError("kind must be closeai or lora")
    return name, kind, Path(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired stratified bootstrap for FlexPension evaluations.")
    parser.add_argument("--dataset-kind", required=True, choices=["distill_jsonl", "external_json"])
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--prompts-path", type=Path)
    parser.add_argument("--ground-truth-path", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model", action="append", required=True, type=parse_model_arg, help="name=kind=path")
    parser.add_argument("--compare", action="append", required=True, help="model_a:model_b")
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=list(METRICS))
    parser.add_argument("--type-f1-mode", choices=list(TYPE_F1_MODES), default="binary")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.dataset_kind == "distill_jsonl":
        if not args.dataset_path:
            raise SystemExit("--dataset-path is required for distill_jsonl")
        records = load_distill_records(args.dataset_path)
    else:
        if not args.prompts_path or not args.ground_truth_path:
            raise SystemExit("--prompts-path and --ground-truth-path are required for external_json")
        records = load_external_records(args.prompts_path, args.ground_truth_path)

    row_level_predictions = args.dataset_kind == "external_json"
    model_predictions: dict[str, PredictionData] = {}
    model_rows: list[dict[str, Any]] = []
    for name, kind, path in args.model:
        predictions = load_predictions(path, kind, row_level=row_level_predictions)
        model_predictions[name] = predictions
        metrics = evaluate(records, predictions, type_f1_mode=args.type_f1_mode)
        model_rows.append({"dataset": args.dataset_name, "model": name, "kind": kind, "path": str(path), **metrics})

    bootstrap_indices = stratified_bootstrap_index_matrix(
        [record.terminal_label for record in records], args.bootstrap, args.seed
    )
    comparison_rows: list[dict[str, Any]] = []
    for comparison in args.compare:
        try:
            model_a_name, model_b_name = comparison.split(":", 1)
        except ValueError as exc:
            raise SystemExit("--compare must use model_a:model_b") from exc
        if model_a_name not in model_predictions or model_b_name not in model_predictions:
            raise SystemExit(f"Unknown comparison model in {comparison}")
        for metric in args.metrics:
            row = bootstrap_delta(
                records,
                model_predictions[model_a_name],
                model_predictions[model_b_name],
                metric=metric,
                b=args.bootstrap,
                seed=args.seed,
                type_f1_mode=args.type_f1_mode,
                bootstrap_indices=bootstrap_indices,
            )
            comparison_rows.append(
                {
                    "dataset": args.dataset_name,
                    "model_a": model_a_name,
                    "model_b": model_b_name,
                    **row,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{args.dataset_name}_model_metrics.csv", model_rows)
    write_csv(args.output_dir / f"{args.dataset_name}_bootstrap_deltas.csv", comparison_rows)
    payload = {
        "dataset": args.dataset_name,
        "dataset_kind": args.dataset_kind,
        "n_records": len(records),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "type_f1_mode": args.type_f1_mode,
        "models": model_rows,
        "comparisons": comparison_rows,
    }
    (args.output_dir / f"{args.dataset_name}_bootstrap_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
