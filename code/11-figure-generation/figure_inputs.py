"""Load aggregate, non-identifying inputs used by supplementary figures."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ErrorProfileRow:
    model: str
    total: int
    total_error: int
    accuracy: float
    missed_participation: int
    false_participation: int
    other_error: int


@dataclass(frozen=True)
class LossHistory:
    train_steps: list[int]
    train_losses: list[float]
    eval_steps: list[int]
    eval_losses: list[float]


def load_error_profile(path: Path | str) -> list[ErrorProfileRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    parsed = [
        ErrorProfileRow(
            model=row["model"],
            total=int(row["total"]),
            total_error=int(row["total_error"]),
            accuracy=float(row["accuracy"]),
            missed_participation=int(row["A_false_negative"]),
            false_participation=int(row["C_false_positive"]),
            other_error=int(row["other_error"]),
        )
        for row in rows
    ]
    for row in parsed:
        if row.missed_participation + row.false_participation + row.other_error != row.total_error:
            raise ValueError(f"Error components do not sum to total_error for {row.model}")
    return parsed


def load_loss_points(path: Path | str) -> LossHistory:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    train = payload.get("train") or []
    evaluation = payload.get("eval") or []
    if not train or not evaluation:
        raise ValueError("Loss input must contain non-empty train and eval series")
    return LossHistory(
        train_steps=[int(row["step"]) for row in train],
        train_losses=[float(row["loss"]) for row in train],
        eval_steps=[int(row["step"]) for row in evaluation],
        eval_losses=[float(row["loss"]) for row in evaluation],
    )
