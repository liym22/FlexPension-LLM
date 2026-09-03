#!/usr/bin/env python
"""Extract aggregate train/evaluation loss points from an ms-swift training log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_points(history: list[dict]) -> dict[str, list[dict[str, int | float]]]:
    train = []
    evaluation = []
    for row in history:
        if row.get("step") is None:
            continue
        if row.get("loss") is not None:
            train.append({"step": int(row["step"]), "loss": round(float(row["loss"]), 8)})
        if row.get("eval_loss") is not None:
            evaluation.append({"step": int(row["step"]), "loss": round(float(row["eval_loss"]), 8)})
    if not train or not evaluation:
        raise ValueError("Training log must contain both loss and eval_loss entries")
    return {"train": train, "eval": evaluation}


def load_history(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in reversed(rows):
            if row.get("log_history"):
                return row["log_history"]
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("log_history") or payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    points = extract_points(load_history(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(points, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
