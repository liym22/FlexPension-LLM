#!/usr/bin/env python
"""Render the supplementary error-profile and loss-curve figures."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from figure_inputs import load_error_profile, load_loss_points


def _load_plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "font-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Times", "serif"],
            "font.size": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _style_axis(axis) -> None:
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.42)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def render_error_profile(input_path: Path, output_path: Path) -> None:
    plt = _load_plotting()
    rows = load_error_profile(input_path)
    colors = ["#315EA8", "#D97706", "#94A3B8"]
    labels = ["Missed participation", "False participation", "Other mismatches"]
    figure, axis = plt.subplots(figsize=(6.85, 2.55))

    bottoms = [0] * len(rows)
    for color, label, values in zip(
        colors,
        labels,
        [
            [row.missed_participation for row in rows],
            [row.false_participation for row in rows],
            [row.other_error for row in rows],
        ],
    ):
        axis.bar(range(len(rows)), values, bottom=bottoms, color=color, label=label, width=0.52)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for index, row in enumerate(rows):
        axis.text(index, row.total_error + 14, f"errors={row.total_error}\nacc={row.accuracy:.1%}", ha="center")
    axis.set_xticks(range(len(rows)), [row.model for row in rows])
    axis.set_ylabel("Number of errors")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False)
    _style_axis(axis)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.31)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _moving_average(values: list[float], window: int = 25) -> list[float]:
    return [sum(values[max(0, index - window + 1) : index + 1]) / min(index + 1, window) for index in range(len(values))]


def render_loss_curves(input_path: Path, output_path: Path) -> None:
    plt = _load_plotting()
    history = load_loss_points(input_path)
    figure, (train_axis, eval_axis) = plt.subplots(1, 2, figsize=(6.85, 2.15), gridspec_kw={"wspace": 0.30})
    train_axis.plot(history.train_steps, history.train_losses, color="#B45309", alpha=0.18, linewidth=0.55, label="Raw train loss")
    train_axis.plot(history.train_steps, _moving_average(history.train_losses), color="#B45309", linewidth=1.25, label="Smoothed train loss")
    train_axis.set_title("(a) Training loss", fontweight="bold")
    train_axis.set_xlabel("Training step")
    train_axis.set_ylabel("Loss")
    train_axis.legend(frameon=False)
    _style_axis(train_axis)

    eval_axis.plot(history.eval_steps, history.eval_losses, color="#2F6DB3", marker="o", markersize=2.8, linewidth=1.15)
    eval_axis.set_title("(b) Evaluation loss", fontweight="bold")
    eval_axis.set_xlabel("Training step")
    eval_axis.set_ylabel("Loss")
    _style_axis(eval_axis)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["error-profile", "loss-curves"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.kind == "error-profile":
        render_error_profile(args.input, args.output)
    else:
        render_loss_curves(args.input, args.output)


if __name__ == "__main__":
    main()
