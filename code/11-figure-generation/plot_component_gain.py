#!/usr/bin/env python
"""Render compact component-gain evidence for the main paper."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "font-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ComponentRow:
    label: str
    slice_label: str
    start: float
    end: float
    delta: float
    ci_low: float
    ci_high: float
    group: str


ROWS = [
    ComponentRow("Claude DKI", "CHFS all", 0.8457, 0.9125, 0.0668, 0.0607, 0.0731, "DKI prompt"),
    ComponentRow("Claude DKI", "Blind", 0.8586, 0.9209, 0.0623, 0.0467, 0.0782, "DKI prompt"),
    ComponentRow("Qwen DKI", "Blind", 0.7750, 0.7871, 0.0121, -0.0159, 0.0400, "DKI prompt"),
    ComponentRow("Full vs. SFT", "Ext. avg.", 0.7378, 0.7549, 0.0171, -0.0111, 0.0446, "Rationale"),
]


def render_component_gain(
    *,
    pdf_path: Path | str,
    svg_path: Path | str | None = None,
    png_path: Path | str | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Times", "serif"],
            "font.size": 9.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    colors = {
        "DKI prompt": "#315EA8",
        "Rationale": "#B45309",
        "text": "#111827",
        "muted": "#6B7280",
        "grid": "#E5E7EB",
    }

    fig = plt.figure(figsize=(3.62, 2.35))
    table_ax = fig.add_axes([0.01, 0.04, 0.98, 0.92])
    table_ax.set_xlim(0, 1)
    table_ax.set_ylim(0, 1)
    table_ax.axis("off")

    rows = ROWS
    header_y = 0.91
    row_ys = [0.735, 0.555, 0.375, 0.195]
    col_label = 0.02
    forest_left = 0.335
    forest_right = 0.785
    col_delta = 0.815

    table_ax.text(col_label, header_y, "Comparison", fontsize=9.2, weight="bold", color=colors["text"], va="center")
    table_ax.text((forest_left + forest_right) / 2, header_y, "$\\Delta$ with 95\\% CI", fontsize=9.2, weight="bold", color=colors["text"], ha="center", va="center")
    table_ax.text(col_delta, header_y, "Estimate", fontsize=9.2, weight="bold", color=colors["text"], va="center")
    table_ax.hlines(0.835, 0.01, 0.99, color="#9CA3AF", linewidth=0.65)

    def map_gain(value: float) -> float:
        min_gain, max_gain = -0.02, 0.08
        return forest_left + (value - min_gain) / (max_gain - min_gain) * (forest_right - forest_left)

    zero_x = map_gain(0)
    table_ax.vlines(zero_x, 0.10, 0.82, color="#9CA3AF", linewidth=0.65)
    table_ax.axvspan(map_gain(0), forest_right, ymin=0.08, ymax=0.82, color="#EFF6FF", alpha=0.55, zorder=0)

    for tick in [-0.02, 0.02, 0.04, 0.06, 0.08]:
        x = map_gain(tick)
        table_ax.vlines(x, 0.10, 0.82, color=colors["grid"], linewidth=0.4)
        label = f"{tick:+.2f}".replace("+0", "+").replace("-0", "-")
        table_ax.text(x, 0.063, label, ha="center", va="center", fontsize=7.9, color=colors["muted"])
    table_ax.text(zero_x, 0.065, "0", ha="center", va="center", fontsize=7.9, color=colors["muted"])

    for y in [0.64, 0.46, 0.28, 0.10]:
        table_ax.hlines(y, 0.01, 0.99, color="#EEF2F7", linewidth=0.45)

    for y, row in zip(row_ys, rows):
        color = colors[row.group]
        table_ax.text(
            col_label,
            y + 0.037,
            f"{row.label} / {row.slice_label}",
            fontsize=9.0,
            color=colors["text"],
            va="center",
        )
        table_ax.text(
            col_label,
            y - 0.038,
            f"{row.start:.4f} $\\rightarrow$ {row.end:.4f}",
            fontsize=8.1,
            color=colors["muted"],
            va="center",
        )

        low_x = map_gain(row.ci_low)
        high_x = map_gain(row.ci_high)
        delta_x = map_gain(row.delta)
        table_ax.hlines(y, low_x, high_x, color=color, linewidth=1.55)
        table_ax.vlines([low_x, high_x], y - 0.032, y + 0.032, color=color, linewidth=1.05)
        table_ax.scatter([delta_x], [y], s=31, color=color, edgecolor="white", linewidth=0.55, zorder=3)

        table_ax.text(col_delta, y + 0.034, f"{row.delta:+.4f}", fontsize=8.7, color=colors["text"], va="center")
        table_ax.text(
            col_delta,
            y - 0.040,
            f"[{row.ci_low:+.4f},{row.ci_high:+.4f}]".replace("[+", "["),
            fontsize=7.7,
            color=colors["muted"],
            va="center",
        )

    output_paths = [Path(pdf_path)]
    if svg_path is not None:
        output_paths.append(Path(svg_path))
    if png_path is not None:
        output_paths.append(Path(png_path))
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    if svg_path is not None:
        fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    if png_path is not None:
        fig.savefig(png_path, format="png", dpi=240, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default=Path("outputs/figures/component_gain_plot.pdf"), type=Path)
    parser.add_argument("--svg", default=Path("outputs/figures/component_gain_plot.svg"), type=Path)
    parser.add_argument("--png", default=Path("outputs/figures/component_gain_plot.png"), type=Path)
    args = parser.parse_args()
    render_component_gain(pdf_path=args.pdf, svg_path=args.svg, png_path=args.png)


if __name__ == "__main__":
    main()
