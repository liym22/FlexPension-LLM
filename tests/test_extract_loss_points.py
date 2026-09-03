import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "11-figure-generation"
    / "extract_loss_points.py"
)
SPEC = importlib.util.spec_from_file_location("extract_loss_points", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_extract_points_ignores_non_loss_log_rows():
    history = [
        {"step": 1, "loss": 0.9},
        {"step": 2, "learning_rate": 0.01},
        {"step": 3, "eval_loss": 0.7},
    ]

    points = MODULE.extract_points(history)

    assert points == {
        "train": [{"step": 1, "loss": 0.9}],
        "eval": [{"step": 3, "loss": 0.7}],
    }
