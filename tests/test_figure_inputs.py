import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "11-figure-generation"
    / "figure_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("figure_inputs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_error_profile_rows_are_complete_aggregates(tmp_path):
    path = tmp_path / "errors.tsv"
    path.write_text(
        "model\ttotal\ttotal_error\taccuracy\tA_false_negative\tC_false_positive\tother_error\n"
        "Model A\t10\t4\t0.6\t2\t1\t1\n",
        encoding="utf-8",
    )

    rows = MODULE.load_error_profile(path)

    assert rows[0].model == "Model A"
    assert rows[0].total_error == 4
    assert rows[0].missed_participation + rows[0].false_participation + rows[0].other_error == 4


def test_loss_points_load_train_and_eval_series(tmp_path):
    path = tmp_path / "loss.json"
    path.write_text(
        json.dumps(
            {
                "train": [{"step": 1, "loss": 0.9}, {"step": 2, "loss": 0.7}],
                "eval": [{"step": 2, "loss": 0.8}],
            }
        ),
        encoding="utf-8",
    )

    history = MODULE.load_loss_points(path)

    assert history.train_steps == [1, 2]
    assert history.train_losses == [0.9, 0.7]
    assert history.eval_steps == [2]
    assert history.eval_losses == [0.8]
