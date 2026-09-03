import json

from evaluate_results import load_predictions


def test_load_predictions_returns_model_metadata(tmp_path):
    path = tmp_path / "demo_results.jsonl"
    row = {
        "run_id": "blind13_20260704",
        "run_type": "blind",
        "model_id": "demo-model",
        "model_short_name": "demo_model",
        "sample_id": "s1",
        "success": True,
        "parse_success": True,
        "predicted_action": "参保",
        "predicted_insurance_type": "城乡居民养老保险",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        "cost_rmb": 0.01,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    predictions, metadata = load_predictions(path)

    assert metadata["run_id"] == "blind13_20260704"
    assert metadata["run_type"] == "blind"
    assert metadata["model_id"] == "demo-model"
    assert metadata["model_short_name"] == "demo_model"
    assert predictions["s1"]["action"] == "参保"
