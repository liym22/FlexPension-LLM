from datasets import EvalSample
from metrics import evaluate_predictions


def test_evaluate_predictions_uses_paper_composite_f1():
    samples = [
        EvalSample("1", "1", "1", "p1", "参保", "城乡居民养老保险"),
        EvalSample("2", "1", "2", "p2", "参保", "城镇职工养老保险"),
        EvalSample("3", "1", "3", "p3", "不参保", "不参保"),
    ]
    predictions = {
        "1": {"action": "参保", "insurance_type": "城乡居民养老保险", "parse_ok": True, "success": True},
        "2": {"action": "参保", "insurance_type": "城乡居民养老保险", "parse_ok": True, "success": True},
        "3": {"action": "不参保", "insurance_type": "不参保", "parse_ok": True, "success": True},
    }
    result = evaluate_predictions(samples, predictions)
    assert result["n_total"] == 3
    assert 0.0 <= result["action_f1"] <= 1.0
    assert 0.0 <= result["type_f1"] <= 1.0
    assert result["composite_f1"] == 0.6 * result["action_f1"] + 0.4 * result["type_f1"]


def test_type_f1_uses_employee_insurance_as_positive_class():
    samples = [
        EvalSample("1", "1", "1", "p1", "参保", "城乡居民养老保险"),
        EvalSample("2", "1", "2", "p2", "参保", "城镇职工养老保险"),
        EvalSample("3", "1", "3", "p3", "不参保", "不参保"),
    ]
    predictions = {
        "1": {"action": "参保", "insurance_type": "城乡居民养老保险", "parse_ok": True, "success": True},
        "2": {"action": "参保", "insurance_type": "城乡居民养老保险", "parse_ok": True, "success": True},
        "3": {"action": "不参保", "insurance_type": "不参保", "parse_ok": True, "success": True},
    }

    result = evaluate_predictions(samples, predictions)

    assert result["action_f1"] == 1.0
    assert result["type_f1"] == 0.0
    assert result["composite_f1"] == 0.6


def test_evaluate_predictions_can_score_sampled_results_only():
    samples = [
        EvalSample("1", "1", "1", "p1", "参保", "城乡居民养老保险"),
        EvalSample("2", "1", "2", "p2", "参保", "城镇职工养老保险"),
    ]
    predictions = {
        "1": {"action": "参保", "insurance_type": "城乡居民养老保险", "parse_ok": True, "success": True},
    }
    result = evaluate_predictions(samples, predictions, require_complete=False)
    assert result["n_total"] == 1
    assert result["n_missing_predictions"] == 1
