import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bootstrap_evaluation import (
    EvalRecord,
    bootstrap_delta,
    evaluate,
    load_lora_predictions,
    metrics_from_contributions,
    paired_delta,
    prediction_contributions,
    stratified_bootstrap_indices,
)


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class BootstrapEvaluationTests(unittest.TestCase):
    def test_load_lora_predictions_extracts_sample_id_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lora.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "response": json.dumps(
                            {
                                "insurance_decision": {
                                    "action": "参保",
                                    "insurance_type": "城镇职工养老保险",
                                }
                            },
                            ensure_ascii=False,
                        ),
                        "labels": json.dumps(
                            {
                                "household_id": "h1",
                                "individual_id": "2",
                                "insurance_decision": {
                                    "action": "参保",
                                    "insurance_type": "城乡居民养老保险",
                                },
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            )

            predictions = load_lora_predictions(path)

        self.assertEqual(
            predictions["h1-2"],
            {
                "success": True,
                "parse_ok": True,
                "action": "参保",
                "insurance_type": "城镇职工养老保险",
            },
        )

    def test_row_level_lora_predictions_preserve_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lora.jsonl"
            label = json.dumps(
                {
                    "household_id": "dup",
                    "individual_id": "1",
                    "insurance_decision": {
                        "action": "参保",
                        "insurance_type": "城乡居民养老保险",
                    },
                },
                ensure_ascii=False,
            )
            write_jsonl(
                path,
                [
                    {
                        "response": json.dumps(
                            {
                                "insurance_decision": {
                                    "action": "参保",
                                    "insurance_type": "城乡居民养老保险",
                                }
                            },
                            ensure_ascii=False,
                        ),
                        "labels": label,
                    },
                    {
                        "response": json.dumps(
                            {
                                "insurance_decision": {
                                    "action": "参保",
                                    "insurance_type": "城镇职工养老保险",
                                }
                            },
                            ensure_ascii=False,
                        ),
                        "labels": label,
                    },
                ],
            )

            predictions = load_lora_predictions(path, row_level=True)

        records = [
            EvalRecord("dup-1", "参保", "城乡居民养老保险"),
            EvalRecord("dup-1", "参保", "城镇职工养老保险"),
        ]
        self.assertEqual(len(predictions), 2)
        self.assertEqual(evaluate(records, predictions)["type_f1"], 1.0)

    def test_row_level_predictions_align_by_sample_id_when_file_order_differs(self):
        records = [
            EvalRecord("a-1", "参保", "城乡居民养老保险"),
            EvalRecord("b-1", "参保", "城镇职工养老保险"),
        ]
        predictions = [
            {
                "sample_id": "b-1",
                "success": True,
                "parse_ok": True,
                "action": "参保",
                "insurance_type": "城镇职工养老保险",
            },
            {
                "sample_id": "a-1",
                "success": True,
                "parse_ok": True,
                "action": "参保",
                "insurance_type": "城乡居民养老保险",
            },
        ]

        self.assertEqual(evaluate(records, predictions)["type_f1"], 1.0)

    def test_stratified_bootstrap_indices_are_reproducible_and_keep_size(self):
        labels = ["不参保", "城乡居民养老保险", "城乡居民养老保险", "城镇职工养老保险"]

        first = stratified_bootstrap_indices(labels, seed=42)
        second = stratified_bootstrap_indices(labels, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(labels))
        self.assertTrue(all(0 <= idx < len(labels) for idx in first))

    def test_bootstrap_delta_reports_observed_delta_and_ci(self):
        records = [
            EvalRecord("1", "不参保", "不参保"),
            EvalRecord("2", "参保", "城乡居民养老保险"),
            EvalRecord("3", "参保", "城镇职工养老保险"),
            EvalRecord("4", "参保", "城镇职工养老保险"),
        ]
        model_a = {
            "1": {"success": True, "parse_ok": True, "action": "不参保", "insurance_type": "不参保"},
            "2": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城乡居民养老保险"},
            "3": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城镇职工养老保险"},
            "4": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城镇职工养老保险"},
        }
        model_b = {
            "1": {"success": True, "parse_ok": True, "action": "不参保", "insurance_type": "不参保"},
            "2": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城乡居民养老保险"},
            "3": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城乡居民养老保险"},
            "4": {"success": True, "parse_ok": True, "action": "不参保", "insurance_type": "不参保"},
        }

        observed = paired_delta(records, model_a, model_b, "composite_f1")
        result = bootstrap_delta(records, model_a, model_b, metric="composite_f1", b=50, seed=7)

        self.assertEqual(result["observed_delta"], observed)
        self.assertEqual(result["n_bootstrap"], 50)
        self.assertLessEqual(result["ci_low"], result["ci_high"])
        self.assertIn("p_delta_le_0", result)

    def test_contribution_metrics_match_evaluate(self):
        records = [
            EvalRecord("1", "不参保", "不参保"),
            EvalRecord("2", "参保", "城乡居民养老保险"),
            EvalRecord("3", "参保", "城镇职工养老保险"),
            EvalRecord("4", "参保", "城镇职工养老保险"),
        ]
        predictions = {
            "1": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城乡居民养老保险"},
            "2": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城镇职工养老保险"},
            "3": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城镇职工养老保险"},
            "4": {"success": True, "parse_ok": False, "action": "参保", "insurance_type": "城镇职工养老保险"},
        }

        direct = evaluate(records, predictions)
        cached = metrics_from_contributions(prediction_contributions(records, predictions))

        self.assertEqual(cached, direct)

    def test_macro_type_f1_averages_employee_and_resident_type_f1(self):
        records = [
            EvalRecord("1", "参保", "城乡居民养老保险"),
            EvalRecord("2", "参保", "城镇职工养老保险"),
        ]
        predictions = {
            "1": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城镇职工养老保险"},
            "2": {"success": True, "parse_ok": True, "action": "参保", "insurance_type": "城镇职工养老保险"},
        }

        binary = evaluate(records, predictions)
        macro = evaluate(records, predictions, type_f1_mode="macro")

        self.assertAlmostEqual(binary["type_f1"], 2 / 3)
        self.assertAlmostEqual(macro["type_f1"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
