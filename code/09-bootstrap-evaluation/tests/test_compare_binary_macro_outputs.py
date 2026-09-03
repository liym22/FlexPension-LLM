import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "compare_binary_macro_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_binary_macro_outputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CompareBinaryMacroOutputsTests(unittest.TestCase):
    def test_join_rows_pairs_binary_and_macro_metrics(self):
        module = load_module()
        binary_rows = [
            {
                "dataset": "blind",
                "model": "flexpension",
                "parse_success_rate": "1.0",
                "action_f1": "0.9",
                "type_f1": "0.8",
                "composite_f1": "0.86",
            }
        ]
        macro_rows = [
            {
                "dataset": "blind",
                "model": "flexpension",
                "parse_success_rate": "1.0",
                "action_f1": "0.9",
                "type_f1": "0.95",
                "composite_f1": "0.92",
            }
        ]

        joined = module.join_metric_rows(binary_rows, macro_rows)

        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]["dataset"], "blind")
        self.assertEqual(joined[0]["type_f1_binary"], 0.8)
        self.assertAlmostEqual(joined[0]["type_f1_delta_macro_minus_binary"], 0.15)
        self.assertAlmostEqual(joined[0]["composite_f1_delta_macro_minus_binary"], 0.06)

    def test_write_outputs_creates_readme_with_updated_external_rank(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            binary_dir = tmp_path / "binary"
            macro_dir = tmp_path / "macro"
            out_dir = tmp_path / "out"
            binary_dir.mkdir()
            macro_dir.mkdir()
            binary_dir.joinpath("all_model_metrics.csv").write_text(
                "\n".join(
                    [
                        "dataset,model,parse_success_rate,action_f1,type_f1,composite_f1",
                        "blind,flexpension,1.0,0.9,0.8,0.86",
                    ]
                ),
                encoding="utf-8",
            )
            macro_dir.joinpath("all_model_metrics.csv").write_text(
                "\n".join(
                    [
                        "dataset,model,parse_success_rate,action_f1,type_f1,composite_f1",
                        "blind,flexpension,1.0,0.9,0.95,0.92",
                    ]
                ),
                encoding="utf-8",
            )
            binary_dir.joinpath("external_macro_average_metrics.csv").write_text(
                "\n".join(
                    [
                        "dataset,model,n_datasets,datasets,parse_success_rate,action_f1,type_f1,composite_f1",
                        "external_macro_average,flexpension,4,a;b;c;d,1.0,0.93,0.48,0.75",
                        "external_macro_average,baseline,4,a;b;c;d,1.0,0.92,0.47,0.74",
                    ]
                ),
                encoding="utf-8",
            )
            macro_dir.joinpath("external_macro_average_metrics.csv").write_text(
                "\n".join(
                    [
                        "dataset,model,n_datasets,datasets,parse_success_rate,action_f1,type_f1,composite_f1",
                        "external_macro_average,flexpension,4,a;b;c;d,1.0,0.93,0.69,0.84",
                        "external_macro_average,baseline,4,a;b;c;d,1.0,0.92,0.68,0.82",
                    ]
                ),
                encoding="utf-8",
            )

            module.write_outputs(binary_dir, macro_dir, out_dir)

            readme = out_dir.joinpath("README.md").read_text(encoding="utf-8")

        self.assertIn("| binary Type F1 | 0.8600 | 0.7500 |", readme)
        self.assertIn("| 1 | flexpension | 0.7500 | 0.8400 | +0.0900 |", readme)


if __name__ == "__main__":
    unittest.main()
