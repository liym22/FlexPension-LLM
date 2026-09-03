import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "convert_teacher_json_to_closeai_jsonl.py"


def load_module():
    spec = importlib.util.spec_from_file_location("convert_teacher_json_to_closeai_jsonl", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConvertTeacherJsonTests(unittest.TestCase):
    def test_default_conversion_preserves_row_level_predictions(self):
        module = load_module()
        rows = [
            {"sample_id": "1-1", "success": False, "parse_success": False},
            {"sample_id": "1-1", "success": True, "parse_success": True},
        ]

        normalized = module.normalize_rows(rows, "run", "model", latest_per_id=False)

        self.assertEqual([row["sample_id"] for row in normalized], ["1-1", "1-1"])
        self.assertEqual([row["parse_success"] for row in normalized], [False, True])

    def test_latest_per_id_conversion_keeps_last_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "teacher.json"
            output_path = tmp_path / "out.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "tests": [
                            {"sample_id": "1-1", "success": False, "parse_success": False},
                            {"sample_id": "2-1", "success": True, "parse_success": True},
                            {"sample_id": "1-1", "success": True, "parse_success": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--run-id",
                    "run",
                    "--latest-per-id",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], ["2-1", "1-1"])
            self.assertEqual([row["parse_success"] for row in rows], [True, True])
            self.assertIn("Converted 3 rows to 2", completed.stdout)


if __name__ == "__main__":
    unittest.main()
