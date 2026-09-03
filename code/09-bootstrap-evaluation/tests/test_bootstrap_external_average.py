import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "bootstrap_external_average.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bootstrap_external_average", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BootstrapExternalAverageTests(unittest.TestCase):
    def test_average_vectors_uses_equal_dataset_weights(self):
        module = load_module()
        vectors = {
            "dataset_a": np.asarray([0.8, 0.6]),
            "dataset_b": np.asarray([0.2, 0.4]),
        }

        averaged = module.average_vectors(vectors)

        np.testing.assert_allclose(averaged, np.asarray([0.5, 0.5]))

    def test_interval_summary_uses_percentile_ci(self):
        module = load_module()
        summary = module.interval_summary(np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]), observed=2.0)

        self.assertEqual(summary["observed"], 2.0)
        self.assertAlmostEqual(summary["bootstrap_mean"], 2.0)
        self.assertAlmostEqual(summary["ci_low"], 0.1)
        self.assertAlmostEqual(summary["ci_high"], 3.9)

    def test_delta_vectors_are_paired_before_averaging(self):
        module = load_module()
        model_a = {
            "dataset_a": np.asarray([0.8, 0.7]),
            "dataset_b": np.asarray([0.3, 0.5]),
        }
        model_b = {
            "dataset_a": np.asarray([0.6, 0.6]),
            "dataset_b": np.asarray([0.4, 0.2]),
        }

        delta = module.average_delta_vectors(model_a, model_b)

        np.testing.assert_allclose(delta, np.asarray([0.05, 0.2]))

    def test_load_bootstrap_module_registers_dataclass_module(self):
        module = load_module()

        bootstrap_module = module.load_bootstrap_module()

        self.assertTrue(hasattr(bootstrap_module, "EvalRecord"))


if __name__ == "__main__":
    unittest.main()
