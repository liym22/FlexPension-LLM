# Bootstrap Evaluation Utilities

This directory contains bootstrap evaluation utilities used for the reported experiments.

The public release includes reusable code and aggregate summaries only. It intentionally excludes raw survey data, sample-level prompts, model outputs, teacher rationales, training checkpoints, and local run logs.

Files:

- `bootstrap_evaluation.py`: paired stratified bootstrap evaluation.
- `bootstrap_external_average.py`: external-average confidence intervals; requires a user-provided dataset manifest.
- `compare_binary_macro_outputs.py`: binary/macro Type-F1 comparison helper.
- `summarize_bootstrap_outputs.py`: aggregate summary generator.
- `summarize_training_robustness_bootstrap.py`: training robustness summary generator.
- `convert_teacher_json_to_closeai_jsonl.py`: internal-format conversion helper; do not publish converted sample-level outputs.
- `tests/`: unit tests based on toy data.
