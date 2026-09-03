# FlexPension-LLM Reproducibility Package

**Paper:** *Can Large Language Models Anticipate Behavioral Responses to Social Policies? A Case of Pension Enrollment Prediction among China's Flexible Workers*

**Authors:** Yumiao Li, Peixin Liu, Donglin Di, Chen Li, Runhuan Feng

This public reproducibility package contains the source code used for data preparation, Probit prior extraction, Domain Knowledge Injection (DKI), teacher rationale generation, error-filtered supervision construction, LoRA/SFT training, blind and external evaluation, and paired bootstrap analysis.

## Availability Boundary

CHFS, CFPS, CHIP, and CLDS are controlled-access survey datasets. Their raw records cannot be redistributed in this package. Researchers must obtain the datasets from their official providers and place the files under the paths described in `REPRODUCTION.md`.

The package does not include survey microdata, sample-level prompts, ground-truth records, teacher or student outputs, rationales, model checkpoints, or API credentials. It includes the two province-level policy workbooks used by the prompt-construction code and non-identifying aggregate statistical outputs. Source URLs for the policy entries are stored in the workbooks.

## Code Map

- `code/01-data-preparation`: CHFS 2019 filtering, reconstruction, diagnostics, and stratified splitting.
- `code/common`: shared helpers, including focal-person-safe household enrollment counts.
- `code/02-probit-regression`: variable transformation, two Probit specifications, marginal effects, and descriptive statistics.
- `code/03-baseline-model-selection`: 30-case screening prompt construction, three-seed API execution, and metric aggregation.
- `code/04-prompt-engineering`: DKI prompt construction and burden-ratio analysis.
- `code/05-teacher-model-inference`: Claude Sonnet 4.5 teacher inference and output evaluation.
- `code/06-results-consolidation`: teacher-output consolidation, teacher-error regeneration, and SFT dataset construction.
- `code/07-knowledge-distillation`: final LoRA/SFT training, data-scale runs, inference, and evaluation.
- `code/08-validation-experiments`: preprocessing and DKI prompt construction for CHFS 2017, CFPS 2018, CHIP 2018, and CLDS 2018.
- `code/08-closeai-model-evaluation`: the unified closed-model benchmark runner and tests.
- `code/09-bootstrap-evaluation`: paired stratified bootstrap confidence intervals and summary generation.
- `code/10-expert-evaluation`: analysis of the anonymized expert ratings included in this package.
- `code/11-figure-generation`: component-gain, supplementary error-profile, and training-loss figure generation from aggregate inputs.
- `code/scripts`: shared environment loading for training and inference wrappers.

## Quick Validation

Create an isolated Python environment, install `requirements.txt`, and run:

```bash
python -m compileall -q code
(python -m pytest -q tests)
(cd code/08-closeai-model-evaluation && python -m pytest -q tests)
(cd code/09-bootstrap-evaluation && python -m pytest -q tests)
```

These checks use only the small synthetic JSON fixtures under the test directories. Full experiment reproduction additionally requires the controlled-access datasets, API access for closed models, the Qwen 3.5-35B-A3B base model, and eight GPUs for the reported global batch size.

Chinese strings in the code are retained where they are part of survey values, pension-policy fields, label protocols, or the Chinese prompts used in the experiments. They must not be translated during reproduction because doing so changes model inputs or parser behavior.
