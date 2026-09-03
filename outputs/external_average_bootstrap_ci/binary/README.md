# External Average Bootstrap CI

Configuration: binary Type F1, B=10,000, and seed=42.

Method: bootstrap samples are stratified by the true terminal label within each external dataset, with shared resampling indices across models. Metrics are computed within each resample and then averaged with equal weight across the four datasets.

## Composite F1 model CI

| rank | model | observed | 95% CI |
| ---: | --- | ---: | --- |
| 1 | claude_opus_4_6 | 0.7724 | [0.7461, 0.7974] |
| 2 | gemini_3_1_pro_preview | 0.7631 | [0.7373, 0.7876] |
| 3 | flexpension | 0.7549 | [0.7256, 0.7818] |
| 4 | correct_only | 0.7378 | [0.7080, 0.7646] |
| 5 | claude_sonnet_4_6 | 0.7324 | [0.7044, 0.7578] |
| 6 | qwen3_7_plus | 0.7216 | [0.6951, 0.7467] |
| 7 | minimax_m2_5 | 0.7004 | [0.6762, 0.7236] |
| 8 | gpt_5_4 | 0.6982 | [0.6793, 0.7166] |
| 9 | claude_sonnet_4_5 | 0.6954 | [0.6678, 0.7210] |
| 10 | deepseek_v4_pro | 0.6950 | [0.6680, 0.7194] |
| 11 | qwen_zs | 0.6378 | [0.6089, 0.6660] |

## FlexPension Composite F1 deltas

| baseline | delta | 95% CI | p(delta <= 0) |
| --- | ---: | --- | ---: |
| qwen_zs | 0.1170 | [0.0793, 0.1534] | 0.0000 |
| claude_sonnet_4_5 | 0.0595 | [0.0293, 0.0895] | 0.0001 |
| qwen3_7_plus | 0.0333 | [0.0019, 0.0641] | 0.0198 |
| claude_sonnet_4_6 | 0.0225 | [-0.0054, 0.0502] | 0.0591 |
| correct_only | 0.0171 | [-0.0111, 0.0446] | 0.1180 |
| gemini_3_1_pro_preview | -0.0083 | [-0.0369, 0.0197] | 0.7284 |
| claude_opus_4_6 | -0.0176 | [-0.0462, 0.0105] | 0.8843 |
