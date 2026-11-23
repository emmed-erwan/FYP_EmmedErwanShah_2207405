# Model Development Results and Comparison

This report is an index and comparison of **existing experimental evidence**. Internal held-out scores and external scores use different evaluation sets and must not be interpreted as interchangeable. The 24-permutation baseline is the original fleeing-versus-wandering classifier, not a fourth distinct saved classifier.

## A. Permutation experiment (fleeing vs wandering)

| Training permutations | Row-level accuracy | Video-level accuracy | Video balanced accuracy | Video F1 (fleeing) |
|---|---:|---:|---:|---:|
| 24 | 91.77% | 90.00% | 89.77% | 91.30% |
| 8 (mean) | 88.65% | 89.17% | 88.79% | 90.66% |
| 4 (mean) | 86.32% | 87.50% | 86.83% | 89.36% |

**Interpretation:** In the recorded experiment, using 24 fish-order permutations produced the strongest mean accuracy. The 8- and 4-permutation values are comparison-run means, not evaluations of separately archived model binaries. Their exact run-by-run metrics are in `02_Permutation_24_8_4/rf_permutation_comparison/all_runs.csv` and their predictions in `video_predictions.csv`.

## B. Class-composition experiments: internal video-level evaluation

| Model | Test videos | Accuracy | Balanced accuracy | Fleeing recall | Fleeing F1 |
|---|---:|---:|---:|---:|---:|
| Original fleeing vs wandering | 40 | 90.00% | 89.77% | 91.30% | 91.30% |
| Fleeing vs feeding | 28 | 89.29% | 70.00% | 100.00% | 93.88% |
| Fleeing vs combined non-fleeing | 45 | 100.00% | 100.00% | 100.00% | 100.00% |

The feeding-only internal test had **23 fleeing and 5 feeding videos**; feeding recall was **40%**. The combined model's 100% internal score should **not** be interpreted as perfect generalisation.

## C. Common unseen-video test (45 external clips)

| Model | Correct / 45 | Accuracy | Balanced accuracy | Fleeing recall | Fleeing F1 |
|---|---:|---:|---:|---:|---:|
| Original fleeing vs wandering | 30 | 66.67% | 70.90% | 88.24% | 66.67% |
| Fleeing vs feeding | 18 | 40.00% | 49.47% | 88.24% | 52.63% |
| Fleeing vs combined non-fleeing | 36 | **80.00%** | 78.15% | 70.59% | 72.73% |

Ground truth: **17 fleeing, 26 wandering, 2 feeding**. For the combined model, the confusion matrix was `[[24, 4], [5, 12]]` (actual rows and predicted columns: non-fleeing then fleeing). This represents **4 false positives** and **5 false negatives**. The 45 clips came from a longer recording, so they do not represent 45 independent sessions.

## D. Practical V2 long-recording trial

A 60-second recording manually labelled as containing no fleeing was segmented into short windows. The first two windows (0–8s, 8–16s) were classified as fleeing with scores **0.5577599829244116** and **0.569985985553286**. They are **two reported false-positive segments**, not two proven stress events. This result was user-reported; the original MP4, exported timeline and feature files are **not archived**.

## E. Where to find original evidence in the ZIP

| Folder | Original evidence |
|---|---|
| `01_Original_Model/rf_results/` | `metrics.csv`, `test_predictions.csv`, `confusion_matrix.png`, original `.joblib` |
| `02_Permutation_24_8_4/rf_permutation_comparison/` | `comparison_summary.csv`, `all_runs.csv`, `video_predictions.csv`, confusion matrices |
| `03_Feeding_Experiment/rf_fleeing_vs_feeding/` | Video/permutation metrics, predictions and confusion matrices |
| `04_Combined_Experiment/rf_fleeing_vs_combined/` | Video/permutation metrics, predictions and confusion matrices |
| `05_External_45_Clips/fish_external_evaluation_kit.zip` | All three saved models, common test features, labels, evaluation script and results |
| `06_AquaWatch_V1_V2/` | V1/V2 source archives and `split.py` |

## F. Conclusions and next evidence

The combined model performed best on the common external set (80%) despite a lower fleeing recall than the original model. The large difference between internal and external accuracy supports further independent-session validation, including continuous recordings. Record each experiment's training configuration, source-video split, random seed, dataset version and model hash. Save 4- and 8-permutation model binaries if retraining them, and collect screenshots plus timestamped V2 prediction exports.
