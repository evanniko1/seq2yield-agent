# Reproducibility gaps & risks

_Generated 2026-06-04T18:42:18.159348+00:00 from `seq2yield.zip` (sha256 `ae1b981d3bd9561a...`)._

- Archive members: 698 (349 macOS/junk skipped, 279 real files).

## Findings

1. SPLIT LOCATION CORRECTION: provided splits live in _saved/iteration_N/{_working_set,_heldout_set}.csv (5 iterations: iteration_1, iteration_2, iteration_3, iteration_4, iteration_5), NOT _saved/saved_sets/ as the deposit README implied. Each iteration = one Monte-Carlo CV repeat with its own held-out set.
2. 8 notebooks contain Colab/Drive path assumptions: 1_kmers.ipynb, 6_Yeast.ipynb, 5_Plot_Results.ipynb, 3_Train_Convolutional_Networks.ipynb, 4_ExplainableAI.ipynb, 2_Train_Non_Deep_Regressors.ipynb, hyper_opt_DL.ipynb, hyper_opt_ML.ipynb.

## Notebook policy

All notebooks above are **seed material only** and were copied read-only to `archive_notebooks_readonly/`. They are never executed in the pipeline (docs/PROJECT_SPEC.md section 11; tests/test_notebooks_not_executed.py).
