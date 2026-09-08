# MSc Thesis: Bias and Fairness in ML Models on Open-Source Datasets

An empirical investigation into how algorithmic bias shows up across different
datasets and classification algorithms, whether training-data composition
affects fairness independently of the model used, and how well a standard
mitigation technique (Reweighing) works in practice.

## What this project does

1. **Baseline experiments** — trains 4 algorithms (Logistic Regression,
   Decision Tree, Random Forest, XGBoost) on 3 datasets and measures both
   accuracy and 4 fairness metrics (statistical parity difference, equal
   opportunity difference, disparate impact, equalized odds difference).

2. **3-subset experiment** — for each dataset, trains every algorithm three
   times: once on the privileged group only, once on the unprivileged group
   only, and once on both combined. All three are tested on the same held-out
   set, to see whether training composition itself affects fairness.

3. **Reweighing mitigation** — applies AIF360's Reweighing pre-processing
   method to German Credit as a deep-dive case study, comparing fairness
   metrics before and after.

## Datasets

| Dataset | Domain | Protected attribute(s) | Source |
|---|---|---|---|
| German Credit | Loan approval | sex, age | AIF360 |
| Taiwan Credit Default | Credit default | sex | UCI |
| Folktables ACS Employment | Employment | sex | US Census (California, 2018) |

## Project structure

```
src/
  data_loading.py    # loads and cleans all 3 datasets into AIF360 objects
  pipeline.py        # training, fairness evaluation, subset experiment, Reweighing

notebooks/
  data_loading.ipynb        # inspect the 3 datasets
  pipeline_runs.ipynb       # baseline: run all 4 models on all 3 datasets
  subset_experiments.ipynb  # the 3-subset experiment
  reweighing.ipynb          # Reweighing case study on German Credit
  visualizations.ipynb      # all charts used in the write-up

results/
  *.csv    # saved metrics tables for every experiment
  *.png    # saved charts (accuracy vs. fairness, subset comparisons, Reweighing before/after)
```

## Running it

1. Install dependencies: `aif360`, `xgboost`, `folktables`, `ucimlrepo`,
   `pandas`, `scikit-learn`, `matplotlib`
2. Run `notebooks/pipeline_runs.ipynb` first (baseline results)
3. Then `notebooks/subset_experiments.ipynb` and `notebooks/reweighing.ipynb`
   (both depend on `src/pipeline.py`, not on each other)
4. `notebooks/visualizations.ipynb` reads from `results/*.csv` and can be run
   any time after the experiments above

## Key findings

- Fairness metrics don't move in one consistent direction across datasets —
  German Credit and Folktables disadvantage the unprivileged group, while
  Taiwan Credit Default shows the reverse pattern.
- More complex models (Random Forest, XGBoost) tend to score better on
  fairness metrics than simpler ones (Logistic Regression) in this pipeline —
  the opposite of what a simple "more accurate = less fair" assumption
  would predict.
- Reweighing's effect on German Credit varies by algorithm and by metric;
  it does not improve every metric uniformly across all 4 models.
