# Bias, Variance and an Unbiased Cross-Validation Score

**ECON 5200 · Causal Machine Learning & Applied Analytics · Lab 15**

## Objective

I measured how far a cross-validation score drifts from the truth when the same score is used to choose the model, and tested what nested cross-validation can and cannot fix.

## Methodology

- Simulated 80 points from a known curve, sin(1.4x) + 0.25x, with noise of standard deviation 0.45, and cross-validated polynomials of degree 1 to 12 with 5 folds.
- Kept the degree with the smallest CV error, as an analyst would, then scored that fit on 20,000 new points from the same simulation to get its true error.
- Reran the search with 2, 4, 8 and 12 candidate degrees to see how the optimism changes with the size of the search.
- Ran nested cross-validation, with the whole degree search inside each of five outer folds, so the reported score was never used to make the choice.
- Split the expected error into bias squared, variance and noise by refitting six degrees on 200 fresh samples.
- Wrote my own k-fold cross-validation function for any scikit-learn model on a table with named columns, and checked it against the lab.
- Used a `model_selection.py` module that returns a nested score with every choice it makes.

## Key Findings

**The winning CV score was too optimistic.** The search chose degree 5 with a CV error of 0.1853, but its true error on 20,000 new points was 0.2330, an optimism of +0.0477. The score was low partly because it was the luckiest of twelve.

**Nested CV removes the selection bias, not the sample's luck.** It estimated 0.1910, closer to the truth than 0.1853. The rest of the gap comes from the 80 points themselves, whose noise (0.1792) was below the population's (0.2025).

**The reported score was below the noise floor.** The noise alone was 0.2025, which no model can beat, so a reported 0.1853 can only be an optimistic estimate. Degree 5 had the smallest expected error of the six degrees, 0.2291, most of it noise. Degree 12's variance of 0.8435 shows what overfitting costs.

**My own function matches the lab.** It reproduced degree 5's CV error of 0.1853 with 5 folds, and gave 0.1921 with 10 folds.

## Repository Contents

| File | Description |
|---|---|
| `lab-ch15-diagnostic.ipynb` | Full lab notebook with rendered output |
| `src/model_selection.py` | A plain k-fold score for a degree fixed in advance, and a degree search that reports its nested CV score |

## Data

- All data are simulated with fixed seeds (`RANDOM_STATE = 42` for the analyst's sample and folds).
