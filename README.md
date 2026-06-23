# Pull-test force prediction for resistance spot welding

Resistance spot welding joins thousands of points on every car body, and the
strength of each weld matters for structural integrity. This project predicts
the tensile-shear pull-test force of a weld from its process parameters and
asks a specific question: how does TabPFN, a pretrained foundation model for
tabular data, compare to tuned tree models and to a physics-guided hybrid on a
small industrial dataset?

The full study is in [paper.pdf](paper.pdf).

## Results

Test-set RMSE on 149 held-out welds (lower is better):

| Model                   | Test RMSE [N] |
| ----------------------- | ------------- |
| Physics-guided XGBoost  | 226.75        |
| Random Forest           | 226.98        |
| TabPFN                  | 227.81        |
| XGBoost                 | 240.40        |

Main findings:

- TabPFN matches the tuned tree models without any hyperparameter tuning or
  feature engineering. It is fit and queried in a single forward pass.
- The physics-guided residual XGBoost has the lowest RMSE and the fewest large
  errors, but needs the extra feature engineering and the analytical prior.
- A purely physical prediction is much worse: RMSE of 524 N using the measured
  nugget diameter and 708 N using the thickness-based approximation. The value
  of the physics term comes from combining it with a data-driven correction,
  not from using it alone.
- The largest errors concentrate in the rare "Explode" category (13 training
  and 8 development samples), where molten-metal expulsion changes the nugget
  geometry in a non-systematic way.
- The dominant feature is Welding Time, followed by Angle and Pressure;
  Welding Current contributes little within the observed parameter range.

## Approach

Four models predict the pull-test force:

- Random Forest and XGBoost, both tuned with Hyperopt (TPE) under 5-fold
  cross-validation on the combined train+development data.
- TabPFN in its default configuration, with no task-specific tuning.
- A physics-guided residual XGBoost.

The physics-guided model uses domain knowledge as a low-fidelity prior. The
nugget diameter is approximated from the minimum sheet thickness with
`d = 4 * sqrt(t_min)`, and the interfacial failure load is estimated as
`F = (pi / 4) * d^2 * tau` with `tau = 292 MPa`. Instead of predicting the
pull force directly, the model learns the residual between the measured force
and this physical estimate, and the final prediction adds the learned residual
back to the physical term.

## Repository structure

```
rsw/                package: config, preprocessing, physics, data,
                    models, evaluation, plotting
scripts/            run_test_comparison.py, build_physics_feature.py
tests/              pytest unit tests
data/               train/development/test splits (CSV)
model_training/     per-model tuning scripts, tuned parameters, CV plots
results_*/          feature importance and t-SNE outputs
statistics/         dataset statistics plots
diagrams/           physics-calculation comparison figures
paper.pdf           the full report
```

## Getting started

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test-set comparison (fits all four models and writes the comparison
plots to `model_training/model_comparison/`):

```
python scripts/run_test_comparison.py
```

Rebuild the physics-augmented data splits:

```
python scripts/build_physics_feature.py
```

Run the unit tests:

```
pytest
```

## Results in detail

Feature importance from the tuned XGBoost (gain-based) ranks Welding Time well
ahead of the rest, followed by Angle and Pressure, then Force and the sheet
thicknesses. Welding Current ranks low, which likely reflects its limited
variation in this dataset rather than its physical role.

The CDF of absolute errors separates the models more clearly than RMSE alone.
The physics-guided model has the steepest curve and the thinnest upper tail,
meaning a higher share of low-error predictions and fewer large deviations.

The high-error cases are dominated by the "Explode" category. The five largest
test errors range from 452 N to 1241 N, and four of the five are Explode or
extreme-condition welds, where expulsion of molten metal reduces the effective
nugget volume and makes the force hard to predict.

## Dataset

Resistance Spot Welding Insights, Dominguez et al. (2025), Data in Brief 59,
111373, DOI [10.1016/j.dib.2025.111373](https://doi.org/10.1016/j.dib.2025.111373).
The dataset has about 495 welds. Only the tabular process and test data are
used here; the thermographic and surface images are not. The dataset is the
property of its authors and is used here for attribution and analysis only; it
is not relicensed by this repository.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)).

## Author

Samuel Rusch, Hochschule Aalen.
