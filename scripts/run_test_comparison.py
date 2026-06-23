"""Compare all models on the held-out test set.

Fits a tuned Random Forest, a tuned XGBoost, a default TabPFN, and the
physics-guided residual XGBoost on the combined train+development data, then
reports test-set RMSE and writes comparison plots.
"""

from rsw import config, data, models, plotting
from rsw.evaluation import rmse
from rsw.preprocessing import get_features_and_target


def main():
    # Standard feature set, shared by Random Forest, XGBoost, and TabPFN.
    combined_df, test_df = data.load_combined_and_test()
    X_train, y_train = get_features_and_target(combined_df)
    X_test, y_test = get_features_and_target(test_df)

    rf_model = models.random_forest()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_rmse = rmse(y_test, rf_pred)

    xgb_model = models.xgboost()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = rmse(y_test, xgb_pred)

    tabpfn_model = models.tabpfn()
    tabpfn_model.fit(X_train, y_train)
    tabpfn_pred = tabpfn_model.predict(X_test)
    tabpfn_rmse = rmse(y_test, tabpfn_pred)

    # Physics-guided residual model. F_pull_physical stays in the feature set
    # here (the standard extractor is used on the physics-augmented data), and
    # the residual target is computed separately from the physical estimate.
    combined_phys_df, test_phys_df = data.load_combined_and_test(physical=True)
    X_phys_train, y_true_phys_train = get_features_and_target(combined_phys_df)
    y_phys_train = combined_phys_df[config.PHYSICAL_PULL_COLUMN]
    y_bias_train = y_true_phys_train - y_phys_train

    X_phys_test, y_true_phys_test = get_features_and_target(test_phys_df)
    y_phys_test = test_phys_df[config.PHYSICAL_PULL_COLUMN]

    xgb_phys_model = models.xgboost_physics()
    xgb_phys_model.fit(X_phys_train, y_bias_train)
    bias_pred_test = xgb_phys_model.predict(X_phys_test)
    f_pull_corrected_test = y_phys_test + bias_pred_test
    xgb_phys_rmse = rmse(y_true_phys_test, f_pull_corrected_test)

    print("\nRMSE on test data:")
    print(f"Random Forest: {rf_rmse:.2f}")
    print(f"XGBoost: {xgb_rmse:.2f}")
    print(f"TabPFN: {tabpfn_rmse:.2f}")
    print(f"XGBoost with Physics: {xgb_phys_rmse:.2f}")

    config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    names = ["Random Forest", "XGBoost", "TabPFN", "XGBoost Physics"]
    rmse_values = [rf_rmse, xgb_rmse, tabpfn_rmse, xgb_phys_rmse]
    results = [
        {"name": "Random Forest", "color": "green", "y_true": y_test, "y_pred": rf_pred},
        {"name": "XGBoost", "color": "red", "y_true": y_test, "y_pred": xgb_pred},
        {"name": "TabPFN", "color": "orange", "y_true": y_test, "y_pred": tabpfn_pred},
        {
            "name": "XGBoost Physics",
            "color": "black",
            "y_true": y_true_phys_test,
            "y_pred": f_pull_corrected_test,
        },
    ]

    plotting.plot_rmse_bar(names, rmse_values, config.COMPARISON_DIR / "model_rmse_test.png")
    plotting.plot_scatter(results, config.COMPARISON_DIR / "scatter_pred_vs_true.png")
    plotting.plot_residual_hist(results, config.COMPARISON_DIR / "residual_histogram.png")
    plotting.plot_error_cdf(results, config.COMPARISON_DIR / "cdf_absolute_errors.png")


if __name__ == "__main__":
    main()
