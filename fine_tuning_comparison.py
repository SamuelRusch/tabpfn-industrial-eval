import argparse
import os
import glob
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import get_features_and_target

sns.set_style("darkgrid")

MODEL_CLASSES = {
    "decision_tree": "dt_model",
    "random_forest": "rf_model",
    "xgboost": "xgb_model",
    "tabpfn": "tabpfn_model"
}

VARIANTS = ["baseline", "tuned"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all models for a given type")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CLASSES.keys(), help="Model type")
    args = parser.parse_args()

    model_name = args.model
    model_prefix = MODEL_CLASSES[model_name]

    # Daten laden
    train_df = pd.read_csv("data/train_data.csv")
    dev_df = pd.read_csv("data/development_data.csv")
    X_train, y_train = get_features_and_target(train_df)
    X_dev, y_dev = get_features_and_target(dev_df)
    X_dev = X_dev[X_train.columns]

    results = []

    for variant in VARIANTS:
        model_dir = os.path.join("model_training", model_name, variant)
        pattern = os.path.join(model_dir, f"{model_prefix}_{variant}*.pkl")
        model_files = glob.glob(pattern)

        if not model_files:
            print(f"❌ No models found for {variant} in {model_name}")
            continue

        for model_path in model_files:
            model = joblib.load(model_path)
            preds = model.predict(X_dev)

            mse = mean_squared_error(y_dev, preds)
            rmse = mse ** 0.5
            r2 = r2_score(y_dev, preds)

            model_name_only = os.path.basename(model_path).replace(".pkl", "")
            results.append({"Model": model_name_only, "Variant": variant, "RMSE": rmse, "R2": r2})

    # Ergebnisse anzeigen
    if results:
        df = pd.DataFrame(results)
        print("\n📊 Results:")
        print(df.round(3))

        # Plot
        plt.figure(figsize=(10, 6))
        df_melted = df.melt(id_vars=["Model", "Variant"], value_vars=["RMSE", "R2"], var_name="Metric")
        sns.barplot(data=df_melted, x="Model", y="value", hue="Metric")
        plt.title(f"Performance Comparison - {model_name}")
        plt.ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        os.makedirs("results/fine_tuning", exist_ok=True)
        plot_path = f"results/fine_tuning/{model_name}_comparison.png"
        plt.savefig(plot_path)
        plt.show()
    else:
        print("⚠️ No results to display.")