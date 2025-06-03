from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocessing import get_features_and_target

train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")
X_train, y_train = get_features_and_target(train_df)
X_dev, y_dev = get_features_and_target(dev_df)
X_dev = X_dev[X_train.columns]

# tsne benötigt keine Zielvariable – nur die Features
X_all = pd.concat([X_train, X_dev])
y_all = pd.concat([y_train, y_dev])

# Reduziere auf 2 Dimensionen
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_all)

# Als DataFrame für Plot
tsne_df = pd.DataFrame(X_tsne, columns=["TSNE-1", "TSNE-2"])
tsne_df["Target"] = y_all.values

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x="TSNE-1", y="TSNE-2", hue="Target", palette="coolwarm", s=50)
plt.title("t-SNE Visualization (colored by target value)")
plt.tight_layout()
plt.savefig("results_comparison/tsne_plot_regression.png")
plt.close()
