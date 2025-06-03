import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from preprocessing import get_features_and_target

sns.set_style("darkgrid")

train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")
X_train, y_train = get_features_and_target(train_df)
X_dev, y_dev = get_features_and_target(dev_df)
X_dev = X_dev[X_train.columns]

# Ziel & Features vorbereiten
X_all = pd.concat([X_train, X_dev])
y_all = pd.concat([y_train, y_dev])

# t-SNE Reduktion (2D)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_all)

# Zielwerte in 4 Gruppen einteilen (z. B. Low, Medium, High, Very High)
target_groups = pd.qcut(y_all, q=4, labels=["Low", "Medium", "High", "Very High"])

# Plot vorbereiten
tsne_df = pd.DataFrame(X_tsne, columns=["TSNE-1", "TSNE-2"])
tsne_df = tsne_df.reset_index(drop=True)
target_groups = target_groups.reset_index(drop=True)
tsne_df["Group"] = target_groups


plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=tsne_df,
    x="TSNE-1",
    y="TSNE-2",
    hue="Group",
    palette="Set2",
    s=60
)
plt.title("t-SNE Visualization (grouped target values)")
plt.legend(title="Target Group")
plt.tight_layout()
plt.savefig("results_comparison/tsne_by_target_groups.png")
plt.close()

print("✅ t-SNE Plot mit gruppierten Zielwerten gespeichert unter: results_comparison/tsne_by_target_groups.png")
