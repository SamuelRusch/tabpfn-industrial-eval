import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# Daten laden
train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")

# Zielspalte
target_col = "Category"

# Zähle Klassenanteile in Prozent
train_counts = train_df[target_col].value_counts(normalize=True) * 100
dev_counts = dev_df[target_col].value_counts(normalize=True) * 100

# Kombiniere in ein gemeinsames DataFrame
combined_df = pd.DataFrame({
    "Train": train_counts,
    "Dev": dev_counts
}).fillna(0)  # falls eine Klasse z. B. in dev fehlt

# Transponieren für Barplot
plot_df = combined_df.T

# Plot
plt.figure(figsize=(8, 5))
plot_df.plot(kind="bar", stacked=False, colormap="Set2")
plt.ylabel("Anteil [%]")
plt.title("Verteilung der Klassen in 'Category' (Train vs. Dev)")
plt.xticks(rotation=0)
plt.legend(title="Klasse")
plt.tight_layout()
plt.savefig("statistics/class_distribution_category.png")
plt.close()

print("Klassenverteilung gespeichert unter: results_comparison/class_distribution_category.png")
