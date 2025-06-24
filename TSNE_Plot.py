import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Daten einlesen
train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")

# 2. Set-Spalte hinzufügen, falls du später unterscheiden willst
train_df['Set'] = 'train'
dev_df['Set'] = 'dev'

# 3. DataFrames kombinieren
df = pd.concat([train_df, dev_df], ignore_index=True)

# 4. Kategorische Features encoden (z.B. "Material")
if df['Material'].dtype == object or str(df['Material'].dtype).startswith('category'):
    df['Material_enc'] = LabelEncoder().fit_transform(df['Material'])
else:
    df['Material_enc'] = df['Material']

# 5. Features auswählen (ohne Target/ohne ID/ohne Kommentare)
features = [
    'Pressure (PSI)', 'Welding Time (ms)', 'Angle (Deg)', 'Force (N)', 'Current (A)',
    'Thickness A (mm)', 'Thickness B (mm)', 'Material_enc'
]
# Manche Spalten könnten fehlen, also vorher checken:
features = [col for col in features if col in df.columns]

# 6. Fehlende Werte behandeln (z.B. mit Mittelwert füllen)
X = df[features].fillna(df[features].mean())

# 7. Standardisieren
X = StandardScaler().fit_transform(X)

# 8. t-SNE anwenden
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

# 9. Plotten, nach 'Category' eingefärbt
plt.figure(figsize=(9,7))
categories = df['Category'].astype(str).unique()
for cat in categories:
    idx = df['Category'] == cat
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=cat, alpha=0.7)
plt.legend()
plt.title("t-SNE der kombinierten Daten (train+dev), nach 'Category' eingefärbt")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()