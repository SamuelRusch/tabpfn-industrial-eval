import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit,  StratifiedShuffleSplit  

data_path = 'data/'

df = pd.read_csv(data_path + 'Data_RSW.csv')
df = df[df["Comments"] != "Communication error"]
df = df.drop_duplicates(subset="Sample ID", keep="last")


# Source - https://stackoverflow.com/a/54814482
# Posted by Negative Correlation, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-19, License - CC BY-SA 4.0

# --- First split: 80% train+dev, 20% test ---
sss1 = StratifiedShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
train_dev_idx, test_idx = next(sss1.split(df["Sample ID"], df["Category"]))

train_dev_groups = df.iloc[train_dev_idx]["Sample ID"]
test_groups      = df.iloc[test_idx]["Sample ID"]

train_dev = df[df["Sample ID"].isin(train_dev_groups)]
test      = df[df["Sample ID"].isin(test_groups)]

# --- Second split: 75% train, 25% dev (→ 20% of total) ---
groups_td = train_dev.groupby("Sample ID")["Category"].first().reset_index()

sss2 = StratifiedShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
train_idx, dev_idx = next(sss2.split(groups_td["Sample ID"], groups_td["Category"]))

train_groups = groups_td.iloc[train_idx]["Sample ID"]
dev_groups   = groups_td.iloc[dev_idx]["Sample ID"]

train = train_dev[train_dev["Sample ID"].isin(train_groups)]
dev   = train_dev[train_dev["Sample ID"].isin(dev_groups)]

# --- Save ---
train.to_csv(data_path + "train_data.csv", index=False)
dev.to_csv(data_path + "development_data.csv", index=False)
test.to_csv(data_path + "test_data.csv", index=False)











