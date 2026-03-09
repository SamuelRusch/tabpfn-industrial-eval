import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit 

data_path = 'data/'

df = pd.read_csv(data_path + 'Data_RSW.csv')
df = df[df["Comments"] != "Communication error"]

# Source - https://stackoverflow.com/a/54814482
# Posted by Negative Correlation, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-19, License - CC BY-SA 4.0

splitter = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 42)
split = splitter.split(df, groups=df['Sample ID'])
intermediary_inds, test_inds = next(split)

intermediary = df.iloc[intermediary_inds]

splitter = GroupShuffleSplit(test_size=.25, n_splits=1, random_state = 42)
split = splitter.split(intermediary, groups=intermediary['Sample ID'])
train_inds, dev_inds = next(split)

train = intermediary.iloc[train_inds]
dev = intermediary.iloc[dev_inds]
test = df.iloc[test_inds]

train.to_csv(data_path + "train_data_classifier.csv", index=False)
dev.to_csv(data_path + "development_data_classifier.csv", index=False)
test.to_csv(data_path + "test_data_classifier.csv", index=False)

print(train["Category"].value_counts(normalize=True))
print(dev["Category"].value_counts(normalize=True))
print(test["Category"].value_counts(normalize=True))
