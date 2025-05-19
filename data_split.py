import pandas as pd
from sklearn.model_selection import train_test_split

data_path = 'data/'

df = pd.read_csv(data_path + 'Data_RSW.csv')

df = df.drop_duplicates(subset=["Sample ID"], keep="last")

df["Thickness A+B (mm)"] = df["Thickness A (mm)"] + df["Thickness B (mm)"]
threshold = df["Thickness A+B (mm)"].quantile(0.99)
df = df[df["Thickness A+B (mm)"] <= threshold]

train_dev_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

train_data, dev_data = train_test_split(train_dev_data, test_size=0.2, random_state=42)

train_data.to_csv(data_path + "train_data.csv", index=False)
dev_data.to_csv(data_path + "development_data.csv", index=False)
test_data.to_csv(data_path + "test_data_DO_NOT_USE_BEFORE_FINAL.csv", index=False)


