import pandas as pd

df = pd.read_csv("merge_dataset.csv", encoding="utf-8")
df_label_0 = df[df['label'] == 0]
df_label_1 = df[df['label'] == 1]
print(f"len(df_label_0):{len(df_label_0)}")
print(f"len(df_label_1):{len(df_label_1)}")