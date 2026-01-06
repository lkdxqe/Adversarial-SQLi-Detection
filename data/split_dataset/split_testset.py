import pandas as pd
import numpy as np

# 读取 CSV 文件
file_path = 'test.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)
df = df[df["label"] == 1]
sampled_df = df.sample(n=1000, random_state=42)
sampled_df.to_csv('light_test.csv', index=False)
