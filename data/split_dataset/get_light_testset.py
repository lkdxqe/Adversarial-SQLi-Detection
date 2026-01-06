import pandas as pd
import numpy as np

# 读取 CSV 文件
file_path = 'test.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 随机打乱数据行
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)


