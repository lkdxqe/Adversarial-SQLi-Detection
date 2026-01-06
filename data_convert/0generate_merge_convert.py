"""
将每个模型对抗生成的convert数据集合并
得到convert_train val test .csv
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

target_dir = "./convert_textcnn/"
# 获取当前目录下所有的 CSV 文件
csv_files = [target_dir+f for f in os.listdir(target_dir) if f.endswith('.csv')]

# 读取所有的 CSV 文件并合并为一个 DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# 按照 6:2:2 的比例分割数据集
train_df, temp_df = train_test_split(combined_df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 保存分割后的数据集到 CSV 文件
train_df.to_csv(target_dir+'convert_train.csv', index=False)
val_df.to_csv(target_dir+'convert_val.csv', index=False)
test_df.to_csv(target_dir+'convert_test.csv', index=False)

print("Datasets created: convert_train.csv, convert_val.csv, convert_test.csv")
