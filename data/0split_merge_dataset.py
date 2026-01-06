import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
input_csv = 'merge_dataset.csv'
df = pd.read_csv(input_csv)

# 6 2 2

train_val, test = train_test_split(df, test_size=0.4, random_state=42)

# 进一步划分临时集为验证集和测试集
val, test = train_test_split(test, test_size=0.5, random_state=42)

# 可选：输出各个数据集的大小
print(f"训练集大小：{len(train_val)}")
print(f"验证集大小：{len(val)}")
print(f"测试集大小：{len(test)}")

# 保存划分后的数据集为新的 CSV 文件
train_val.to_csv('./split_dataset/train.csv', index=False)
val.to_csv('./split_dataset/val.csv', index=False)
test.to_csv('./split_dataset/test.csv', index=False)
