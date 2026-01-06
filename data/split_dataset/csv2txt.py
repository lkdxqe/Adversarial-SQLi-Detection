import pandas as pd
import numpy as np

# 读取 CSV 文件
csv_file = 'test.csv'  # CSV 文件路径
df = pd.read_csv(csv_file)

# 提取 'text' 列
df = df.loc[df["label"] == 1]
text_data = df['text'].dropna()  # 去除缺失值


# 转义特殊字符
def escape_special_chars(text):
    if isinstance(text, str):
        text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    return text


# 保存到 TXT 文件
txt_file = 'test.txt'  # TXT 文件路径
with open(txt_file, 'w', encoding="utf-8") as file:
    for text in text_data:
        escaped_text = escape_special_chars(text)
        file.write(f"{escaped_text}\n")

# 对npy文件不进行转义
npy_file = 'test.npy'
escaped_text_data = [text for text in text_data]
np.save(npy_file, escaped_text_data)

for text in text_data:
    print(text)

print(f"Text data has been saved to {txt_file}")
