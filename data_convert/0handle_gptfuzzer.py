import pandas as pd
import random
import csv
import urllib.parse

# 定义输入的txt文件名和输出的csv文件名
input_txt_file = './gptfuzzer_dataset.txt'
output_csv_file = 'gptfuzzer_dataset.csv'
target_size_1 = 120000  # 读取样本的数量
target_size_2 = 60000  # 最后保留的数量
# 初始化一个空的列表来存储样本数据
data = []

# 读取txt文件的每一行，并添加到数据列表中
with open(input_txt_file, 'r', encoding='utf-8') as file:
    count = 0
    for line in file:
        line = line.strip()  # 去除行末的换行符和多余的空格
        if line:  # 确保行不是空的
            data.append({'label': 1, 'text': line})
            print(line)
            count += 1
            print(count)
            if count > target_size_1:
                break

# 从数据中随机选择30000条样本
sample_data = random.sample(data, target_size_2)

# 将数据转换为pandas DataFrame
df = pd.DataFrame(sample_data)


def decode_url(encoded_str, encoding='latin-1'):
    decoded = urllib.parse.unquote_plus(encoded_str, encoding=encoding)
    while decoded != encoded_str:
        encoded_str = decoded
        decoded = urllib.parse.unquote_plus(encoded_str, encoding=encoding)
    decoded = decoded.lower()
    return decoded


df['text'] = df['text'].apply(decode_url)

# 将DataFrame保存为csv文件
df.to_csv(output_csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f'Data has been successfully written to {output_csv_file}')
