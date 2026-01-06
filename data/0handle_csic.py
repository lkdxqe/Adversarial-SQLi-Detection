import pandas as pd
import os
import urllib.parse

data_dir = "."
# 读取Excel文件
df = pd.read_csv(os.path.join(data_dir, 'csic_raw.csv'), sep=None, header=0, encoding='utf-8', engine='python')

# 提取所需的列
df = df[['Class', 'POST-Data', 'GET-Query']]

# 修改Class列的值并重命名为label
df['label'] = df['Class'].apply(lambda x: 0 if x == 'Valid' else 1)
df.drop(columns=['Class'], inplace=True)

# 提取POST-Data和GET-Query中的值并重命名为text
df['text'] = df.apply(lambda row: row['POST-Data'] if pd.notna(row['POST-Data']) else row['GET-Query'], axis=1)
df.drop(columns=['POST-Data', 'GET-Query'], inplace=True)

# 剔除text为空的行
df.dropna(subset=['text'], inplace=True)


# 进行url解码
# URL循环解码函数
def decode_url(encoded_str, encoding='latin-1'):
    decoded = urllib.parse.unquote_plus(encoded_str, encoding=encoding)
    while decoded != encoded_str:
        encoded_str = decoded
        decoded = urllib.parse.unquote_plus(encoded_str, encoding=encoding)
    decoded = decoded.lower()
    return decoded


df['text'] = df['text'].apply(decode_url)
print(df['text'])

# # 调整每一行的text值长度，现在调整长度是在训练的时候调整
# length = 200
# df['text'] = df['text'].apply(lambda x: (x[:length] if len(x) > length else x.ljust(length, '0')))

# 保存到新的Excel文件
df.to_csv(os.path.join(data_dir, 'csic.csv'))
