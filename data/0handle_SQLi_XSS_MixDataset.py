import pandas as pd

in_file = './SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv'
df = pd.read_csv(in_file)

df['label'] = df['Normal'].apply(lambda x: 0 if x == 1 else 1)

df['text'] = df['Sentence']

required_columns = ['label', 'text']
df = df[required_columns]
df = df[df['text'].apply(lambda x: isinstance(x, str))]
# 把label中的nan删掉，并转换为Int
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])
df = df[~df['label'].isin([float('inf'), -float('inf')])]
df['label'] = df['label'].astype(int)


df.to_csv(in_file, index=False)
