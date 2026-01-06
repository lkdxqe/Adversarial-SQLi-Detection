import pandas as pd
import chardet


def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def sample_my(df, sample_size):
    df_label_0 = df[df['label'] == 0]
    df_label_1 = df[df['label'] == 1]
    sample_size_label_0 = min(int(sample_size * 0.6), len(df_label_0))
    sample_size_label_1 = min(int(sample_size * 0.4), len(df_label_1), int(sample_size_label_0 * 2 / 3))
    df_label_0_sampled = df_label_0.sample(n=sample_size_label_0, random_state=1)
    df_label_1_sampled = df_label_1.sample(n=sample_size_label_1, random_state=1)
    df_sampled = pd.concat([df_label_0_sampled, df_label_1_sampled])
    df_sampled = df_sampled.sample(frac=1, random_state=1).reset_index(drop=True)
    return df_sampled


def merge_csv_files(file_paths, output_path, sample_size=30000):
    dataframes = []
    required_columns = ['label', 'text']

    for file_path in file_paths:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        df = df[required_columns]  # 只保留所需的列
        df = df[df['text'].apply(lambda x: isinstance(x, str))]
        # 把label中的nan删掉，并转换为Int
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])
        df = df[~df['label'].isin([float('inf'), -float('inf')])]
        df['label'] = df['label'].astype(int)

        df = sample_my(df, sample_size)

        num_benign = (df['label'] == 0).sum()
        num_malicious = (df['label'] == 1).sum()
        print(".........................")
        print(f"{file_path} ,encoding: {encoding}")
        print(f"{file_path} contains {df.shape[0]} rows")
        print(f"num_benign:{num_benign}")
        print(f"num_malicious:{num_malicious}")
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    num_benign = (merged_df['label'] == 0).sum()
    num_malicious = (merged_df['label'] == 1).sum()
    print("......................")
    print(f"num_benign:{num_benign}")
    print(f"num_malicious:{num_malicious}")
    merged_df.to_csv(output_path, index=False, encoding='utf-8')


# 'sqli.csv','./sqliv2.csv' 来自于https://github.com/ajinmathew/SQL-data
# './cnn_sql.csv' 来自于https://github.com/fishyyh/CNN-SQL
# './sqli' 来自于Kaggle https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset?resource=download
# './Modified_SQL_Dataset.csv' 来自于Kaggle https://www.kaggle.com/datasets/sajid576/sql-injection-dataset/data
# "./clean_sql_dataset.csv" 来自于https://www.kaggle.com/datasets/gambleryu/biggest-sql-injection-dataset?resource=download
# './SQL_Dataset.csv' 来自于https://www.kaggle.com/datasets/kholoodsalah/sql-injection-dataset
# './SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv'来自于https://www.kaggle.com/datasets/alextrinity/sqli-xss-dataset

# "./clean_sql_dataset.csv"没有再使用，因为里面有很多不合理的分类
file_paths = ['./cnn_sql.csv', './csic.csv', 'sqli.csv', './sqliv2.csv', './SQLiV3.csv', './Modified_SQL_Dataset.csv',
              './SQL_Dataset.csv',
              './SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv']
output_path = './merge_dataset.csv'

# 下面的包含混淆后的SQL数据集
# './gptfuzzer_dataset.csv' 来自于https://github.com/HongliangLiang/gptfuzzer
# './sqliv5.csv' 来自于https://github.com/nidnogg/sqliv5-dataset
# file_paths = ['./cnn_sql.csv', './csic.csv', './gptfuzzer_dataset.csv', 'sqli.csv', './sqliv2.csv']
# output_path = './merge_dataset.csv'

merge_csv_files(file_paths, output_path)
print(f"Files have been merged and saved to {output_path}")
