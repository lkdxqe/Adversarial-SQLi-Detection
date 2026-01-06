import pandas as pd
import chardet


def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def read_and_rename_columns(file_path):
    # 检测文件编码
    encoding = detect_encoding(file_path)

    # 读取CSV文件
    df = pd.read_csv(file_path, encoding=encoding)

    # 打印列名以进行检查
    print(f"Columns in {file_path}: {df.columns.tolist()}")

    # 重命名列
    df.rename(columns={'Query': 'text', 'Label': 'label'}, inplace=True)

    # 确保只保留需要的列
    required_columns = ['label', 'text']
    df = df[required_columns]
    df = df[df['text'].apply(lambda x: isinstance(x, str))]
    # 把label中的nan删掉，并转换为Int
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df = df[~df['label'].isin([float('inf'), -float('inf')])]
    df['label'] = df['label'].astype(int)
    df.to_csv(file_path, index=False)

    df_label_1 = df[df['label'] == 1]
    sample_size = 20000
    df_sampled = df_label_1.sample(n=sample_size, random_state=42)
    df_sampled.to_csv(file_path[:-4]+"_split.csv", index=False)
    return


if __name__ == "__main__":
    read_and_rename_columns("./clean_sql_dataset.csv")
    # read_and_rename_columns("./SQL_Dataset.csv")