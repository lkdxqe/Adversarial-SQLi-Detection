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
    df.to_csv(file_path, index=False)
    return


if __name__ == "__main__":
    read_and_rename_columns("./Modified_SQL_Dataset.csv")