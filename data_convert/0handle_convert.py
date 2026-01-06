import pandas as pd


def remove_invalid_rows(input_csv):
    # 读取CSV文件
    # 可是，载荷内部也可能会出现\n
    df = pd.read_csv(input_csv)

    # 过滤掉"text"列中包含"can not parsed to a valid tree"的行
    df_filtered = df[~df['text'].str.contains("can not parsed to a valid tree", na=False)]

    # 将过滤后的数据写入新的CSV文件
    df_filtered.to_csv(input_csv, index=False)


input_csv = ['sqli_convert.csv', 'sqliv2_convert.csv', 'SQLiV3_convert.csv','cnn_sql_convert.csv',
             'Modified_SQL_Dataset_convert.csv', 'clean_sql_dataset_split_convert.csv']
dir = ['convert_clcnn','convert_textcnn']

for d in dir:
    for in_csv in input_csv:
        file_path = "./" + d + "/" + in_csv
        print(file_path)
        remove_invalid_rows(file_path)
