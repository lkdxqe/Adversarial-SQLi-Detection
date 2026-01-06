import pandas as pd
import random


def load_csv(file_path):
    # 读取CSV文件，确保label为int类型，text为str类型
    df = pd.read_csv(file_path, dtype={'label': int, 'text': str})
    df = df[['label', 'text']]
    return df


def duplicate_and_mix_csv(csv_file_1, csv_file_2, target_size, target_size2=0):
    # 加载CSV文件
    df1 = load_csv(csv_file_1)
    df2 = load_csv(csv_file_2)

    # 随机复制第一个CSV文件中的条目到target_size大小
    if target_size > len(df1):
        df1_expanded = df1.sample(n=target_size, replace=True, random_state=random.randint(1, 1000))
    else:
        df1_expanded = df1.sample(n=target_size, random_state=random.randint(1, 1000))

    if target_size2 != 0 and len(df2) > target_size2:
        df2 = df2.sample(n=target_size2, random_state=random.randint(1, 1000))

    # 将两个数据框混合在一起
    combined_df = pd.concat([df1_expanded, df2]).sample(frac=1, random_state=random.randint(1, 1000)).reset_index(
        drop=True)

    return combined_df


# 从csv_file_1中采样10000万条（重复采样），和csv_file_2混合在一起
name = 'clcnn_rl_rnn_advr_1'
csv_file_1 = './combation/' + name + '.csv'
csv_file_2 = './split_dataset/train.csv'
# csv_file_2 = './combation/ann_advr_2/train.csv'
target_size_1 = 10000  # 复制后的目标条目数量
target_size2 = 30000
# 调用函数并保存结果
combined_df = duplicate_and_mix_csv(csv_file_1, csv_file_2, target_size_1,target_size2=target_size2)
combined_df.to_csv('./combation/' + name + '/train.csv', index=False)
