import pandas as pd


def sample_and_save_csv(input_csv, output_csv, num_samples):
    print(input_csv)
    df = pd.read_csv(input_csv)

    # 检查样本数是否足够
    if len(df) < num_samples:
        print(f"样本数不足，只有 {len(df)} 个样本，无法抽取 {num_samples} 个样本。将保存整个数据集。")
        df.to_csv(output_csv, index=False)
    else:
        # 随机选择指定数量的样本
        sampled_df = df.sample(n=num_samples, random_state=42)  # 设置 random_state 以确保结果可重复

        # 保存选中的样本到新的 CSV 文件
        sampled_df.to_csv(output_csv, index=False)
        print(f"已从 {input_csv} 随机选择 {num_samples} 个样本并保存到 {output_csv}")


file_paths = ['cnn_sql.csv', 'sqli.csv', 'sqliv2.csv', 'SQLiV3.csv', 'Modified_SQL_Dataset.csv',
              "clean_sql_dataset.csv", 'SQL_Dataset.csv',
              'SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv']
num_samples = 20000  # 要选择的样本数量

for input_csv in file_paths:
    sample_and_save_csv(input_csv, "./light_weight_dataset/"+input_csv, num_samples)
