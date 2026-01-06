import pandas as pd
import json


def json_write_csv(json_file_name):
    with open(json_file_name+".json", 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df_sqli = df[df['type'] == 'sqli'].copy()
    df_sqli['label'] = 1
    df_sqli = df_sqli[['label', 'pattern']]
    df_sqli.columns = ['label', 'text']

    csv_file_path = json_file_name+'.csv'
    df_sqli.to_csv(csv_file_path, index=False)

    print(f'Data has been written to {csv_file_path}')

if __name__ == "__main__":
    json_write_csv("SQLiV4")
    json_write_csv("SQLiV5")
