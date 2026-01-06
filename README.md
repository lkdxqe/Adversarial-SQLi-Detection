# SQL Injection Detection Project

## Dataset Download & Processing

1. Download the datasets from the following links and place them in the `data` folder:

   - (https://kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset)
   - (https://kaggle.com/datasets/sajid576/sql-injection-dataset)
   - (https://kaggle.com/datasets/gambleryu/biggest-sql-injection-dataset)
   - (https://kaggle.com/datasets/alextrinity/sqli-xss-dataset)
   - (https://github.com/fishyyh/CNN-SQL)
   - (https://github.com/ajinmathew/SQL-data)
   - (https://kaggle.com/datasets/ispangler/csic-2010-web-application-attacks)

2. Run `0handle_xxx.py` to process the datasets.

3. Run `0merge_dataset.py` to merge the datasets.

4. Run `0split_merge_dataset.py` to generate the training, testing, and validation datasets.

## Model Training & Testing

1. Run `0CLCNN_train.py` and `0RL_train.py` to train the models.

2. Run `0CLCNN_test.py` and `RL_test.py` to test the models. `RL_test.py` tests the CLCNN + RL hybrid model.

## Extension Features

This document tests only the regular datasets. You can use the following tools to process the regular datasets and create mutated samples for further testing:

- [AdvSQLi](https://github.com/u21h2/AutoSpear)
- [WAF-A-MoLE](https://github.com/AvalZ/WAF-A-MoLE)
- [GPTFuzzer](https://github.com/HongliangLiang/gptfuzzer)