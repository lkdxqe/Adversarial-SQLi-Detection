from functools import partial

import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import sys

# Make MyDataset
class MyDataset(Dataset):
    def __init__(self, sentences, labels, character_len, mode="CLCNN"):
        self.sentences = sentences
        self.labels = labels

        dataset = list()
        index = 0
        if mode == "CLCNN":
            # 只使用CLCNN，则直接转换到ascii编码并填充
            for data in sentences:
                # print(data)
                # data = data.lower()
                tokens = list(data)  # 按字符分割
                tokens = [0 if ord(word) - 32 < 0 or ord(word) - 32 > 95 else ord(word) - 32 for word in tokens]
                if len(tokens) > character_len:
                    tokens = tokens[:character_len]
                else:
                    tokens += [0] * (character_len - len(tokens))

                labels_id = labels[index]
                index += 1
                dataset.append((tokens, labels_id))
        elif mode == "CLCNN+RL":
            # 使用RL需要得到原始字符串
            for idx in range(len(sentences)):
                data = sentences[idx]
                label = labels[idx]
                dataset.append((data, label))

        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self.sentences)


def my_collate(batch):
    tokens, label_ids = map(list, zip(*batch))
    if isinstance(tokens[0], str):
        # 用于mode="CLCNN+RL"，返回原始字符串
        return tokens,label_ids
    return torch.tensor(tokens), torch.tensor(label_ids)


# def load_dataset(data_size, dataset, data_dir, workers,train_batch_size=4, test_batch_size=32, character_len=200,mode="CLCNN",train_size=0.7):
#     csv_path = os.path.join(data_dir, dataset)
#     print(f"csv_path:{csv_path}")
#     data = pd.read_csv(csv_path, encoding='utf-8')
#     len1 = int(len(list(data['label'])) * data_size)
#     labels = list(data['label'])[0:len1]
#     sentences = list(data['text'])[0:len1]
#
#     # split train_set and test_set
#     tr_sen, te_sen, tr_lab, te_lab = train_test_split(sentences, labels, train_size=train_size)
#     # print(te_lab)
#     # Dataset
#     train_set = MyDataset(tr_sen, tr_lab, character_len,mode)
#     test_set = MyDataset(te_sen, te_lab, character_len,mode)
#     # DataLoader
#     collate_fn = partial(my_collate)
#     train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
#                               collate_fn=collate_fn, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
#                              collate_fn=collate_fn, pin_memory=True)
#     return train_loader, test_loader

def load_dataset(data_size, dataset, data_dir, workers,train_batch_size=4, test_batch_size=32, character_len=200,mode="CLCNN",target="train"):
    if mode == "CLCNN+RL":
        # 使用强化学习处理convert数据集
        if target == "train":
            if dataset.endswith('.csv'):
                train_path = os.path.join(data_dir, "convert_train.csv")
                val_path = os.path.join(data_dir, "convert_val.csv")
            elif dataset == "":
                train_path = os.path.join(data_dir, "train.csv")
                val_path = os.path.join(data_dir, "val.csv")
            else:
                train_path = os.path.join(data_dir, dataset + "_train.csv")
                val_path = os.path.join(data_dir, dataset + "_val.csv")
        else:
            if dataset.endswith('.csv'):
                train_path = os.path.join(data_dir, "convert_val.csv")
                val_path = os.path.join(data_dir, "convert_test.csv")
            elif dataset == "":
                train_path = os.path.join(data_dir, "val.csv")
                val_path = os.path.join(data_dir, "test.csv")
            else:
                train_path = os.path.join(data_dir, dataset + "_val.csv")
                val_path = os.path.join(data_dir, dataset + "_test.csv")

    else:
        # 正常训练
        if target == "train":
            if dataset == "":
                train_path = os.path.join(data_dir, "train.csv")
                val_path = os.path.join(data_dir, "val.csv")
            elif dataset.endswith('.csv'):
                # 使用train_and_10%数据集
                train_path = os.path.join(data_dir, "convert_train.csv")
                val_path = os.path.join(data_dir, "convert_val.csv")
        else:
            if dataset == "":
                train_path = os.path.join(data_dir, "val.csv")
                val_path = os.path.join(data_dir, "test.csv")
            elif dataset.endswith('.csv'):
                # 使用train_and_10%数据集
                train_path = os.path.join(data_dir, "convert_val.csv")
                val_path = os.path.join(data_dir, "convert_test.csv")
            else:
                train_path = os.path.join(data_dir, dataset+"_val.csv")
                val_path = os.path.join(data_dir, dataset+"_test.csv")

    print(f"train_path:{train_path}")
    print(f"val_path:{val_path}")
    data = pd.read_csv(train_path, encoding='utf-8')
    len1 = int(len(list(data['label'])) * data_size)
    tr_lab = list(data['label'])[0:len1]
    tr_sen = list(data['text'])[0:len1]

    data = pd.read_csv(val_path, encoding='utf-8')
    len1 = int(len(list(data['label'])) * data_size)
    val_lab = list(data['label'])[0:len1]
    val_sen = list(data['text'])[0:len1]

    # Dataset
    train_set = MyDataset(tr_sen, tr_lab, character_len,mode)
    val_set = MyDataset(val_sen, val_lab, character_len,mode)
    # DataLoader
    collate_fn = partial(my_collate)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                             collate_fn=collate_fn, pin_memory=True)
    return train_loader, val_loader