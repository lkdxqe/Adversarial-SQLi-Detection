"""
主要用来测试
"""
import sys

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel

from config import get_config
from data_utils import load_dataset
from model_cnn import CLCNN_Model
import time
from sklearn.metrics import accuracy_score, recall_score, f1_score

import pickle

# model_path = './model_my/backup2/clcnn_model_5_64_32_2024-08-30_19.pth'
model_path = './model_my/clcnn_model_5_64_32_2024-09-12_24.pth'
# 跟config.py里的填写模式一样

# test_dataset = ['clean_sql_dataset_split_convert', 'cnn_sql_convert', 'gptfuzzer_dataset',
#                   'Modified_SQL_Dataset_convert',
#                   'SQL_Dataset_convert', 'sqli_convert', 'sqliv2_convert', 'SQLiV3_convert',
#                   'SQLiV4', 'SQLiV5']

# test_data_dir = './data_convert/split_convert'
# test_dataset = 'clean_sql_dataset_split_convert'

test_dataset = ""
test_data_dir = './data/split_dataset'

# test_dataset = "convert_test.csv"
# test_data_dir = './data_convert/convert_clcnn/'


class Instructor:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.padded = False

        # Operate the method
        if args.method_name == 'clcnn':
            self.Mymodel = CLCNN_Model(args.num_classes, args.character_len)
        else:
            raise ValueError('unknown method')

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

        self.Mymodel.load_state_dict(torch.load(model_path))

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        # Turn on the eval mode

        all_targets = []
        all_predictions = []
        all_predictions_score = []

        correct_label_0 = 0
        total_label_0 = 0
        correct_label_1 = 0
        total_label_1 = 0

        self.Mymodel.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

                correct_label_1 += ((torch.argmax(predicts, dim=1) == targets) & (targets == 1)).sum().item()
                total_label_1 += (targets == 1).sum().item()
                correct_label_0 += ((torch.argmax(predicts, dim=1) == targets) & (targets == 0)).sum().item()
                total_label_0 += (targets == 0).sum().item()
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(torch.argmax(predicts, dim=1).cpu().numpy())
                all_predictions_score.extend(predicts.cpu().numpy())

                # incorrect_indices = (torch.argmax(predicts, dim=1) != targets)
                # incorrect_indices = torch.nonzero(incorrect_indices, as_tuple=False).squeeze()
                # if incorrect_indices.numel() > 1:
                #     incorrect_inputs = inputs[incorrect_indices]
                #     targets_ = targets[incorrect_indices]
                #     for idx in range(len(incorrect_inputs)):
                #         incorrect_input = incorrect_inputs[idx]
                #         target = targets_[idx]
                #         incorrect_input = incorrect_input.tolist()
                #         incorrect_input = [x + 32 for x in incorrect_input]
                #         string = ''.join(chr(c) for c in incorrect_input)
                # print(f"target:{target}")
                # print(string)
                # print(incorrect_input)
                # sys.exit()

        print(f'Label=1中被正确分类的数量: {correct_label_1}/{total_label_1}')
        print(f'Label=0中被正确分类的数量: {correct_label_0}/{total_label_0}')
        accuracy = accuracy_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions, pos_label=1)  # 只计算正类的召回率
        f1 = f1_score(all_targets, all_predictions, average='macro')

        pic_path = './pic/test_data.pkl'
        with open(pic_path, 'wb') as file:
            pickle.dump(all_targets, file)
            pickle.dump(all_predictions_score, file)
            print(f"保存至{pic_path}")

        return test_loss / n_test, accuracy, recall, f1
        # return test_loss / n_test, n_correct / n_test

    def run(self):
        # 此时拿到的test_dataloader才是读test.csv
        train_dataloader, test_dataloader = load_dataset(data_size=self.args.data_size,
                                                         dataset=test_dataset,
                                                         data_dir=test_data_dir,
                                                         train_batch_size=self.args.train_batch_size,
                                                         test_batch_size=self.args.test_batch_size,
                                                         character_len=self.args.character_len,
                                                         workers=self.args.workers,
                                                         target="test")

        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()

        test_loss, test_acc, test_recall, test_f1 = self._test(test_dataloader, criterion)
        print(f"test_loss:{test_loss},test_acc:{test_acc},test_recall:{test_recall},test_f1:{test_f1}")


if __name__ == '__main__':
    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()
    print(begin_time, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
