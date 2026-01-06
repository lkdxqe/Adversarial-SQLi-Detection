import sys

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel

from config import get_config
from data_utils import load_dataset
from model_cnn import CLCNN_Model
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, recall_score, f1_score


class CustomLoss(nn.Module):
    def __init__(self, penalty_weight=0.3):
        super(CustomLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        # 计算标准交叉熵损失
        loss = self.cross_entropy(predictions, targets)

        # # 让模型更倾向于分类为Malicious
        # probs_class_0 = predictions[:, 0]
        # penalty = torch.mean(probs_class_0)

        # 避免极端概率的出现（0或者1）
        predictions, _ = torch.max(predictions, dim=1)
        penalty = torch.mean(predictions)

        # print(f"loss:{loss},penalty_weight:{penalty}")
        # 总损失 = 交叉熵损失 + 惩罚项
        total_loss = loss + self.penalty_weight * penalty
        return total_loss


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

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")
        self.logger.info(self.Mymodel.get_arg())

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0

        all_targets = []
        all_predictions = []

        correct_label_0 = 0
        total_label_0 = 0
        # 计算label=1中被正确分类的数量
        correct_label_1 = 0
        total_label_1 = 0

        # Turn on the train mode
        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs = inputs.to(self.args.device)
            targets = targets.to(self.args.device)  # 32
            predicts = self.Mymodel(inputs)
            loss = criterion(predicts, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

            correct_label_1 += ((torch.argmax(predicts, dim=1) == targets) & (targets == 1)).sum().item()
            total_label_1 += (targets == 1).sum().item()
            correct_label_0 += ((torch.argmax(predicts, dim=1) == targets) & (targets == 0)).sum().item()
            total_label_0 += (targets == 0).sum().item()
            # Record targets and predictions
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(torch.argmax(predicts, dim=1).cpu().numpy())

        print(f'Label=1中被正确分类的数量: {correct_label_1}/{total_label_1}')
        print(f'Label=0中被正确分类的数量: {correct_label_0}/{total_label_0}')
        # Calculate recall and F1 score
        accuracy = accuracy_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions, pos_label=1)
        f1 = f1_score(all_targets, all_predictions, average='macro')

        return train_loss / n_train, accuracy, recall, f1
        # return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        # Turn on the eval mode
        self.Mymodel.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

                # Record targets and predictions
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(torch.argmax(predicts, dim=1).cpu().numpy())

        # Calculate recall and F1 score
        accuracy = accuracy_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions, pos_label=1)  # 只计算正类的召回率
        f1 = f1_score(all_targets, all_predictions, average='macro')

        return test_loss / n_test, accuracy, recall, f1
        # return test_loss / n_test, n_correct / n_test

    def run(self):
        # 这里的test_dataloader实际上加载的是val.csv
        train_dataloader, test_dataloader = load_dataset(data_size=self.args.data_size,
                                                         dataset=self.args.dataset,
                                                         data_dir=self.args.data_dir,
                                                         train_batch_size=self.args.train_batch_size,
                                                         test_batch_size=self.args.test_batch_size,
                                                         character_len=self.args.character_len,
                                                         workers=self.args.workers,
                                                         target="train")

        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())

        # 使用自定义损失函数
        criterion = CustomLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        l_acc, l_trloss, l_teloss, l_epo = [], [], [], []
        l_recall, l_f1 = [], []
        # Get the best_loss and the best_acc
        best_loss, best_acc = 0, 0
        time_count = 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc, train_recall, train_f1 = self._train(train_dataloader, criterion, optimizer)
            end = time.time()
            test_loss, test_acc, test_recall, test_f1 = self._test(test_dataloader, criterion)
            start = time.time()
            time_count += start - end
            print(start - end)
            l_epo.append(epoch), l_acc.append(test_acc), l_trloss.append(train_loss), l_teloss.append(test_loss)
            l_recall.append(test_recall), l_f1.append(test_f1)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info(
                '[train] loss: {:.4f}, acc: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(train_loss, train_acc * 100,
                                                                                       train_recall * 100,
                                                                                       train_f1 * 100))
            self.logger.info(
                '[test] loss: {:.4f}, acc: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(test_loss, test_acc * 100,
                                                                                      test_recall * 100, test_f1 * 100))

            if (epoch + 1) % 5 == 0:
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                filters_num = str(self.Mymodel.num_filters)
                num_kernels = str(self.Mymodel.num_kernels)
                embedding_dims = str(self.Mymodel.embedding_dims)
                torch.save(self.Mymodel.state_dict(), './model_my/clcnn_model_' + filters_num + "_"
                           + num_kernels + "_" + embedding_dims + "_" + date_str + "_" + str(epoch) + '.pth')

        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        # Draw the training process
        print(time_count, time_count / self.args.num_epoch)
        # plt.plot(l_epo, l_acc)
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.savefig('acc.png')
        #
        # plt.plot(l_epo, l_teloss)
        # plt.ylabel('test-loss')
        # plt.xlabel('epoch')
        # plt.savefig('teloss.png')
        #
        # plt.plot(l_epo, l_trloss)
        # plt.ylabel('train-loss')
        # plt.xlabel('epoch')
        # plt.savefig('trloss.png')
        #
        # plt.plot(l_epo, l_recall)
        # plt.ylabel('recall')
        # plt.xlabel('epoch')
        # plt.savefig('recall.png')
        #
        # plt.plot(l_epo, l_f1)
        # plt.ylabel('f1')
        # plt.xlabel('epoch')
        # plt.savefig('f1.png')


if __name__ == '__main__':
    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()
    print(begin_time, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
