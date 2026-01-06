import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime

import torch

def get_config():
    parser = argparse.ArgumentParser()
    '''Base'''
    parser.add_argument('--data_dir', type=str, default='./data/combation/clcnn_my_advr_3')
    parser.add_argument('--character_len', type=int, default=200)   # 每个序列的字符长度
    parser.add_argument('--data_size', type=float, default=1)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--method_name', type=str, default='clcnn',
                        choices=['gru', 'rnn', 'bilstm_ma','bilstm', 'lstm', 'fnn', 'clcnn', 'attention', 'lstm+textcnn',
                                 'lstm_textcnn_attention'])

    '''Optimization'''
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    # parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    '''Environment'''
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))

    args = parser.parse_args()
    args.device = torch.device(args.device)

    '''logger'''
    args.log_name = '{}_{}.log'.format(args.method_name,
                                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger
