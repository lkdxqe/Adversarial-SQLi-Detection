import torch
import torch.nn.functional as F
from torch import nn
import utils
import sys
from collections import deque
import numpy as np
import random


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=2):
        super(ChannelAttention, self).__init__()
        # 通道注意力机制
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力机制
        max_out = self.max_pool(x)  # (batch_size,num_kernels,character_len)
        max_out = self.mlp(max_out.view(max_out.size(0), -1))  # [batch_size, num_kernels]

        avg_out = self.avg_pool(x)
        avg_out = self.mlp(avg_out.view(avg_out.size(0), -1))  # [batch_size, num_kernels]
        channel_out = self.sigmoid(max_out + avg_out)  # [batch_size, num_kernels]
        channel_out = channel_out.view(x.size(0), x.size(1), 1)
        return x * channel_out  # [batch_size, num_kernels, character_len]


# CLCNN_Model
class CLCNN_Model(nn.Module):
    def __init__(self, num_classes, character_len=200):
        super().__init__()
        self.num_classes = num_classes
        self.character_len = character_len
        self.filter_sizes = [1, 3, 5, 7, 9]  # 由于填充的问题，现在只能是奇数
        self.num_filters = len(self.filter_sizes)
        self.num_kernels = 64
        self.max_features = 100  # 字符的个数
        self.embedding_dims = 32
        self.Embed = nn.Embedding(self.max_features, self.embedding_dims)

        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=self.num_kernels,
                          kernel_size=(K, self.embedding_dims)),
                nn.BatchNorm2d(self.num_kernels),
                nn.ReLU(),
                nn.Dropout(0.5)
            ) for K in self.filter_sizes]
        )

        self.channel_attentions = nn.ModuleList(
            [ChannelAttention(in_channel=self.num_kernels, reduction=2) for _ in range(self.num_filters)]
        )

        input_dim = self.num_kernels * self.num_filters
        self.block = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def get_arg(self):
        arg = f"self.filter_sizes:{self.filter_sizes}, " \
              f"self.num_kernels:{self.num_kernels}, " \
              f"self.max_features:{self.max_features}, " \
              f"self.embedding_dims:{self.embedding_dims}"
        return arg

    def calculate_padding(self, kernel_size):
        return kernel_size[0] // 2, 0

    def conv_pool(self, tokens, idx):
        # 进行填充，使得不同卷积核大小得到相同长度的张量
        conv = self.convs[idx]
        channel_attention = self.channel_attentions[idx]
        kernel_size = (self.filter_sizes[idx], self.embedding_dims)  # (filters_num, dim)
        pad = self.calculate_padding(kernel_size)
        tokens = F.pad(tokens, (0, 0, pad[0], pad[0]))
        tokens = conv(tokens)  # [batch_size,num_kernels,character_len,1]

        tokens = tokens.squeeze(3)  # [batch_size,num_kernels,character_len]

        # 对同一尺度下，不同的卷积核乘以注意力权值
        tokens = channel_attention(tokens)
        # (batch_size,num_kernels,character_len)
        return tokens

    def forward(self, inputs):
        embeds = self.Embed(inputs)
        embeds = embeds.unsqueeze(1)  # [32, 1, 200, 45]

        out = torch.stack([self.conv_pool(embeds, idx) for idx in range(len(self.convs))], 1)
        # (batch_size,num_filters,num_kernels,character_len)

        out, _ = torch.max(out, dim=3, keepdim=False)  # （batch_size, filters_num, num_kernels）

        out = out.view(out.size(0), -1)  # （batch_size, filters_num * num_kernels）
        predicts = self.block(out)
        return predicts

    def feature_out(self, inputs):
        embeds = self.Embed(inputs)
        embeds = embeds.unsqueeze(1)  # [32, 1, 200, 45]

        out = torch.stack([self.conv_pool(embeds, idx) for idx in range(len(self.convs))], 1)
        # (batch_size,filter_types,num_filters,character_len)
        out = torch.mean(out, dim=2, keepdim=False)  # (batch_size,filters_num,character_len)
        out = out.transpose(1, 2)  # (batch_size,character_len,filters_num)

        # out = out.view(out.size(0), -1)  # (batch_size,character_len * filters_num)
        # 由5种尺度卷积核的平均结果来代表character_len中对应位置的特征
        return out

    def embedding_out(self,inputs):
        embeds = self.Embed(inputs)
        return embeds

# 通过clcnn模型得到字符串的特征向量
def string_to_feature(clcnn_model: CLCNN_Model, string, max_len=100):
    ascii_codes = [0 if ord(c) - 32 < 0 or ord(c) - 32 > 95 else ord(c) - 32 for c in string]
    if len(ascii_codes) > max_len:
        ascii_codes = ascii_codes[:max_len]
    else:
        ascii_codes += [0] * (max_len - len(ascii_codes))
    ascii_codes = torch.tensor([ascii_codes])
    # 只需要模型的输出
    with torch.no_grad():
        features = clcnn_model.feature_out(ascii_codes)

    # features = features.squeeze(0)
    # print(features.shape)
    return features


# 通过clcnn模型得到字符串的预测结果
def string_to_predicts(clcnn_model: CLCNN_Model, string, max_len=100):
    ascii_codes = [0 if ord(c) - 32 < 0 or ord(c) - 32 > 95 else ord(c) - 32 for c in string]
    if len(ascii_codes) > max_len:
        ascii_codes = ascii_codes[:max_len]
    else:
        ascii_codes += [0] * (max_len - len(ascii_codes))
    ascii_codes = torch.tensor([ascii_codes])
    # 只需要模型的输出
    clcnn_model.eval()
    with torch.no_grad():
        predicts = clcnn_model(ascii_codes)
    return predicts



def string_to_embedding(clcnn_model: CLCNN_Model, string, max_len=100):
    ascii_codes = [0 if ord(c) - 32 < 0 or ord(c) - 32 > 95 else ord(c) - 32 for c in string]
    if len(ascii_codes) > max_len:
        ascii_codes = ascii_codes[:max_len]
    else:
        ascii_codes += [0] * (max_len - len(ascii_codes))
    ascii_codes = torch.tensor([ascii_codes])
    print(ascii_codes)
    emb = clcnn_model.embedding_out(ascii_codes)
    print(emb.size())
    return emb