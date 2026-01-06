import matplotlib.pyplot as plt
import numpy as np
from model_cnn import CLCNN_Model,string_to_feature
import torch
import sys


def visualize_feature_vector(feature_vector):
    # 归一化特征向量到[0, 1]范围
    normalized_vector = (feature_vector - np.min(feature_vector)) / (np.max(feature_vector) - np.min(feature_vector))
    # 创建颜色映射，从浅蓝到深红
    colors = plt.cm.bwr(normalized_vector)
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 1))
    ax.imshow([colors], aspect='auto')
    # 移除轴
    ax.set_axis_off()
    # 显示图形
    plt.show()


def visualize_string_with_colors(input_string, output_values, target_length=100):
    if len(input_string) < target_length:
        input_string += chr(0) * (target_length - len(input_string))

    normalized_values = (output_values - np.min(output_values)) / (np.max(output_values) - np.min(output_values))

    fig, ax = plt.subplots(figsize=(int(target_length / 10), 1))
    ax.axis('off')

    for i, (char, value) in enumerate(zip(input_string, normalized_values)):
        if ord(char) == 0:
            color = (1, 1, 1, 0)
        else:
            color = plt.cm.coolwarm(value)
        ax.text(i * (1 / target_length), 0.5, char if ord(char) != 0 else ' ',
                ha='center', va='center', fontsize=10, color=color, fontweight='bold')
    plt.show()


def visualize_feature_vector_and_string(input_string, feature_vector, target_length=100):
    # 归一化特征向量到[0, 1]范围
    normalized_vector = (feature_vector - np.min(feature_vector)) / (np.max(feature_vector) - np.min(feature_vector))
    # 创建颜色映射，从浅蓝到深红
    colors = plt.cm.bwr(normalized_vector)

    if len(input_string) < target_length:
        input_string += chr(0) * (target_length - len(input_string))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(int(target_length / 10), 2))

    # 特征向量可视化
    ax1.imshow([colors], aspect='auto')
    ax1.set_axis_off()

    # 字符串可视化
    ax2.axis('off')
    for i, (char, value) in enumerate(zip(input_string, normalized_vector)):
        if ord(char) == 0:
            color = (1, 1, 1, 0)
        else:
            color = plt.cm.coolwarm(value)
        ax2.text(i * (1 / target_length), 0.5, char if ord(char) != 0 else ' ',
                 ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    plt.show()


if __name__ == "__main__":
    max_len = 200
    model_path = "./model_my/clcnn_model2024-07-19_25.pth"
    clcnn_model = CLCNN_Model(2, character_len=max_len)
    clcnn_model.load_state_dict(torch.load(model_path))

    # text = "errorMsg=Credenciales incorrectasbob@<SCRipt>alert(Paros)</scrIPT>.parosproxy.org"
    # text = "-161 union select 1,2,3,4,5,6,group_concat(table_name separator 0x3a),8 from information_schema.tables where table_schema=0x753637393239--"
    # text = "mid=4df38eacb8d07691aad768c651fb32ce&status=11&v=281474976714751&r=195948557&911=3&911e=2147500037&usn=469718"
    text = "regcheck.php?item=u&username=undefined&ajax_request=1488931245973' , (select (case when (8281=8281) then 1 else (select 1 from (select 1 union select 2)x) end)) )-- -"

    feature = string_to_feature(clcnn_model, text, max_len=max_len)
    print(f"feature:{feature.shape}")
    # feature = feature[0][max_len * 0:max_len * 1] + feature[0][max_len * 1:max_len * 2] + feature[0][
    #                                                                                       max_len * 2:max_len * 3] + \
    #           feature[0][max_len * 3:max_len * 4]
    # print(feature.shape)
    # feature_vector = feature.numpy()
    # # visualize_feature_vector(feature_vector)
    # # visualize_string_with_colors(text, feature_vector,target_length=100)
    # print(text)
    # visualize_feature_vector_and_string(text, feature_vector, target_length=max_len)
