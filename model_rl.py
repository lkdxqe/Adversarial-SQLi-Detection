import torch
import torch.nn.functional as F
from torch import nn
import utils
import sys
from collections import deque
import numpy as np
import random
from model_cnn import CLCNN_Model, string_to_predicts, string_to_feature


class WebRequestEnv:
    def __init__(self, confidence, mode="train", max_len=200, max_steps=10):
        self.current_string = None
        self.state = None
        self.label = None  # 只有训练模式时才根据已知的label进行奖励
        self.current_predicts = None
        self.mode = mode
        self.max_len = max_len
        self.max_steps = max_steps
        self.steps = 0
        self.confidence = confidence  # [0.4,0.6]
        self.invalid_actions = []  # 当前字符串无效动作（采取之后不会发生改变）
        self.actions = {
            "skip": utils.skip,
            "remove_comments": utils.remove_comments,
            "toggle_case": utils.toggle_case,
            "DML_substitution": utils.DML_substitution,
            "hex2decimal": utils.hex2decimal,
            "placeholders_replace": utils.placeholders_replace,
            "decode_url_encoding": utils.decode_url_encoding,
            "simplify_logical_expressions": utils.simplify_logical_expressions,
            "convert_harmless_subquery_to_true": utils.convert_harmless_subquery_to_true,
            "replace_equals_to_true": utils.replace_equals_to_true,
            "remove_comments_2a": utils.remove_comments_2a,
            "remove_comments_2b": utils.remove_comments_2b,
            "remove_comments_3": utils.remove_comments_3,
        }
        self.action_space = list(self.actions.keys())
        print(f"action_space:{self.action_space}")

        self.reset()

    def reset(self):
        self.invalid_actions = []
        self.steps = 0
        return

    def set_current_string(self, current_string):
        self.current_string = current_string
        return

    def set_state(self, state):
        self.state = state
        return

    def set_target(self, label, predicts):
        # 设置目标类别和当前分类概率
        self.label = label
        self.current_predicts = predicts
        return 0

    def get_reward(self, predicts):
        predicts_old = self.current_predicts
        if self.mode == "train":
            # 根据目标类别和分类概率的改变，计算奖励值
            scale_factor = 100
            # 如果反而导致分类错误，会给一个极大的惩罚，让模型面对非对抗性样本时保持谨慎
            if self.label == 1:
                if (predicts_old[1] > 0.5) and (predicts[1] < 0.5):
                    reward = -300
                else:
                    # 如果将目标正确识别为阳性，那么给予一个更大的奖励
                    scale_factor = 200
                    reward = (2.72 ** (predicts[1] - predicts_old[1]) - 1) * scale_factor
            else:
                # 如果过去分类正确，但是现在分类错误，那么给予一个极大的惩罚
                if (predicts_old[0] > 0.5) and (predicts[0] < 0.5):
                    reward = -300
                else:
                    reward = (2.72 ** (predicts[0] - predicts_old[0]) - 1) * scale_factor
            # print(reward)
        else:
            # 无所谓了
            scale_factor = 50
            reward = scale_factor * abs(predicts[1] - predicts_old[1])

        return reward

    def step(self, act_values, clcnn_model: CLCNN_Model):
        self.steps += 1
        # 如果"skip"的可能性很高，那么直接跳出，并给予奖励
        act_values_np = np.array(act_values)[0]
        act_values_np = act_values_np + abs(np.min(act_values_np))
        act_values_np = act_values_np / np.sum(act_values_np)

        if act_values_np[0] > 0.3:
            done = True
            # 如果"skip"导致分类正确，那么给予奖励，否则惩罚
            if self.mode == "train":
                if torch.argmax(self.current_predicts) == self.label:
                    reward = 50
                else:
                    reward = -30
            else:
                reward = 0

            return self.state, reward, done, 0

        _, action_idx_sort = torch.sort(act_values, dim=1, descending=True)
        action_idx_sort = action_idx_sort.tolist()[0]

        action_idx = 0  # 默认选概率最高的
        for action_idx_ in action_idx_sort:
            if action_idx_ not in self.invalid_actions:
                action_idx = action_idx_
                break
        action = self.action_space[action_idx]
        next_string = self.actions[action](self.current_string)

        if next_string != self.current_string:
            self.invalid_actions = []
            next_state = string_to_feature(clcnn_model, next_string, self.max_len)[0]
            predicts = string_to_predicts(clcnn_model, next_string, self.max_len)[0]
            reward = self.get_reward(predicts)
            self.current_string = next_string
            self.state = next_state
            self.current_predicts = predicts
        else:
            # 如果动作导致没有变化产生
            # 主动的skip在该函数的开头已经处理
            self.invalid_actions.append(action_idx)
            next_state = self.state
            predicts = self.current_predicts
            # 字符串没有发生改变，给予惩罚
            # 但应该更尽量避免改动后反而出现错误的情况
            reward = -10

        # if reward != -50:
        #     print(f"reward:{reward}")

        done = self.is_done(predicts)
        if len(self.invalid_actions) == len(self.action_space):
            done = True

        return next_state, reward, done, action_idx

    def is_done(self, predicts):
        done = False
        if self.mode == "train" and (
                (predicts[1] < self.confidence[0] and self.label == 0) or (
                predicts[1] > self.confidence[1] and self.label == 1)):
            done = True

        if self.mode == "test" and (
                predicts[1] <= self.confidence[0] or predicts[1] >= self.confidence[1]):
            done = True

        if self.steps >= self.max_steps:
            done = True
        return done


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, experience, td_error):
        self.buffer.append(experience)
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        sampling_probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probs)
        samples = [self.buffer[idx] for idx in indices]
        importance = (len(self.buffer) * sampling_probs[indices]) ** (-beta)
        importance /= importance.max()
        return samples, indices, importance

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, character_len, num_filters)
        x = x.permute(0, 2, 1)  # (batch_size, num_filters, character_len)

        # 平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, character_len)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, character_len)

        # 拼接平均池化和最大池化的结果
        concat_out = torch.cat([avg_out, max_out], dim=1)  # (batch_size, 2, character_len)
        attention_map = self.conv1(concat_out)  # (batch_size, 1, character_len)
        attention_map = self.sigmoid(attention_map)  # (batch_size, 1, character_len)
        attention_map = attention_map.permute(0, 2, 1)  # (batch_size, character_len, 1)

        x = x.permute(0, 2, 1)  # (batch_size, character_len, num_filters)
        return x * attention_map  # (batch_size, character_len, num_filters)


class LightRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LightRNN, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, character_len, num_filters)
        x = self.spatial_attention(x)
        x = x.transpose(1, 2)  # (batch_size, num_filters, seq_length)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (batch_size, seq_length // 5, num_filters)

        h_0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        rnn_out, _ = self.rnn(x, h_0)
        out = rnn_out[:, -1, :]
        out = self.fc(out)
        return out


class LightGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LightGRU, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, character_len, num_filters)
        x = self.spatial_attention(x)
        x = x.transpose(1, 2)  # (batch_size, num_filters, seq_length)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (batch_size, seq_length // 5, num_filters)

        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        gru_out, _ = self.gru(x, h_0)
        out = gru_out[:, -1, :]
        out = self.fc(out)
        return out


class LightLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LightLSTM, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, character_len, num_filters)
        x = self.spatial_attention(x)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out


class LightFC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LightFC, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, character_len, num_filters)
        x = self.spatial_attention(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        out = self.fc2(x)
        return out


class DQNAgent:
    def __init__(self, state_size, action_size, num_filters=5, character_len=200, random_select=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.random_select = random_select

        self.model = LightRNN(num_filters, hidden_size=16, output_size=action_size, num_layers=2)
        self.target_model = LightRNN(num_filters, hidden_size=16, output_size=action_size, num_layers=2)
        # self.model = LightLSTM(num_filters, hidden_size=32, output_size=action_size, num_layers=1)
        # self.target_model = LightLSTM(num_filters, hidden_size=32, output_size=action_size, num_layers=1)
        # self.model = LightGRU(input_size=num_filters, hidden_size=16, output_size=action_size)
        # self.target_model = LightGRU(input_size=num_filters, hidden_size=16, output_size=action_size)
        # self.model = LightFC(input_size=num_filters * character_len, hidden_size=32, output_size=action_size)
        # self.target_model = LightFC(input_size=num_filters * character_len, hidden_size=32, output_size=action_size)

        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.step_count = 0  # 计数器，用于更新目标Q网络
        self.update_frequency = 100  # 每50次replay更新一次目标Q网络

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def eval(self):
        self.model.eval()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"DQN模型保存至 {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        with torch.no_grad():
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)[0]).item()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor)
            td_error = target - target_f[0][action].item()
        self.memory.add((state, action, reward, next_state, done), td_error)

    def act(self, state):
        if self.random_select or np.random.rand() <= self.epsilon:
            # 随机排个序就返回
            # sorted_action_indices = list(range(self.action_size))
            # random.shuffle(sorted_action_indices)
            # return sorted_action_indices
            return torch.tensor([[random.random() for x in range(self.action_size)]])

        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        # _, act_idx_sort = torch.sort(act_values, dim=1, descending=True)
        # act_idx_sort = act_idx_sort.tolist()[0]
        return act_values.detach()

    def replay(self, batch_size, beta=0.4):
        if len(self.memory.buffer) < batch_size:
            return
        minibatch, indices, importance = self.memory.sample(batch_size, beta)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        importance = torch.FloatTensor(importance)

        # Double DQN target calculation
        with torch.no_grad():
            next_action_values = self.model(next_states)
            next_action_indices = torch.argmax(next_action_values, dim=1)
            next_q_values = self.target_model(next_states).gather(1, next_action_indices.unsqueeze(1)).squeeze(1)
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = targets - current_q_values

        loss = (importance * (td_errors ** 2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count == self.update_frequency:
            self.step_count = 0
            self.update_target_model()
