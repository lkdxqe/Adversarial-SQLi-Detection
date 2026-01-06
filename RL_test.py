import random
import sys
from model_cnn import CLCNN_Model, string_to_predicts, string_to_feature
from model_rl import WebRequestEnv, DQNAgent
import torch
from config import get_config
from data_utils import load_dataset
from tqdm import tqdm
import utils

confidence = [0.4, 0.6]


def test_dqn(dqn_model_path, dataloader, agent, env, episodes, batch_size, clcnn_model: CLCNN_Model, max_len=200):
    agent.load(dqn_model_path)
    agent.eval()
    count_all = 0
    correct_count_all = 0
    clcnn_count = 0
    clcnn_correct_count = 0
    rl_count = 0
    rl_correct_count = 0
    # 如果rl占比的那部分不经过处理，原先的准确率该是多少？
    my_count = 0
    step_counts = []
    actions_logs = []
    for inputs, targets in tqdm(dataloader, ascii='>='):
        # if random.random() > 0.3:
        #     continue
        count_all += 1
        env.reset()
        original_string = inputs[0]
        label = targets[0]

        original_predicts = string_to_predicts(clcnn_model, original_string, max_len=max_len)[0]
        confidence = env.confidence

        # 如果不在uncertain区间，那么无需进入强化学习模块
        if original_predicts[1] < confidence[0] or original_predicts[1] > confidence[1]:
            clcnn_count += 1
            if (original_predicts[1] <= 0.5 and label == 0) or (original_predicts[1] > 0.5 and label == 1):
                correct_count_all += 1
                clcnn_correct_count += 1
            continue

        rl_count += 1
        if torch.argmax(original_predicts) == label:
            my_count += 1

        original_state = string_to_feature(clcnn_model, original_string, max_len=max_len)[0]
        state = original_state
        env.set_current_string(original_string)
        env.set_state(state)
        env.set_target(label, original_predicts)

        actions_log = []
        step_count = 0
        for time in range(20):
            # 通过agent.act得到降序排列的动作
            act_values = agent.act(state)
            # print(act_values)
            # step也会返回这一步所采取的动作
            next_state, reward, done, action_idx = env.step(act_values, clcnn_model)
            state = next_state

            if action_idx is not None:
                actions_log.append(env.action_space[action_idx])

            if env.action_space[action_idx] != "skip":
                step_count += 1

            if done:
                break

        actions_logs.append(actions_log)
        step_counts.append(step_count)

        # if torch.argmax(env.current_predicts) == label:
        #     correct_count_all += 1
        #     rl_correct_count += 1

        if (env.current_predicts[1] <= 0.5 and label == 0) or (env.current_predicts[1] > 0.5 and label == 1):
            correct_count_all += 1
            rl_correct_count += 1

    # 统计动作频次并排序
    # utils.store_actions_logs(actions_logs, "./pic/actions_log/rl_actions_log_advr_0.json")

    print(f"correct_count/count_all: {correct_count_all}/{count_all} = {correct_count_all / count_all}")
    print(f"ratio clcnn_count/count_all :{clcnn_count}/{count_all} = {clcnn_count / count_all}")
    print(f"Accuracy clcnn_count/count_all :{clcnn_correct_count}/{clcnn_count} = {clcnn_correct_count / clcnn_count}")
    print(f"ratio rl_count/count_all :{rl_count}/{count_all} = {rl_count / count_all}")
    print(f"Accuracy rl_count/rl_count :{rl_correct_count}/{rl_count} = {rl_correct_count / rl_count}")
    print(f"Accuracy my_count/rl_count :{my_count}/{rl_count} = {my_count / rl_count}")
    print(f"steps: {sum(step_counts)}")


def test_model(cnn_model_path, dqn_model_path, test_data_dir, test_dataset, confidence, random_select=False):
    args, logger = get_config()
    clcnn_model = CLCNN_Model(args.num_classes, args.character_len)
    clcnn_model.load_state_dict(torch.load(cnn_model_path))
    # 加载数据集，在RL中每次只会有一个字符串进行训练
    train_dataloader, test_dataloader = load_dataset(data_size=args.data_size,
                                                     dataset=test_dataset,
                                                     data_dir=test_data_dir,
                                                     train_batch_size=1,
                                                     test_batch_size=1,
                                                     character_len=args.character_len,
                                                     workers=args.workers,
                                                     mode="CLCNN+RL",
                                                     target="test")

    env = WebRequestEnv(mode="test", confidence=confidence)
    agent = DQNAgent(clcnn_model.character_len * clcnn_model.num_filters, len(env.actions), random_select=random_select)

    test_dqn(dqn_model_path, test_dataloader, agent, env, episodes=1000, batch_size=1, clcnn_model=clcnn_model,
             max_len=args.character_len)


if __name__ == "__main__":
    cnn_model_path = './model_my/clcnn_model_5_64_32_2024-08-30_19.pth'
    dqn_model_path = "./model_my/dqn_model_rnn.pth"
    # test_dataset = "convert_test.csv"
    # test_data_dir = './data_convert/convert_clcnn'
    # test_dataset = ""
    # test_data_dir = './data/combation/clcnn_rl_advr_1/'
    test_dataset = ""
    test_data_dir = './data/split_dataset/'
    test_model(cnn_model_path, dqn_model_path, test_data_dir, test_dataset, confidence=confidence, random_select=False)
