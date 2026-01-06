from model_cnn import CLCNN_Model, string_to_predicts, string_to_feature
from model_rl import WebRequestEnv, DQNAgent
import torch
from config import get_config
from data_utils import load_dataset
from tqdm import tqdm


cnn_model_path = './model_my/clcnn_model_5_64_32_2024-08-30_19.pth'
dqn_model_path = "./model_my/dqn_model_rnn.pth"
confidence = [0.4, 0.6]
train_from_old = True
epochs = 5

# test_dataset = ""
# test_data_dir = './data/combation/clcnn_rl_s_advr_3'

test_dataset = ""
test_data_dir = './data/split_dataset'

# test_dataset = ['clean_sql_dataset_convert', 'cnn_sql_convert', 'gptfuzzer_dataset',
#                   'Modified_SQL_Dataset_convert',
#                   'SQL_Dataset_convert', 'sqli_convert', 'sqliv2_convert', 'SQLiV3_convert',
#                   'SQLiV4', 'SQLiV5']
# test_data_dir = './data_convert/split_convert'
# test_dataset = 'clean_sql_dataset_convert'

# test_dataset = ""
# test_data_dir = './data/split_dataset'


# test_dataset = "convert_test.csv"
# test_data_dir = './data_convert/convert_clcnn'


def train_dqn(dataloader, agent, env, episodes, batch_size, clcnn_model: CLCNN_Model, max_len=200, epochs=epochs):
    for epoch in range(epochs):
        agent.set_epsilon(1.0)
        for inputs, targets in tqdm(dataloader, ascii='>='):
            # if random.random() > 0.4:
            #     continue

            env.reset()
            original_string = inputs[0]
            label = targets[0]
            original_predicts = string_to_predicts(clcnn_model, original_string, max_len=max_len)[0]
            original_state = string_to_feature(clcnn_model, original_string, max_len=max_len)[0]
            state = original_state
            env.set_current_string(original_string)
            env.set_state(state)
            env.set_target(label, original_predicts)

            for time in range(20):
                # 通过agent.act得到降序排列的动作
                act_values = agent.act(state)
                # step也会返回这一步所采取的动作
                next_state, reward, done, action_idx = env.step(act_values, clcnn_model)
                # print(f"reward:{reward}")
                if next_state is not None:
                    agent.remember(state, action_idx, reward, next_state, done)
                state = next_state
                if done:
                    break
                if len(agent.memory.buffer) > batch_size:
                    agent.replay(batch_size)

        print(f"...............epoch:{epoch}...............")
        agent.save(dqn_model_path)
        # test_model(cnn_model_path, dqn_model_path, test_data_dir, test_dataset, confidence=env.confidence)


if __name__ == "__main__":
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
                                                     target="train")

    env = WebRequestEnv(max_len=200, confidence=confidence, mode="train")
    agent = DQNAgent(clcnn_model.character_len * clcnn_model.num_filters, len(env.actions),
                     character_len=clcnn_model.character_len, random_select=False)

    """"""
    if train_from_old:
        agent.load(dqn_model_path)
    """"""
    train_dqn(train_dataloader, agent, env, episodes=1000, batch_size=32, clcnn_model=clcnn_model,
              max_len=args.character_len)


