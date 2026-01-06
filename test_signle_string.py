from model_cnn import CLCNN_Model, string_to_predicts, string_to_embedding
from model_rl import WebRequestEnv, DQNAgent
import torch
import sys

confidence = [0.4, 0.6]
cnn_model_path = './model_my/clcnn_model_5_64_32_2024-08-30_19.pth'
# dqn_model_path = "./model_my/dqn_model_advr_2.pth"

max_len = 200
clcnn_model = CLCNN_Model(2, max_len)
clcnn_model.load_state_dict(torch.load(cnn_model_path))
# env = WebRequestEnv(mode="test", confidence=confidence)
# agent = DQNAgent(clcnn_model.character_len * clcnn_model.num_filters, len(env.actions), random_select=False)
# agent.load(dqn_model_path)
# agent.eval()

payload = "1’ or 1/*i’am a teapot*/=1 --"

original_string = payload

emb = string_to_embedding(clcnn_model, original_string, max_len=max_len)
for i in range(len(payload)):
    print(emb[0][i])

print(ord("'"))
# original_predicts = string_to_predicts(clcnn_model, original_string, max_len=max_len)[0]
# confidence = env.confidence
# if original_predicts[1] < confidence[0] or original_predicts[1] > confidence[1]:
#     score_1 = original_predicts[1].item()
#     print(f"score:{score_1}")
#     # sys.exit()
#
# env.reset()
# original_state = string_to_feature(clcnn_model, original_string, max_len=max_len)[0]
# state = original_state
# env.set_current_string(original_string)
# env.set_state(state)
# env.set_target(1, original_predicts)
#
# actions_log = []
# for time in range(20):
#     act_values = agent.act(state)
#     # step也会返回这一步所采取的动作
#     next_state, reward, done, action_idx = env.step(act_values, clcnn_model)
#     actions_log.append(env.action_space[action_idx])
#     state = next_state
#     if done:
#         break
#
# print(f"actions_log:{actions_log}")
# score_1 = env.current_predicts[1].item()
# print(f"score:{score_1}")
# sys.exit()
