import torch
import torchvision.transforms as T
from mario_env import MarioEnv
from cnn_model import MarioCNN
# TODO: from dqn_model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MarioEnv('SuperMarioBros-1-1-v0')
cnn = MarioCNN().to(device)
# TODO: dqn = DQN().to(device)

transform = T.Compose([
    T.ToPILImage(),  # 将numpy数组转换为PIL图像
    T.Resize((84, 84)),  # 调整图像大小以匹配CNN输入
    T.ToTensor(),  # 将PIL图像转换为Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

state = env.reset()
state = transform(state).unsqueeze(0).to(device)  # 预处理并添加batch维度

for step in range(5000):
    action = env.action_space.sample()  # pick action randomly
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = transform(next_state).unsqueeze(0).to(device)  # 预处理下一状态

    # Use CNN to get features
    state_features = cnn(state)
    next_state_features = cnn(next_state)
    print("CNN output features:", state_features)
    print("Shape of output:", state_features.shape)

    # Use DQN to make decisions
    # TODO: use DQN to train the model

    # 准备下一步
    state = next_state

    if terminated or truncated:
        state = env.reset()
        state = transform(state).unsqueeze(0).to(device)

env.close()
