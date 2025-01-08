import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import torch
import torch.nn as nn

class QAgent(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建行为网络和目标网络
        self.behavior_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()  # 将目标网络初始化为与行为网络相同
        
        self.optimizer = optim.Adam(self.behavior_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.behavior_net.state_dict())

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.behavior_net(state)
        action = torch.argmax(q_values).item()
        return action
    
    def update(self, batch_size, gamma):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # 计算行为网络的Q值
        q_values = self.behavior_net(states).gather(1, actions)
        
        # 计算目标网络的Q值（使用贪婪策略，选择最大Q值）
        next_q_values = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        
        # 计算目标Q值
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        # 更新行为网络
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x