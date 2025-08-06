import argparse
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_utils import GraphNeuralNetwork, GraphFeatureExtractor, DynamicVehicleGraph
from utils.env_utils import VECEnvironmentAdapter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='GNN-Enhanced PPO for VEC')

parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--learning_rate', default=8e-4, type=float)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--capacity', default=2000, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--update_iteration', default=20, type=int)
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
parser.add_argument('--save_interval', default=10, type=int)

# GNN相关参数
parser.add_argument('--gnn_type', default='GAT', type=str, choices=['GAT', 'GCN'])
parser.add_argument('--gnn_hidden_dim', default=64, type=int)
parser.add_argument('--gnn_output_dim', default=128, type=int)
parser.add_argument('--gnn_layers', default=2, type=int)

args = parser.parse_args()

# 环境参数
T = 20
num_tcar = 15
num_scar = 5
num_car = num_tcar + num_scar
num_task = num_tcar
num_uav = 1
num_rsu = 1

env = VECEnvironmentAdapter(num_car, num_tcar, num_scar, num_task, num_uav, num_rsu)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.state_space_shape[0]
action_dim = env.action_space_shape[0]
max_action = 1.0

print('State dim:{} \t Action dim:{} \t GNN output dim:{}'.format(state_dim, action_dim, args.gnn_output_dim))

layer1 = 512
layer2 = 256

Transition = namedtuple('Transition', ['s', 'graph_s', 'a', 'a_log_p', 'r', 's_', 'graph_s_'])


class GNNActor(nn.Module):
    """
    集成GNN的Actor网络
    结合传统状态特征和图网络特征进行决策
    """
    
    def __init__(self, state_dim, action_dim, gnn_output_dim, min_std=1e-4, max_std=10):
        super(GNNActor, self).__init__()
        
        # 传统状态处理分支
        self.state_fc1 = nn.Linear(state_dim, layer1)
        self.state_fc2 = nn.Linear(layer1, layer2)
        
        # 图特征处理分支
        self.graph_fc1 = nn.Linear(gnn_output_dim, layer2)
        self.graph_fc2 = nn.Linear(layer2, layer2)
        
        # 融合层
        self.fusion_fc = nn.Linear(layer2 + layer2, layer2)
        
        # 输出层
        self.mu_head = nn.Linear(layer2, action_dim)
        self.sigma_head = nn.Linear(layer2, action_dim)
        
        self.min_std = min_std
        self.max_std = max_std
        
    def forward(self, state, graph_embedding):
        # 处理传统状态特征
        state_feat = F.relu(self.state_fc1(state))
        state_feat = F.relu(self.state_fc2(state_feat))
        
        # 处理图特征，确保维度匹配
        if graph_embedding.dim() == 3:
            # 如果是3维（batch, seq, features），取最后一个时间步
            graph_embedding = graph_embedding[:, -1, :]
        elif graph_embedding.dim() == 1:
            # 如果是1维，添加batch维度
            graph_embedding = graph_embedding.unsqueeze(0)
        
        graph_feat = F.relu(self.graph_fc1(graph_embedding))
        graph_feat = F.relu(self.graph_fc2(graph_feat))
        
        # 特征融合
        fused_feat = torch.cat([state_feat, graph_feat], dim=-1)
        fused_feat = F.relu(self.fusion_fc(fused_feat))
        
        # 生成策略参数
        mu = 2.0 * torch.tanh(self.mu_head(fused_feat))
        sigma = F.softplus(self.sigma_head(fused_feat))
        sigma = torch.clamp(sigma, self.min_std, self.max_std)
        
        return (mu, sigma)


class GNNCritic(nn.Module):
    """
    集成GNN的Critic网络
    结合传统状态特征和图网络特征进行价值评估
    """
    
    def __init__(self, state_dim, gnn_output_dim):
        super(GNNCritic, self).__init__()
        
        # 传统状态处理分支
        self.state_fc1 = nn.Linear(state_dim, layer1)
        self.state_fc2 = nn.Linear(layer1, layer2)
        
        # 图特征处理分支  
        self.graph_fc1 = nn.Linear(gnn_output_dim, layer2)
        self.graph_fc2 = nn.Linear(layer2, layer2)
        
        # 融合层
        self.fusion_fc = nn.Linear(layer2 + layer2, layer2)
        self.value_head = nn.Linear(layer2, 1)
        
    def forward(self, state, graph_embedding):
        # 处理传统状态特征
        state_feat = F.relu(self.state_fc1(state))
        state_feat = F.relu(self.state_fc2(state_feat))
        
        # 处理图特征，确保维度匹配
        if graph_embedding.dim() == 3:
            # 如果是3维（batch, seq, features），取最后一个时间步
            graph_embedding = graph_embedding[:, -1, :]
        elif graph_embedding.dim() == 1:
            # 如果是1维，添加batch维度
            graph_embedding = graph_embedding.unsqueeze(0)
        
        graph_feat = F.relu(self.graph_fc1(graph_embedding))
        graph_feat = F.relu(self.graph_fc2(graph_feat))
        
        # 特征融合
        fused_feat = torch.cat([state_feat, graph_feat], dim=-1)
        fused_feat = F.relu(self.fusion_fc(fused_feat))
        
        # 输出状态价值
        value = self.value_head(fused_feat)
        return value


class GNN_PPO(object):
    """
    图神经网络增强的PPO算法
    """
    
    def __init__(self):
        self.training_step = 0
        
        # 初始化GNN编码器
        self.gnn_encoder = GraphNeuralNetwork(
            input_dim=5,  # 节点特征维度
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            gnn_type=args.gnn_type,
            num_layers=args.gnn_layers
        ).to(device)
        
        # 初始化图特征提取器
        self.graph_extractor = GraphFeatureExtractor()
        
        # 初始化Actor和Critic网络
        self.anet = GNNActor(state_dim, action_dim, args.gnn_output_dim).to(device)
        self.cnet = GNNCritic(state_dim, args.gnn_output_dim).to(device)
        
        self.buffer = []
        self.counter = 0
        self.buffer_capacity = args.capacity
        self.batch_size = args.batch_size
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        
        # 修复：使用单独的优化器避免参数冲突
        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=args.learning_rate)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=args.learning_rate)
        self.optimizer_gnn = optim.Adam(self.gnn_encoder.parameters(), lr=args.learning_rate)
    
    def extract_graph_features(self, env_state, env_info, training=True):
        """从环境状态提取图特征"""
        try:
            # 构建图数据
            graph_data = self.graph_extractor.extract_graph_from_env(env_state, env_info)
            
            # 修复：移除no_grad，允许梯度流通
            if training:
                graph_embedding = self.gnn_encoder(graph_data.to(device))
            else:
                with torch.no_grad():
                    graph_embedding = self.gnn_encoder(graph_data.to(device))
            
            return graph_embedding
        except Exception as e:
            print(f"⚠️  Graph construction failed: {e}")
            # 改进：使用可学习的默认嵌入而不是零向量
            if not hasattr(self, 'default_graph_embedding'):
                self.default_graph_embedding = nn.Parameter(
                    torch.randn(1, args.gnn_output_dim) * 0.1
                ).to(device)
            return self.default_graph_embedding
    
    def select_action(self, state, env_info=None):
        """
        选择动作 - 集成图网络特征
        
        Args:
            state: 传统环境状态
            env_info: 环境额外信息用于构建图
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # 提取图特征
        if env_info is not None:
            graph_embedding = self.extract_graph_features(state, env_info, training=False)
        else:
            # 如果没有图信息，使用可学习的默认嵌入
            if not hasattr(self, 'default_graph_embedding'):
                self.default_graph_embedding = nn.Parameter(
                    torch.randn(1, args.gnn_output_dim) * 0.1
                ).to(device)
            graph_embedding = self.default_graph_embedding
        
        with torch.no_grad():
            (mu, sigma) = self.anet(state_tensor, graph_embedding)
            
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-max_action, max_action)
        
        return (action.cpu().data.numpy().flatten(), 
                action_log_prob.cpu().data.numpy().flatten(),
                graph_embedding.cpu().data.numpy())
    
    def store(self, transition):
        """存储经验 - 包括图特征"""
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0
    
    def update(self):
        """更新网络参数 - 包括GNN参数"""
        self.training_step += 1
        
        # 提取经验数据
        s, graph_s, a, r, s_, graph_s_, old_action_log_probs = [], [], [], [], [], [], []
        for t in self.buffer:
            s.append(t.s)
            graph_s.append(t.graph_s)
            a.append(t.a)
            r.append(t.r)
            s_.append(t.s_)
            graph_s_.append(t.graph_s_)
            old_action_log_probs.append(t.a_log_p)
        
        # 转换为tensor
        s = torch.tensor(s, dtype=torch.float).to(device)
        graph_s = torch.tensor(graph_s, dtype=torch.float).to(device)
        a = torch.tensor(a, dtype=torch.float).to(device)
        r = torch.tensor(r, dtype=torch.float).view(-1, 1).to(device)
        s_ = torch.tensor(s_, dtype=torch.float).to(device)
        graph_s_ = torch.tensor(graph_s_, dtype=torch.float).to(device)
        old_action_log_probs = torch.tensor(old_action_log_probs, dtype=torch.float).to(device)
        
        # 奖励标准化
        r = (r - r.mean()) / (r.std() + 1e-5)
        
        # 计算目标价值和优势
        with torch.no_grad():
            target_v = r + args.gamma * self.cnet(s_, graph_s_)
        adv = (target_v - self.cnet(s, graph_s)).detach()
        
        # PPO更新
        for _ in range(args.update_iteration):
            for index in BatchSampler(
                    SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                
                # Actor更新
                (mu, sigma) = self.anet(s[index], graph_s[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])
                
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                  1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                
                # 修复：分别更新Actor、Critic和GNN
                self.optimizer_a.zero_grad()
                self.optimizer_gnn.zero_grad()
                action_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.gnn_encoder.parameters(), self.max_grad_norm)
                self.optimizer_a.step()
                self.optimizer_gnn.step()
                
                # Critic更新
                value_loss = F.smooth_l1_loss(self.cnet(s[index], graph_s[index]), target_v[index])
                self.optimizer_c.zero_grad()
                self.optimizer_gnn.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.gnn_encoder.parameters(), self.max_grad_norm)
                self.optimizer_c.step()
                self.optimizer_gnn.step()
        
        del self.buffer[:]
    
    def save(self):
        """保存模型"""
        torch.save({
            'anet': self.anet.state_dict(),
            'cnet': self.cnet.state_dict(),
            'gnn_encoder': self.gnn_encoder.state_dict()
        }, 'saved_models/GNN_PPO_model.pth')
    
    def load(self, model_path='saved_models/GNN_PPO_model.pth'):
        """加载模型"""
        checkpoint = torch.load(model_path)
        self.anet.load_state_dict(checkpoint['anet'])
        self.cnet.load_state_dict(checkpoint['cnet'])
        self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])


def create_mock_env_info(env):
    """
    创建模拟的环境信息用于图构建
    兼容VECEnvironmentAdapter
    """
    # 直接调用环境适配器的方法
    _, env_info = env.reset()
    return env_info


def main():
    """主训练/测试循环"""
    reward_record = []
    delay_record = []
    rate_record = []
    
    agent = GNN_PPO()
    
    if args.mode == 'test':
        agent.load()
        for i in range(1):
            total_reward = 0.0
            state, _ = env.reset()  # 解包元组
            
            for t in range(T):
                env_info = create_mock_env_info(env)
                action, _, _ = agent.select_action(state, env_info)
                action = action.clip(-max_action, max_action)
                next_state, reward, done, _ = env.step(action)  # 环境适配器返回4个值
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            # 获取系统指标
            metrics = env.get_system_metrics()
            total_delay = metrics['total_delay']
            total_success_rate = metrics['success_rate']
            print(f"Test Results - Reward: {total_reward:.2f}, Delay: {total_delay:.2f}, Success Rate: {total_success_rate:.2f}")
    
    elif args.mode == 'train':
        # 创建模型保存目录
        import os
        os.makedirs('saved_models', exist_ok=True)
        os.makedirs('experiment_plot_new', exist_ok=True)
        
        f = open(f'experiment_plot_new/GNN_PPO_{args.gnn_type}_{num_tcar}_{num_scar}.log', 'w')
        print('episode', 'mean_reward', 'mean_delay', 'mean_rate', file=f)
        
        for i in range(500):
            total_reward = 0.0
            state, _ = env.reset()  # 解包元组
            
            for t in range(T):
                # 创建环境信息用于图构建
                env_info = create_mock_env_info(env)
                
                action, action_log_prob, graph_embedding = agent.select_action(state, env_info)
                next_state, reward, done, _ = env.step(action)  # 环境适配器返回4个值
                
                # 存储经验 - 包括图特征
                next_env_info = create_mock_env_info(env)
                _, _, next_graph_embedding = agent.select_action(next_state, next_env_info)
                
                if agent.store(Transition(state, graph_embedding, action, action_log_prob, 
                                        reward, next_state, next_graph_embedding)):
                    agent.update()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # 获取系统指标
            metrics = env.get_system_metrics()
            total_delay = metrics['total_delay']
            total_success_rate = metrics['success_rate']
            
            reward_record.append(total_reward)
            delay_record.append(total_delay)
            rate_record.append(total_success_rate)
            
            reward_record_step = np.mean(reward_record[-100:])  # 最近100次的平均
            delay_record_step = np.mean(delay_record[-100:])
            rate_record_step = np.mean(rate_record[-100:])
            
            print(f"Episode: {i}, Reward: {reward_record_step:.2f}, "
                  f"Delay: {delay_record_step:.2f}, Success Rate: {rate_record_step:.3f}")
            
            print(i, reward_record_step, delay_record_step, rate_record_step, file=f)
            
            # 定期保存模型
            if i % args.save_interval == 0 and i > 0:
                agent.save()
        
        f.close()
        agent.save()  # 最终保存
        print("Training completed and model saved!")


if __name__ == '__main__':
    main()