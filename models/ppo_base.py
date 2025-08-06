"""
PPO基础实现
传统PPO算法，用于对比基准和算法验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List
from collections import namedtuple

# 经验元组
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])


class Actor(nn.Module):
    """
    传统PPO Actor网络
    生成连续动作的策略分布
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim1: int = 512, hidden_dim2: int = 256,
                 min_std: float = 1e-4, max_std: float = 10):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.mu_head = nn.Linear(hidden_dim2, action_dim)
        self.sigma_head = nn.Linear(hidden_dim2, action_dim)
        
        self.min_std = min_std
        self.max_std = max_std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 状态张量 [batch_size, state_dim]
            
        Returns:
            mu: 动作均值 [batch_size, action_dim]
            sigma: 动作标准差 [batch_size, action_dim]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mu = 2.0 * torch.tanh(self.mu_head(x))  # 动作范围[-2, 2]
        sigma = F.softplus(self.sigma_head(x))  # 确保标准差为正
        sigma = torch.clamp(sigma, self.min_std, self.max_std)
        
        return mu, sigma


class Critic(nn.Module):
    """
    传统PPO Critic网络
    估计状态价值函数
    """
    
    def __init__(self, state_dim: int, hidden_dim1: int = 512, hidden_dim2: int = 256):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.value_head = nn.Linear(hidden_dim2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 状态张量 [batch_size, state_dim]
            
        Returns:
            value: 状态价值 [batch_size, 1]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        
        return value


class PPOBase:
    """
    传统PPO算法基础实现
    用于对比和验证
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 clip_param: float = 0.2, buffer_capacity: int = 2000,
                 batch_size: int = 256, update_iteration: int = 10,
                 max_grad_norm: float = 0.5, device: str = 'cpu'):
        """
        初始化PPO算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            clip_param: PPO裁剪参数
            buffer_capacity: 经验池容量
            batch_size: 批处理大小
            update_iteration: 更新迭代次数
            max_grad_norm: 梯度裁剪范数
            device: 计算设备
        """
        self.device = device
        self.gamma = gamma
        self.clip_param = clip_param
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.update_iteration = update_iteration
        self.max_grad_norm = max_grad_norm
        
        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 经验池
        self.buffer = []
        self.counter = 0
        
        # 训练统计
        self.training_step = 0
        
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择动作
        
        Args:
            state: 状态数组
            
        Returns:
            action: 选择的动作
            action_log_prob: 动作对数概率
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu, sigma = self.actor(state_tensor)
            
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        # 限制动作范围
        action = torch.clamp(action, -1.0, 1.0)
        
        return (action.cpu().numpy().flatten(), 
                action_log_prob.cpu().numpy().flatten())
    
    def store_transition(self, transition: Transition) -> bool:
        """
        存储经验
        
        Args:
            transition: 经验元组
            
        Returns:
            是否需要更新（缓冲区是否满）
        """
        self.buffer.append(transition)
        self.counter += 1
        
        return self.counter % self.buffer_capacity == 0
    
    def update(self) -> dict:
        """
        更新网络参数
        
        Returns:
            训练统计信息
        """
        self.training_step += 1
        
        # 提取经验数据
        states, actions, rewards, next_states, old_action_log_probs = [], [], [], [], []
        
        for transition in self.buffer:
            states.append(transition.s)
            actions.append(transition.a)
            rewards.append(transition.r)
            next_states.append(transition.s_)
            old_action_log_probs.append(transition.a_log_p)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        old_action_log_probs = torch.FloatTensor(old_action_log_probs).to(self.device)
        
        # 奖励标准化
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # 计算目标价值和优势
        with torch.no_grad():
            next_values = self.critic(next_states)
            target_values = rewards + self.gamma * next_values
            
        current_values = self.critic(states)
        advantages = (target_values - current_values).detach()
        
        # 统计信息
        actor_losses = []
        critic_losses = []
        
        # 多次更新
        from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
        
        for _ in range(self.update_iteration):
            # 随机批采样
            for indices in BatchSampler(
                SubsetRandomSampler(range(len(self.buffer))), 
                self.batch_size, False):
                
                # Actor更新
                mu, sigma = self.actor(states[indices])
                dist = Normal(mu, sigma)
                new_action_log_probs = dist.log_prob(actions[indices])
                
                # 重要性采样比率
                ratio = torch.exp(new_action_log_probs - old_action_log_probs[indices])
                
                # PPO裁剪目标
                surr1 = ratio * advantages[indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 
                                  1.0 + self.clip_param) * advantages[indices]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                
                # Critic更新
                current_values_batch = self.critic(states[indices])
                critic_loss = F.smooth_l1_loss(current_values_batch, target_values[indices])
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                critic_losses.append(critic_loss.item())
        
        # 清空缓冲区
        self.buffer.clear()
        self.counter = 0
        
        # 返回统计信息
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'mean_advantage': advantages.mean().item(),
            'mean_target_value': target_values.mean().item()
        }
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        
        print(f"Model loaded from {path}")
    
    def get_statistics(self) -> dict:
        """
        获取训练统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.buffer),
            'device': str(self.device)
        }


def test_ppo_base():
    """测试基础PPO实现"""
    print("Testing PPO Base Implementation...")
    
    # 参数设置
    state_dim = 10
    action_dim = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建PPO实例
    ppo = PPOBase(state_dim, action_dim, device=device)
    
    # 测试动作选择
    state = np.random.randn(state_dim)
    action, log_prob = ppo.select_action(state)
    print(f"Action shape: {action.shape}, Log prob shape: {log_prob.shape}")
    
    # 测试经验存储和更新
    for i in range(100):  # 收集一些经验
        next_state = np.random.randn(state_dim)
        reward = np.random.randn()
        
        transition = Transition(state, action, log_prob, reward, next_state)
        should_update = ppo.store_transition(transition)
        
        if should_update:
            stats = ppo.update()
            print(f"Update statistics: {stats}")
            break
        
        # 下一步
        state = next_state
        action, log_prob = ppo.select_action(state)
    
    # 测试统计信息
    stats = ppo.get_statistics()
    print(f"PPO statistics: {stats}")
    
    print("PPO Base test completed!")


if __name__ == "__main__":
    test_ppo_base()