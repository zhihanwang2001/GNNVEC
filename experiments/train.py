"""
训练脚本 - 支持基线对比和GNN-PPO训练
"""

import argparse
import sys
import os
import numpy as np
import torch
import yaml
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.ppo_base import PPOBase, Transition
from models.gnn_ppo import GNN_PPO, Transition as GNNTransition
from utils.env_utils import VECEnvironmentAdapter


def train_baseline_ppo(config: dict, log_file: str = None):
    """
    训练基线PPO算法
    
    Args:
        config: 配置字典
        log_file: 日志文件路径
    """
    print("=" * 60)
    print("训练基线PPO算法")
    print("=" * 60)
    
    # 环境设置
    env_config = config['environment']
    env = VECEnvironmentAdapter(
        num_car=env_config['num_car'],
        num_tcar=env_config['num_tcar'], 
        num_scar=env_config['num_scar'],
        num_task=env_config['num_task'],
        num_uav=env_config['num_uav'],
        num_rsu=env_config['num_rsu']
    )
    
    # 设置随机种子
    if config['experiment']['seed']:
        seed = config['experiment']['random_seed']
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 初始化PPO智能体
    train_config = config['training']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    agent = PPOBase(
        state_dim=env.state_space_shape[0],
        action_dim=env.action_space_shape[0],
        lr=float(train_config['learning_rate']),
        gamma=float(train_config['gamma']),
        clip_param=float(train_config['clip_param']),
        buffer_capacity=int(train_config['capacity']),
        batch_size=int(train_config['batch_size']),
        update_iteration=int(train_config['update_iteration']),
        max_grad_norm=float(train_config['max_grad_norm']),
        device=device
    )
    
    # 训练记录
    reward_record = []
    delay_record = []
    rate_record = []
    
    # 日志文件
    if log_file is None:
        log_file = f'experiment_plot_new/PPO_baseline_{env_config["num_tcar"]}_{env_config["num_scar"]}.log'
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    f = open(log_file, 'w')
    print('episode', 'mean_reward', 'mean_delay', 'mean_rate', file=f)
    
    # 训练循环
    episodes = int(train_config['episodes'])
    time_slots = int(env_config['time_slots'])
    
    for episode in range(episodes):
        total_reward = 0.0
        state, _ = env.reset()
        
        for t in range(time_slots):
            # 选择动作
            action, action_log_prob = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            transition = Transition(state, action, action_log_prob, reward, next_state)
            should_update = agent.store_transition(transition)
            
            # 更新网络
            if should_update:
                agent.update()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 获取系统指标
        metrics = env.get_system_metrics()
        total_delay = metrics['total_delay']
        total_success_rate = metrics['success_rate']
        
        # 记录结果
        reward_record.append(total_reward)
        delay_record.append(total_delay)
        rate_record.append(total_success_rate)
        
        # 计算移动平均
        window = min(100, episode + 1)
        reward_avg = np.mean(reward_record[-window:])
        delay_avg = np.mean(delay_record[-window:])
        rate_avg = np.mean(rate_record[-window:])
        
        # 输出进度
        if episode % int(config['experiment']['log_interval']) == 0:
            print(f"Episode: {episode}, Reward: {reward_avg:.2f}, "
                  f"Delay: {delay_avg:.2f}, Success Rate: {rate_avg:.3f}")
        
        # 写入日志
        print(episode, reward_avg, delay_avg, rate_avg, file=f)
        
        # 定期保存模型
        if episode % int(train_config['save_interval']) == 0 and episode > 0:
            model_path = f'saved_models/PPO_baseline_ep{episode}.pth'
            agent.save_model(model_path)
    
    f.close()
    
    # 保存最终模型
    final_model_path = 'saved_models/PPO_baseline_final.pth'
    agent.save_model(final_model_path)
    
    print(f"基线PPO训练完成！模型保存至: {final_model_path}")
    print(f"训练日志保存至: {log_file}")
    
    return {
        'reward_record': reward_record,
        'delay_record': delay_record,
        'rate_record': rate_record,
        'final_metrics': {
            'avg_reward': np.mean(reward_record[-100:]),
            'avg_delay': np.mean(delay_record[-100:]),
            'avg_rate': np.mean(rate_record[-100:])
        }
    }


def train_gnn_ppo(config: dict, log_file: str = None):
    """
    训练GNN增强PPO算法
    
    Args:
        config: 配置字典
        log_file: 日志文件路径
    """
    print("=" * 60)
    print("训练GNN增强PPO算法")
    print("=" * 60)
    
    # 调用现有的GNN_PPO训练逻辑
    from models.gnn_ppo import main as gnn_ppo_main
    
    # 设置环境变量
    os.environ['GNN_PPO_CONFIG'] = yaml.dump(config)
    
    # 构建参数
    original_argv = sys.argv
    sys.argv = [
        'gnn_ppo.py',
        '--mode', 'train',
        '--gnn_type', config['gnn']['type'],
        '--learning_rate', str(config['training']['learning_rate']),
        '--batch_size', str(config['training']['batch_size']),
        '--gnn_hidden_dim', str(config['gnn']['hidden_dim']),
        '--gnn_output_dim', str(config['gnn']['output_dim'])
    ]
    
    try:
        gnn_ppo_main()
    finally:
        sys.argv = original_argv
    
    print("GNN-PPO训练完成！")


def compare_training_runs():
    """
    运行对比训练实验
    """
    print("=" * 80)
    print("开始对比训练实验：PPO基线 vs GNN-PPO")
    print("=" * 80)
    
    # 加载配置
    baseline_config_path = Path(__file__).parent.parent / 'configs' / 'ppo_baseline.yaml'
    gnn_config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    
    with open(baseline_config_path, 'r') as f:
        baseline_config = yaml.safe_load(f)
    
    with open(gnn_config_path, 'r') as f:
        gnn_config = yaml.safe_load(f)
    
    # 确保使用相同的随机种子
    baseline_config['experiment']['random_seed'] = 9527
    gnn_config['experiment']['random_seed'] = 9527
    
    results = {}
    
    # 训练基线PPO
    print("\n🔵 步骤1: 训练基线PPO...")
    baseline_results = train_baseline_ppo(
        baseline_config, 
        'experiment_plot_new/comparison_PPO_baseline.log'
    )
    results['baseline'] = baseline_results
    
    # 训练GNN-PPO
    print("\n🟢 步骤2: 训练GNN-PPO...")
    train_gnn_ppo(
        gnn_config,
        'experiment_plot_new/comparison_GNN_PPO.log'  
    )
    
    print("\n✅ 对比训练完成！")
    print("📊 可以使用visualize.py生成对比图表")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--algorithm', type=str, default='compare', 
                       choices=['baseline', 'gnn_ppo', 'compare'],
                       help='训练算法类型')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    
    args = parser.parse_args()
    
    if args.algorithm == 'compare':
        compare_training_runs()
    elif args.algorithm == 'baseline':
        config_path = args.config or 'configs/ppo_baseline.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        train_baseline_ppo(config)
    elif args.algorithm == 'gnn_ppo':
        config_path = args.config or 'configs/default.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        train_gnn_ppo(config)


if __name__ == '__main__':
    main()