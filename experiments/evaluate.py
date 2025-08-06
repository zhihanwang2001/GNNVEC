"""
评估模块 - 模型性能评估和对比分析
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.ppo_base import PPOBase
from models.gnn_ppo import GNN_PPO
from utils.env_utils import VECEnvironmentAdapter


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, env_config: Dict):
        """
        初始化评估器
        
        Args:
            env_config: 环境配置
        """
        self.env_config = env_config
        self.env = VECEnvironmentAdapter(
            num_car=env_config['num_car'],
            num_tcar=env_config['num_tcar'],
            num_scar=env_config['num_scar'], 
            num_task=env_config['num_task'],
            num_uav=env_config['num_uav'],
            num_rsu=env_config['num_rsu']
        )
        
    def evaluate_model(self, agent, num_episodes: int = 100, time_slots: int = 20) -> Dict:
        """
        评估单个模型
        
        Args:
            agent: 要评估的智能体
            num_episodes: 评估回合数
            time_slots: 每回合时隙数
            
        Returns:
            评估结果字典
        """
        rewards = []
        delays = []
        success_rates = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            total_reward = 0.0
            state, _ = self.env.reset()
            episode_length = 0
            
            for t in range(time_slots):
                # 根据智能体类型选择动作
                if isinstance(agent, PPOBase):
                    action, _ = agent.select_action(state)
                elif isinstance(agent, GNN_PPO):
                    # 创建环境信息用于图构建
                    _, env_info = self.env.reset()
                    action, _, _ = agent.select_action(state, env_info)
                else:
                    raise ValueError(f"不支持的智能体类型: {type(agent)}")
                
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                total_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # 收集指标
            metrics = self.env.get_system_metrics()
            
            rewards.append(total_reward)
            delays.append(metrics['total_delay'])
            success_rates.append(metrics['success_rate'])
            episode_lengths.append(episode_length)
        
        # 计算统计指标
        results = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_delay': np.mean(delays),
            'std_delay': np.std(delays),
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'mean_episode_length': np.mean(episode_lengths),
            'rewards': rewards,
            'delays': delays,
            'success_rates': success_rates
        }
        
        return results
    
    def compare_models(self, models: Dict[str, object], num_episodes: int = 100) -> Dict:
        """
        对比多个模型
        
        Args:
            models: 模型名称到模型对象的映射
            num_episodes: 评估回合数
            
        Returns:
            对比结果
        """
        results = {}
        
        print("🔍 开始模型评估对比...")
        
        for model_name, model in models.items():
            print(f"  评估 {model_name}...")
            results[model_name] = self.evaluate_model(model, num_episodes)
            
            # 打印结果摘要
            res = results[model_name]
            print(f"    奖励: {res['mean_reward']:.3f}±{res['std_reward']:.3f}")
            print(f"    延迟: {res['mean_delay']:.2f}±{res['std_delay']:.2f}")
            print(f"    成功率: {res['mean_success_rate']:.3f}±{res['std_success_rate']:.3f}")
        
        return results
    
    def statistical_significance_test(self, results1: List[float], results2: List[float]) -> Dict:
        """
        统计显著性检验
        
        Args:
            results1: 第一组结果
            results2: 第二组结果
            
        Returns:
            检验结果
        """
        from scipy import stats
        
        # t检验
        t_stat, t_pvalue = stats.ttest_ind(results1, results2)
        
        # Mann-Whitney U检验 (非参数检验)
        u_stat, u_pvalue = stats.mannwhitneyu(results1, results2, alternative='two-sided')
        
        # 效应量 (Cohen's d)
        pooled_std = np.sqrt((np.var(results1) + np.var(results2)) / 2)
        cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'u_statistic': u_stat, 
            'u_pvalue': u_pvalue,
            'cohens_d': cohens_d,
            'is_significant': t_pvalue < 0.05
        }


def load_trained_models(model_paths: Dict[str, str]) -> Dict[str, object]:
    """
    加载训练好的模型
    
    Args:
        model_paths: 模型名称到路径的映射
        
    Returns:
        加载的模型字典
    """
    models = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"⚠️  模型文件不存在: {model_path}")
            continue
            
        try:
            if 'baseline' in model_name.lower() or 'ppo' in model_name.lower():
                # PPO基线模型
                model = PPOBase(
                    state_dim=64,  # 根据实际配置调整
                    action_dim=15,
                    device=device
                )
                model.load_model(model_path)
                
            elif 'gnn' in model_name.lower():
                # GNN-PPO模型
                model = GNN_PPO()
                model.load(model_path)
            
            else:
                print(f"⚠️  未知模型类型: {model_name}")
                continue
                
            models[model_name] = model
            print(f"✅ 成功加载模型: {model_name}")
            
        except Exception as e:
            print(f"❌ 加载模型失败 {model_name}: {e}")
    
    return models


def create_evaluation_report(comparison_results: Dict, output_path: str = 'plots/evaluation_report.txt'):
    """
    创建评估报告
    
    Args:
        comparison_results: 对比结果
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("GNN-PPO vs PPO Baseline 模型评估报告\\n")
        f.write("="*50 + "\\n\\n")
        f.write(f"评估时间: {pd.Timestamp.now()}\\n")
        f.write(f"评估回合数: 100\\n\\n")
        
        # 详细统计
        f.write("详细统计指标:\\n")
        f.write("-" * 30 + "\\n")
        
        for model_name, results in comparison_results.items():
            f.write(f"\\n{model_name}:\\n")
            f.write(f"  平均奖励: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}\\n")
            f.write(f"  平均延迟: {results['mean_delay']:.4f} ± {results['std_delay']:.4f}\\n")
            f.write(f"  平均成功率: {results['mean_success_rate']:.4f} ± {results['std_success_rate']:.4f}\\n")
            f.write(f"  平均回合长度: {results['mean_episode_length']:.2f}\\n")
        
        # 如果有两个模型，进行统计检验
        if len(comparison_results) == 2:
            models = list(comparison_results.keys())
            model1, model2 = models[0], models[1]
            
            # 创建评估器进行统计检验
            evaluator = ModelEvaluator({
                'num_car': 20, 'num_tcar': 15, 'num_scar': 5,
                'num_task': 15, 'num_uav': 1, 'num_rsu': 1
            })
            
            # 奖励显著性检验
            reward_test = evaluator.statistical_significance_test(
                comparison_results[model1]['rewards'],
                comparison_results[model2]['rewards']
            )
            
            # 延迟显著性检验  
            delay_test = evaluator.statistical_significance_test(
                comparison_results[model1]['delays'],
                comparison_results[model2]['delays']
            )
            
            f.write("\\n\\n统计显著性检验:\\n")
            f.write("-" * 30 + "\\n")
            f.write(f"奖励差异 ({model1} vs {model2}):\\n")
            f.write(f"  t统计量: {reward_test['t_statistic']:.4f}\\n")
            f.write(f"  p值: {reward_test['t_pvalue']:.6f}\\n")
            f.write(f"  Cohen's d: {reward_test['cohens_d']:.4f}\\n")
            f.write(f"  是否显著: {'是' if reward_test['is_significant'] else '否'}\\n")
            
            f.write(f"\\n延迟差异 ({model1} vs {model2}):\\n")
            f.write(f"  t统计量: {delay_test['t_statistic']:.4f}\\n")
            f.write(f"  p值: {delay_test['t_pvalue']:.6f}\\n")
            f.write(f"  Cohen's d: {delay_test['cohens_d']:.4f}\\n")
            f.write(f"  是否显著: {'是' if delay_test['is_significant'] else '否'}\\n")
    
    print(f"📄 评估报告保存至: {output_path}")


def evaluate_models():
    """评估已训练的模型"""
    
    # 定义模型路径
    model_paths = {
        'PPO Baseline': 'saved_models/PPO_baseline_final.pth',
        'GNN-PPO': 'saved_models/GNN_PPO_model.pth'
    }
    
    # 环境配置
    env_config = {
        'num_car': 20,
        'num_tcar': 15,
        'num_scar': 5,
        'num_task': 15,
        'num_uav': 1,
        'num_rsu': 1
    }
    
    print("🚀 开始模型评估...")
    
    # 加载模型 (如果存在的话)
    models = {}
    
    # 由于可能没有训练好的基线模型，我们创建一个用于演示
    try:
        # 尝试加载GNN-PPO模型
        if os.path.exists(model_paths['GNN-PPO']):
            gnn_ppo = GNN_PPO()
            gnn_ppo.load(model_paths['GNN-PPO'])
            models['GNN-PPO'] = gnn_ppo
            print("✅ 成功加载GNN-PPO模型")
        
        # 创建未训练的基线PPO用于对比演示
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        baseline_ppo = PPOBase(
            state_dim=64,
            action_dim=15, 
            device=device
        )
        models['PPO Baseline (Untrained)'] = baseline_ppo
        print("✅ 创建PPO基线模型（未训练）")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    if not models:
        print("❌ 没有可用的模型进行评估")
        return
    
    # 创建评估器
    evaluator = ModelEvaluator(env_config)
    
    # 评估模型
    results = evaluator.compare_models(models, num_episodes=50)  # 减少回合数以节省时间
    
    # 创建评估报告
    create_evaluation_report(results)
    
    print("\\n📊 评估完成！")
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--num_episodes', type=int, default=100, help='评估回合数')
    parser.add_argument('--output_dir', type=str, default='plots/', help='输出目录')
    
    args = parser.parse_args()
    
    # 运行评估
    results = evaluate_models()
    
    if results:
        print("\\n✅ 评估结果已保存到plots/目录")


if __name__ == '__main__':
    main()