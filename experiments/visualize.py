"""
可视化模块 - 生成训练曲线和性能对比图表
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_training_log(log_file: str) -> pd.DataFrame:
    """
    加载训练日志文件
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        包含训练数据的DataFrame
    """
    try:
        df = pd.read_csv(log_file, sep=' ')
        return df
    except Exception as e:
        print(f"加载日志文件失败: {log_file}, 错误: {e}")
        return None


def plot_training_curves(log_files: Dict[str, str], output_dir: str = 'plots/'):
    """
    绘制训练曲线
    
    Args:
        log_files: 算法名称到日志文件路径的映射
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GNN-PPO vs PPO Baseline Training Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (alg_name, log_file) in enumerate(log_files.items()):
        df = load_training_log(log_file)
        if df is None:
            continue
            
        color = colors[i % len(colors)]
        
        # 1. 奖励曲线
        axes[0, 0].plot(df['episode'], df['mean_reward'], 
                       label=alg_name, color=color, linewidth=2)
        axes[0, 0].set_title('Training Reward', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 延迟曲线
        axes[0, 1].plot(df['episode'], df['mean_delay'], 
                       label=alg_name, color=color, linewidth=2)
        axes[0, 1].set_title('Average Delay', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Delay (ms)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 成功率曲线
        axes[1, 0].plot(df['episode'], df['mean_rate'], 
                       label=alg_name, color=color, linewidth=2)
        axes[1, 0].set_title('Success Rate', fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 收敛性分析 - 移动平均
        window = 50
        smooth_reward = df['mean_reward'].rolling(window=window).mean()
        axes[1, 1].plot(df['episode'], smooth_reward, 
                       label=f'{alg_name} (MA-{window})', color=color, linewidth=2)
    
    axes[1, 1].set_title('Convergence Analysis', fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Smoothed Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'training_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练对比图保存至: {plot_path}")
    
    plt.show()


def plot_performance_comparison(log_files: Dict[str, str], output_dir: str = 'plots/'):
    """
    绘制性能对比柱状图
    
    Args:
        log_files: 算法名称到日志文件路径的映射
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集最终性能指标
    performance_data = []
    
    for alg_name, log_file in log_files.items():
        df = load_training_log(log_file)
        if df is None:
            continue
            
        # 取最后100个episode的平均值
        final_metrics = {
            'Algorithm': alg_name,
            'Final Reward': df['mean_reward'].tail(100).mean(),
            'Final Delay': df['mean_delay'].tail(100).mean(),
            'Final Success Rate': df['mean_rate'].tail(100).mean(),
        }
        performance_data.append(final_metrics)
    
    if not performance_data:
        print("没有可用的性能数据")
        return
    
    perf_df = pd.DataFrame(performance_data)
    
    # 创建对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Final Performance Comparison (Last 100 Episodes Average)', 
                fontsize=16, fontweight='bold')
    
    # 1. 奖励对比
    bars1 = axes[0].bar(perf_df['Algorithm'], perf_df['Final Reward'], 
                       color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    axes[0].set_title('Final Average Reward', fontweight='bold')
    axes[0].set_ylabel('Reward')
    
    # 添加数值标签
    for bar, value in zip(bars1, perf_df['Final Reward']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 延迟对比
    bars2 = axes[1].bar(perf_df['Algorithm'], perf_df['Final Delay'], 
                       color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    axes[1].set_title('Final Average Delay', fontweight='bold')
    axes[1].set_ylabel('Delay (ms)')
    
    for bar, value in zip(bars2, perf_df['Final Delay']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 成功率对比
    bars3 = axes[2].bar(perf_df['Algorithm'], perf_df['Final Success Rate'], 
                       color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    axes[2].set_title('Final Success Rate', fontweight='bold')
    axes[2].set_ylabel('Success Rate')
    
    for bar, value in zip(bars3, perf_df['Final Success Rate']):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"性能对比图保存至: {plot_path}")
    
    # 计算改进百分比
    if len(performance_data) >= 2:
        baseline = performance_data[0]  # 假设第一个是基线
        gnn_ppo = performance_data[1]   # 第二个是GNN-PPO
        
        reward_improvement = (gnn_ppo['Final Reward'] - baseline['Final Reward']) / abs(baseline['Final Reward']) * 100
        delay_improvement = (baseline['Final Delay'] - gnn_ppo['Final Delay']) / baseline['Final Delay'] * 100
        rate_improvement = (gnn_ppo['Final Success Rate'] - baseline['Final Success Rate']) / baseline['Final Success Rate'] * 100
        
        print(f"\\n📊 GNN-PPO相比PPO基线的改进:")
        print(f"🎯 奖励改进: {reward_improvement:+.1f}%")
        print(f"⚡ 延迟降低: {delay_improvement:+.1f}%")
        print(f"✅ 成功率提升: {rate_improvement:+.1f}%")
    
    plt.show()


def plot_convergence_analysis(log_files: Dict[str, str], output_dir: str = 'plots/'):
    """
    绘制详细的收敛性分析图
    
    Args:
        log_files: 算法名称到日志文件路径的映射
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Convergence Analysis: GNN-PPO vs PPO Baseline', 
                fontsize=16, fontweight='bold')
    
    for alg_name, log_file in log_files.items():
        df = load_training_log(log_file)
        if df is None:
            continue
        
        color = '#1f77b4' if 'baseline' in alg_name.lower() else '#ff7f0e'
        
        # 1. 奖励方差分析
        window = 50
        rolling_std = df['mean_reward'].rolling(window=window).std()
        axes[0, 0].plot(df['episode'], rolling_std, label=f'{alg_name} Variance', 
                       color=color, linewidth=2)
        axes[0, 0].set_title('Reward Variance (Stability)', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Rolling Std')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 学习率分析 (奖励变化率)
        reward_diff = df['mean_reward'].diff().abs()
        smooth_diff = reward_diff.rolling(window=window).mean()
        axes[0, 1].plot(df['episode'][1:], smooth_diff[1:], label=f'{alg_name} Learning Rate', 
                       color=color, linewidth=2)
        axes[0, 1].set_title('Learning Rate (Reward Change)', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Abs Reward Change')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 累积性能分析
        cumulative_reward = df['mean_reward'].cumsum()
        axes[1, 0].plot(df['episode'], cumulative_reward, label=f'{alg_name} Cumulative', 
                       color=color, linewidth=2)
        axes[1, 0].set_title('Cumulative Reward', fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 多指标综合得分
        # 标准化各指标到0-1范围
        norm_reward = (df['mean_reward'] - df['mean_reward'].min()) / (df['mean_reward'].max() - df['mean_reward'].min())
        norm_delay = 1 - (df['mean_delay'] - df['mean_delay'].min()) / (df['mean_delay'].max() - df['mean_delay'].min())  # 延迟越低越好
        norm_rate = (df['mean_rate'] - df['mean_rate'].min()) / (df['mean_rate'].max() - df['mean_rate'].min())
        
        # 综合得分 (权重: 奖励40%, 延迟30%, 成功率30%)
        composite_score = 0.4 * norm_reward + 0.3 * norm_delay + 0.3 * norm_rate
        smooth_score = composite_score.rolling(window=window).mean()
        
        axes[1, 1].plot(df['episode'], smooth_score, label=f'{alg_name} Composite', 
                       color=color, linewidth=2)
        axes[1, 1].set_title('Composite Performance Score', fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Normalized Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'convergence_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"收敛性分析图保存至: {plot_path}")
    
    plt.show()


def create_summary_report(log_files: Dict[str, str], output_dir: str = 'plots/'):
    """
    创建实验总结报告
    
    Args:
        log_files: 算法名称到日志文件路径的映射
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\\n" + "="*80)
    print("📊 GNN-PPO vs PPO Baseline 实验总结报告")
    print("="*80)
    
    summary_data = []
    
    for alg_name, log_file in log_files.items():
        df = load_training_log(log_file)
        if df is None:
            continue
        
        # 计算关键统计指标
        final_100 = df.tail(100)
        
        stats = {
            '算法': alg_name,
            '最终平均奖励': final_100['mean_reward'].mean(),
            '最终平均延迟': final_100['mean_delay'].mean(),
            '最终成功率': final_100['mean_rate'].mean(),
            '奖励标准差': final_100['mean_reward'].std(),
            '最佳奖励': df['mean_reward'].max(),
            '最低延迟': df['mean_delay'].min(),
            '最高成功率': df['mean_rate'].max(),
            '收敛episode': len(df)
        }
        
        summary_data.append(stats)
    
    # 打印对比表格
    if len(summary_data) >= 2:
        baseline = summary_data[0]
        gnn_ppo = summary_data[1]
        
        print(f"\\n{'指标':<20} {'PPO基线':<15} {'GNN-PPO':<15} {'改进幅度':<15}")
        print("-" * 70)
        
        metrics = [
            ('最终平均奖励', 'mean_reward'),
            ('最终平均延迟', 'mean_delay'), 
            ('最终成功率', 'mean_rate'),
            ('奖励稳定性', 'reward_std')
        ]
        
        improvements = {}
        
        for metric_name, key in metrics:
            if key == 'mean_delay':
                # 延迟越低越好
                baseline_val = baseline['最终平均延迟']
                gnn_val = gnn_ppo['最终平均延迟']
                improvement = (baseline_val - gnn_val) / baseline_val * 100
                improvements[metric_name] = improvement
                print(f"{metric_name:<20} {baseline_val:<15.2f} {gnn_val:<15.2f} {improvement:<15.1f}%")
            elif key == 'reward_std':
                # 标准差越小越稳定
                baseline_val = baseline['奖励标准差']
                gnn_val = gnn_ppo['奖励标准差']
                improvement = (baseline_val - gnn_val) / baseline_val * 100
                improvements[metric_name] = improvement
                print(f"{metric_name:<20} {baseline_val:<15.3f} {gnn_val:<15.3f} {improvement:<15.1f}%")
            else:
                # 其他指标越大越好
                baseline_val = baseline[f'最终{metric_name[2:]}'] if metric_name.startswith('最终') else baseline[metric_name]
                gnn_val = gnn_ppo[f'最终{metric_name[2:]}'] if metric_name.startswith('最终') else gnn_ppo[metric_name]
                improvement = (gnn_val - baseline_val) / abs(baseline_val) * 100
                improvements[metric_name] = improvement
                print(f"{metric_name:<20} {baseline_val:<15.3f} {gnn_val:<15.3f} {improvement:<15.1f}%")
        
        print("\\n🎯 关键发现:")
        print(f"• GNN-PPO在奖励上{'优于' if improvements.get('最终平均奖励', 0) > 0 else '不如'}PPO基线 ({improvements.get('最终平均奖励', 0):.1f}%)")
        print(f"• GNN-PPO在延迟上{'优于' if improvements.get('最终平均延迟', 0) > 0 else '不如'}PPO基线 ({improvements.get('最终平均延迟', 0):.1f}%)")
        print(f"• GNN-PPO在成功率上{'优于' if improvements.get('最终成功率', 0) > 0 else '不如'}PPO基线 ({improvements.get('最终成功率', 0):.1f}%)")
        print(f"• GNN-PPO{'更' if improvements.get('奖励稳定性', 0) > 0 else '不够'}稳定 ({improvements.get('奖励稳定性', 0):.1f}%)")
    
    # 保存报告到文件
    report_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GNN-PPO vs PPO Baseline 实验总结报告\\n")
        f.write("="*50 + "\\n\\n")
        
        for stats in summary_data:
            f.write(f"算法: {stats['算法']}\\n")
            for key, value in stats.items():
                if key != '算法':
                    f.write(f"  {key}: {value:.4f}\\n")
            f.write("\\n")
    
    print(f"\\n📝 详细报告保存至: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--baseline_log', type=str, 
                       default='experiment_plot_new/PPO_baseline_15_5.log',
                       help='PPO基线日志文件')
    parser.add_argument('--gnn_log', type=str,
                       default='experiment_plot_new/GNN_PPO_GAT_15_5.log', 
                       help='GNN-PPO日志文件')
    parser.add_argument('--output_dir', type=str, default='plots/',
                       help='输出目录')
    parser.add_argument('--plot_type', type=str, default='all',
                       choices=['curves', 'comparison', 'convergence', 'all'],
                       help='绘制图表类型')
    
    args = parser.parse_args()
    
    # 检查日志文件是否存在
    log_files = {
        'PPO Baseline': args.baseline_log,
        'GNN-PPO': args.gnn_log
    }
    
    existing_logs = {}
    for name, path in log_files.items():
        if os.path.exists(path):
            existing_logs[name] = path
        else:
            print(f"⚠️  日志文件不存在: {path}")
    
    if not existing_logs:
        print("❌ 没有找到可用的日志文件")
        return
    
    print(f"📊 找到 {len(existing_logs)} 个日志文件，开始生成图表...")
    
    # 生成图表
    if args.plot_type in ['curves', 'all']:
        plot_training_curves(existing_logs, args.output_dir)
    
    if args.plot_type in ['comparison', 'all']:
        plot_performance_comparison(existing_logs, args.output_dir)
    
    if args.plot_type in ['convergence', 'all']:
        plot_convergence_analysis(existing_logs, args.output_dir)
    
    if args.plot_type == 'all':
        create_summary_report(existing_logs, args.output_dir)
    
    print("\\n✅ 图表生成完成！")


if __name__ == '__main__':
    main()