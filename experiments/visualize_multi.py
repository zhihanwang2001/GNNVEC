"""
多算法综合对比可视化
支持PPO基线、GNN-GAT、GNN-GCN三算法全面对比
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import os
from typing import Dict, List
from scipy import stats

# 设置样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")


def load_training_log(log_file: str) -> pd.DataFrame:
    """加载训练日志"""
    try:
        df = pd.read_csv(log_file, sep=' ')
        return df
    except Exception as e:
        print(f"加载日志文件失败: {log_file}, 错误: {e}")
        return None


def plot_comprehensive_comparison(log_files: Dict[str, str], output_dir: str = 'plots/comprehensive/'):
    """
    生成三算法综合对比图表
    
    Args:
        log_files: 算法名称到日志文件路径的映射
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 训练过程全面对比
    fig = plt.figure(figsize=(20, 12))
    
    # 设置子图布局 (2行3列)
    axes = [
        plt.subplot(2, 3, 1),  # 奖励对比
        plt.subplot(2, 3, 2),  # 延迟对比  
        plt.subplot(2, 3, 3),  # 成功率对比
        plt.subplot(2, 3, 4),  # 收敛速度对比
        plt.subplot(2, 3, 5),  # 稳定性分析
        plt.subplot(2, 3, 6),  # 综合性能雷达图
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    algorithm_data = {}
    
    # 加载所有数据
    for i, (alg_name, log_file) in enumerate(log_files.items()):
        df = load_training_log(log_file)
        if df is None:
            continue
            
        algorithm_data[alg_name] = df
        color = colors[i % len(colors)]
        
        # 1. 奖励曲线
        axes[0].plot(df['episode'], df['mean_reward'], 
                    label=alg_name, color=color, linewidth=2.5, alpha=0.8)
        
        # 2. 延迟曲线
        axes[1].plot(df['episode'], df['mean_delay'], 
                    label=alg_name, color=color, linewidth=2.5, alpha=0.8)
        
        # 3. 成功率曲线
        axes[2].plot(df['episode'], df['mean_rate'], 
                    label=alg_name, color=color, linewidth=2.5, alpha=0.8)
        
        # 4. 收敛速度分析 (奖励导数)
        reward_diff = df['mean_reward'].diff().abs()
        smooth_diff = reward_diff.rolling(window=20).mean()
        axes[3].plot(df['episode'][1:], smooth_diff[1:], 
                    label=f'{alg_name} 收敛速度', color=color, linewidth=2, alpha=0.7)
        
        # 5. 稳定性分析 (奖励方差)
        rolling_std = df['mean_reward'].rolling(window=50).std()
        axes[4].plot(df['episode'], rolling_std, 
                    label=f'{alg_name} 稳定性', color=color, linewidth=2, alpha=0.7)
    
    # 设置子图样式
    titles = ['Training Reward Comparison', 'Average Delay Comparison', 
              'Success Rate Comparison', 'Convergence Speed Analysis', 
              'Training Stability Analysis']
    ylabels = ['Average Reward', 'Delay (ms)', 'Success Rate', 
               'Reward Change Rate', 'Reward Std Dev']
    
    for i in range(5):
        axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Episode', fontsize=12)
        axes[i].set_ylabel(ylabels[i], fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    # 6. 综合性能雷达图
    if len(algorithm_data) >= 2:
        plot_radar_chart(algorithm_data, axes[5])
    
    plt.suptitle('Comprehensive Algorithm Comparison: PPO vs GNN-PPO Variants', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"综合对比图保存至: {plot_path}")
    
    plt.show()


def plot_radar_chart(algorithm_data: Dict[str, pd.DataFrame], ax):
    """绘制综合性能雷达图"""
    
    # 计算各算法的最终性能指标
    metrics = {}
    metric_names = ['Final Reward', 'Low Delay', 'High Success Rate', 
                   'Fast Convergence', 'High Stability']
    
    for alg_name, df in algorithm_data.items():
        final_100 = df.tail(100)
        
        # 标准化指标 (0-1范围)
        reward_score = (final_100['mean_reward'].mean() - df['mean_reward'].min()) / \
                      (df['mean_reward'].max() - df['mean_reward'].min())
        
        delay_score = 1 - (final_100['mean_delay'].mean() - df['mean_delay'].min()) / \
                     (df['mean_delay'].max() - df['mean_delay'].min())  # 延迟越低越好
        
        rate_score = (final_100['mean_rate'].mean() - df['mean_rate'].min()) / \
                    (df['mean_rate'].max() - df['mean_rate'].min())
        
        # 收敛速度 (前50%达到最终性能90%的速度)
        final_reward = final_100['mean_reward'].mean()
        target_reward = final_reward * 0.9
        convergence_episode = len(df[df['mean_reward'] < target_reward])
        convergence_score = 1 - convergence_episode / len(df)
        
        # 稳定性 (最后100个episode的方差，越小越好)
        stability_score = 1 - (final_100['mean_reward'].std() / df['mean_reward'].std())
        
        metrics[alg_name] = [reward_score, delay_score, rate_score, 
                           convergence_score, stability_score]
    
    # 绘制雷达图
    angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (alg_name, values) in enumerate(metrics.items()):
        values += values[:1]  # 闭合数据
        color = colors[i % len(colors)]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=alg_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Comprehensive Performance Radar', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)


def plot_statistical_comparison(log_files: Dict[str, str], output_dir: str = 'plots/comprehensive/'):
    """
    统计显著性对比分析
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    algorithm_data = {}
    for alg_name, log_file in log_files.items():
        df = load_training_log(log_file)
        if df is not None:
            algorithm_data[alg_name] = df.tail(100)  # 最后100个episodes
    
    if len(algorithm_data) < 2:
        print("数据不足，无法进行统计对比")
        return
    
    # 创建对比矩阵图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')
    
    algorithms = list(algorithm_data.keys())
    metrics = ['mean_reward', 'mean_delay', 'mean_rate']
    metric_names = ['Average Reward', 'Average Delay', 'Success Rate']
    
    # 1. 箱线图对比
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2] if i < 3 else axes[1, 1]
        
        data_for_box = []
        labels_for_box = []
        
        for alg_name in algorithms:
            if metric in algorithm_data[alg_name].columns:
                data_for_box.append(algorithm_data[alg_name][metric].values)
                labels_for_box.append(alg_name)
        
        if data_for_box:
            bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{name} Distribution', fontweight='bold')
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
    
    # 4. 统计显著性矩阵
    if len(algorithms) >= 2:
        ax = axes[1, 1]
        
        # 创建p值矩阵
        n_algs = len(algorithms)
        p_matrix = np.ones((n_algs, n_algs))
        
        for i in range(n_algs):
            for j in range(i+1, n_algs):
                alg1, alg2 = algorithms[i], algorithms[j]
                
                # 对奖励进行t检验
                data1 = algorithm_data[alg1]['mean_reward'].values
                data2 = algorithm_data[alg2]['mean_reward'].values
                
                _, p_value = stats.ttest_ind(data1, data2)
                p_matrix[i, j] = p_value
                p_matrix[j, i] = p_value
        
        # 绘制热力图
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
        
        # 添加文本注释
        for i in range(n_algs):
            for j in range(n_algs):
                text = ax.text(j, i, f'{p_matrix[i, j]:.4f}', 
                             ha='center', va='center', fontweight='bold',
                             color='white' if p_matrix[i, j] < 0.05 else 'black')
        
        ax.set_xticks(range(n_algs))
        ax.set_yticks(range(n_algs))
        ax.set_xticklabels(algorithms, rotation=45)
        ax.set_yticklabels(algorithms)
        ax.set_title('Statistical Significance (p-values)', fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('p-value', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'statistical_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"统计分析图保存至: {plot_path}")
    
    plt.show()


def create_comprehensive_report(log_files: Dict[str, str], output_dir: str = 'plots/comprehensive/'):
    """
    创建综合实验报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\\n" + "="*100)
    print("📊 GNN-PPO 多算法综合对比实验报告")
    print("="*100)
    
    # 收集数据
    algorithm_results = {}
    for alg_name, log_file in log_files.items():
        df = load_training_log(log_file)
        if df is not None:
            final_100 = df.tail(100)
            
            algorithm_results[alg_name] = {
                '最终平均奖励': final_100['mean_reward'].mean(),
                '奖励标准差': final_100['mean_reward'].std(),
                '最终平均延迟': final_100['mean_delay'].mean(),
                '延迟标准差': final_100['mean_delay'].std(),
                '最终成功率': final_100['mean_rate'].mean(),
                '成功率标准差': final_100['mean_rate'].std(),
                '最佳奖励': df['mean_reward'].max(),
                '最低延迟': df['mean_delay'].min(),
                '最高成功率': df['mean_rate'].max(),
                '数据点数': len(df)
            }
    
    # 打印对比表格
    if len(algorithm_results) >= 2:
        print(f"\\n{'算法':<20} {'最终奖励':<15} {'最终延迟':<15} {'成功率':<15} {'奖励稳定性':<15}")
        print("-" * 80)
        
        for alg_name, results in algorithm_results.items():
            print(f"{alg_name:<20} "
                  f"{results['最终平均奖励']:<15.3f} "
                  f"{results['最终平均延迟']:<15.2f} "
                  f"{results['最终成功率']:<15.3f} "
                  f"{results['奖励标准差']:<15.4f}")
        
        # 计算相对改进
        if len(algorithm_results) >= 2:
            algorithms = list(algorithm_results.keys())
            baseline = None
            gnn_algorithms = []
            
            for alg_name in algorithms:
                if 'baseline' in alg_name.lower() or 'ppo' in alg_name.lower():
                    baseline = algorithm_results[alg_name]
                    baseline_name = alg_name
                elif 'gnn' in alg_name.lower():
                    gnn_algorithms.append((alg_name, algorithm_results[alg_name]))
            
            if baseline and gnn_algorithms:
                print(f"\\n🎯 相对于{baseline_name}的改进:")
                print("-" * 50)
                
                for gnn_name, gnn_results in gnn_algorithms:
                    reward_improvement = (gnn_results['最终平均奖励'] - baseline['最终平均奖励']) / abs(baseline['最终平均奖励']) * 100
                    delay_improvement = (baseline['最终平均延迟'] - gnn_results['最终平均延迟']) / baseline['最终平均延迟'] * 100
                    rate_improvement = (gnn_results['最终成功率'] - baseline['最终成功率']) / baseline['最终成功率'] * 100
                    
                    print(f"{gnn_name}:")
                    print(f"  📈 奖励改进: {reward_improvement:+.2f}%")
                    print(f"  ⚡ 延迟降低: {delay_improvement:+.2f}%")
                    print(f"  ✅ 成功率提升: {rate_improvement:+.2f}%")
                    print()
    
    # 保存详细报告
    report_path = os.path.join(output_dir, 'comprehensive_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GNN-PPO 多算法综合对比实验报告\\n")
        f.write("="*50 + "\\n\\n")
        
        for alg_name, results in algorithm_results.items():
            f.write(f"算法: {alg_name}\\n")
            for key, value in results.items():
                f.write(f"  {key}: {value:.4f}\\n")
            f.write("\\n")
    
    print(f"📄 详细报告保存至: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多算法综合对比可视化')
    parser.add_argument('--logs', type=str, nargs='+', required=True,
                       help='算法名称:日志路径 格式，例如 PPO:log1.log GNN_GAT:log2.log')
    parser.add_argument('--output_dir', type=str, default='plots/comprehensive/',
                       help='输出目录')
    parser.add_argument('--plot_type', type=str, default='all',
                       choices=['comparison', 'statistical', 'report', 'all'],
                       help='绘制类型')
    
    args = parser.parse_args()
    
    # 解析日志文件参数
    log_files = {}
    for log_arg in args.logs:
        if ':' in log_arg:
            alg_name, log_path = log_arg.split(':', 1)
            if os.path.exists(log_path):
                log_files[alg_name] = log_path
            else:
                print(f"⚠️  日志文件不存在: {log_path}")
        else:
            print(f"⚠️  日志参数格式错误: {log_arg}")
    
    if not log_files:
        print("❌ 没有找到可用的日志文件")
        return
    
    print(f"📊 找到 {len(log_files)} 个算法日志，开始分析...")
    for alg_name, log_path in log_files.items():
        print(f"  - {alg_name}: {log_path}")
    
    # 生成分析图表
    if args.plot_type in ['comparison', 'all']:
        plot_comprehensive_comparison(log_files, args.output_dir)
    
    if args.plot_type in ['statistical', 'all']:
        plot_statistical_comparison(log_files, args.output_dir)
    
    if args.plot_type in ['report', 'all']:
        create_comprehensive_report(log_files, args.output_dir)
    
    print("\\n✅ 多算法对比分析完成！")


if __name__ == '__main__':
    main()