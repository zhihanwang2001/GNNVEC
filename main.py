#!/usr/bin/env python3
"""
GNN_PPO_VEC主入口文件
图神经网络增强的车联网边缘计算任务调度系统
"""

import argparse
import sys
import os
import yaml
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# from models.gnn_ppo import main as gnn_ppo_main  # 延迟导入避免argparse冲突
from utils.env_utils import test_environment_adapter
from utils.graph_utils import test_graph_construction
from utils.data_utils import test_ngsim_processor


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = "configs/default.yaml"
    
    config_file = Path(__file__).parent / config_path
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_file}")
    else:
        # 默认配置
        config = {
            'training': {
                'episodes': 500,
                'learning_rate': 8e-4,
                'gamma': 0.9,
                'batch_size': 256,
                'update_iteration': 20
            },
            'gnn': {
                'type': 'GAT',
                'hidden_dim': 64,
                'output_dim': 128,
                'layers': 2
            },
            'environment': {
                'num_car': 20,
                'num_tcar': 15,
                'num_scar': 5,
                'num_task': 15,
                'num_uav': 1,
                'num_rsu': 1
            }
        }
        print("Using default configuration")
    
    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GNN-Enhanced PPO for Vehicular Edge Computing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 训练模型（使用默认配置）
  python main.py --mode train
  
  # 训练模型（使用自定义配置）
  python main.py --mode train --config configs/gnn_config.yaml
  
  # 测试模型
  python main.py --mode test --model saved_models/gnn_ppo_best.pth
  
  # 运行测试套件
  python main.py --mode test_components
  
  # 处理NGSIM数据
  python main.py --mode process_data --data_path data/ngsim/trajectory.csv
        """
    )
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'test_components', 'process_data', 
                               'train_baseline', 'compare', 'visualize', 'evaluate'],
                       help='运行模式')
    
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件路径（测试模式使用）')
    
    parser.add_argument('--data_path', type=str, default=None,
                       help='NGSIM数据文件路径')
    
    parser.add_argument('--output_dir', type=str, default='results/',
                       help='结果输出目录')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    return parser.parse_args()


def test_all_components():
    """测试所有组件功能"""
    print("=" * 50)
    print("GNN_PPO_VEC 组件测试")
    print("=" * 50)
    
    try:
        print("\\n1. 测试环境适配器...")
        test_environment_adapter()
        print("✓ 环境适配器测试通过")
    except Exception as e:
        print(f"✗ 环境适配器测试失败: {e}")
    
    try:
        print("\\n2. 测试图构建功能...")
        test_graph_construction()
        print("✓ 图构建测试通过")
    except Exception as e:
        print(f"✗ 图构建测试失败: {e}")
    
    try:
        print("\\n3. 测试数据处理器...")
        test_ngsim_processor()
        print("✓ 数据处理器测试通过")
    except Exception as e:
        print(f"✗ 数据处理器测试失败: {e}")
    
    print("\\n" + "=" * 50)
    print("组件测试完成")
    print("=" * 50)


def process_ngsim_data(data_path: str, output_dir: str):
    """处理NGSIM数据"""
    print(f"处理NGSIM数据: {data_path}")
    
    from utils.data_utils import NGSIMDataProcessor
    
    processor = NGSIMDataProcessor()
    
    # 加载数据
    df = processor.load_ngsim_data(data_path) if data_path else processor._generate_synthetic_trajectory_data()
    
    # 处理轨迹
    trajectories = processor.process_trajectories(df)
    
    # 生成图快照
    snapshots = processor.generate_graph_snapshots()
    
    # 创建数据集
    dataset = processor.create_training_dataset(snapshots)
    
    # 保存数据集
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, 'processed_dataset.pkl')
    processor.save_dataset(dataset, dataset_path)
    
    # 保存统计信息
    stats = processor.get_data_statistics(dataset)
    stats_path = os.path.join(output_dir, 'dataset_stats.yaml')
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f)
    
    print(f"数据处理完成，结果保存在: {output_dir}")


def main():
    """主函数"""
    args = parse_args()
    
    print("GNN_PPO_VEC: 图神经网络增强的车联网边缘计算任务调度")
    print("-" * 60)
    
    if args.mode == 'test_components':
        test_all_components()
    elif args.mode == 'process_data':
        process_ngsim_data(args.data_path, args.output_dir)
    elif args.mode == 'train_baseline':
        # 直接调用函数避免argparse冲突
        config = load_config('configs/ppo_baseline.yaml')
        
        # 导入并调用训练函数
        from experiments.train import train_baseline_ppo
        train_baseline_ppo(config)
    elif args.mode == 'compare':
        # 直接调用对比训练函数，避免argparse冲突
        from experiments.train import compare_training_runs
        compare_training_runs()
    elif args.mode == 'visualize':
        # 直接调用可视化函数，避免argparse冲突
        from experiments.visualize import plot_training_curves, plot_performance_comparison, plot_convergence_analysis, create_summary_report
        
        # 设置默认日志文件路径
        log_files = {}
        baseline_log = 'experiment_plot_new/PPO_baseline_15_5.log'
        gnn_log = 'experiment_plot_new/GNN_PPO_GAT_15_5.log'
        output_dir = 'plots/'
        
        # 检查文件是否存在
        if os.path.exists(baseline_log):
            log_files['PPO Baseline'] = baseline_log
        if os.path.exists(gnn_log):
            log_files['GNN-PPO'] = gnn_log
        
        if not log_files:
            print("❌ 没有找到训练日志文件")
            print(f"期望的日志文件:")
            print(f"  PPO基线: {baseline_log}")
            print(f"  GNN-PPO: {gnn_log}")
            return
        
        print(f"📊 找到 {len(log_files)} 个日志文件，开始生成图表...")
        for name, path in log_files.items():
            print(f"  - {name}: {path}")
        
        # 生成图表
        plot_training_curves(log_files, output_dir)
        plot_performance_comparison(log_files, output_dir)
        plot_convergence_analysis(log_files, output_dir)
        create_summary_report(log_files, output_dir)
        
        print("\\n✅ 可视化图表生成完成！")
    elif args.mode == 'evaluate':
        # 直接调用评估函数，避免argparse冲突  
        from experiments.evaluate import evaluate_models
        evaluate_models()
    elif args.mode in ['train', 'test']:
        # 加载配置
        config = load_config(args.config)
        
        # 设置环境变量传递参数给gnn_ppo模块
        os.environ['GNN_PPO_CONFIG'] = yaml.dump(config)
        
        # 构建sys.argv给gnn_ppo模块使用
        gnn_args = ['gnn_ppo.py', '--mode', args.mode]
        
        # 传递配置参数到gnn_ppo模块
        if 'gnn' in config and config['gnn']['type']:
            gnn_args.extend(['--gnn_type', str(config['gnn']['type'])])
        if 'training' in config:
            if 'learning_rate' in config['training']:
                gnn_args.extend(['--learning_rate', str(config['training']['learning_rate'])])
            if 'batch_size' in config['training']:
                gnn_args.extend(['--batch_size', str(config['training']['batch_size'])])
            if 'update_iteration' in config['training']:
                gnn_args.extend(['--update_iteration', str(config['training']['update_iteration'])])
            if 'capacity' in config['training']:
                gnn_args.extend(['--capacity', str(config['training']['capacity'])])
            if 'gamma' in config['training']:
                gnn_args.extend(['--gamma', str(config['training']['gamma'])])
        if 'gnn' in config:
            if 'hidden_dim' in config['gnn']:
                gnn_args.extend(['--gnn_hidden_dim', str(config['gnn']['hidden_dim'])])
            if 'output_dim' in config['gnn']:
                gnn_args.extend(['--gnn_output_dim', str(config['gnn']['output_dim'])])
            if 'layers' in config['gnn']:
                gnn_args.extend(['--gnn_layers', str(config['gnn']['layers'])])
        if 'experiment' in config:
            if 'random_seed' in config['experiment']:
                gnn_args.extend(['--random_seed', str(config['experiment']['random_seed'])])
            if 'seed' in config['experiment']:
                gnn_args.extend(['--seed', str(config['experiment']['seed']).lower()])
        
        # 备份原始sys.argv
        original_argv = sys.argv
        sys.argv = gnn_args
        
        try:
            # 延迟导入避免argparse冲突
            from models.gnn_ppo import main as gnn_ppo_main
            # 运行GNN-PPO主函数
            gnn_ppo_main()
        except KeyboardInterrupt:
            print("\\n训练被用户中断")
        except Exception as e:
            print(f"运行出错: {e}")
            raise
        finally:
            # 恢复原始sys.argv
            sys.argv = original_argv
    
    print("\\n程序结束")


if __name__ == '__main__':
    main()