"""
NGSIM数据处理和图数据生成工具
用于处理车辆轨迹数据，生成训练和测试数据集
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Optional
import os
import pickle
from dataclasses import dataclass


@dataclass
class VehicleTrajectory:
    """车辆轨迹数据结构"""
    vehicle_id: int
    positions: List[Tuple[float, float]]  # (x, y) 位置序列
    timestamps: List[float]  # 时间戳序列
    velocities: List[Tuple[float, float]]  # (vx, vy) 速度序列
    vehicle_type: str  # 'task_vehicle' or 'service_vehicle'


@dataclass
class GraphDataset:
    """图数据集"""
    graphs: List[Data]  # 图数据列表
    states: List[np.ndarray]  # 对应的环境状态
    actions: List[np.ndarray]  # 对应的动作
    rewards: List[float]  # 对应的奖励
    metadata: Dict  # 数据集元信息


class NGSIMDataProcessor:
    """
    NGSIM数据处理器
    处理NGSIM车辆轨迹数据，生成适用于GNN-PPO的训练数据
    """
    
    def __init__(self, data_dir: str = "data/ngsim/"):
        """
        初始化数据处理器
        
        Args:
            data_dir: NGSIM数据目录
        """
        self.data_dir = data_dir
        self.trajectories = []
        self.processed_data = None
        
        # 数据处理参数
        self.time_window = 20  # 时间窗口长度（秒）
        self.sampling_rate = 1.0  # 采样率（秒）
        self.area_bounds = (0, 0, 400, 400)  # (xmin, ymin, xmax, ymax)
        
        # 车辆分类参数
        self.task_vehicle_ratio = 0.75  # 任务车辆比例
        
    def load_ngsim_data(self, file_path: str) -> pd.DataFrame:
        """
        加载NGSIM数据文件
        
        Args:
            file_path: NGSIM数据文件路径
            
        Returns:
            处理后的DataFrame
        """
        if not os.path.exists(file_path):
            print(f"Warning: NGSIM data file {file_path} not found. Using synthetic data.")
            return self._generate_synthetic_trajectory_data()
        
        try:
            # NGSIM数据通常是txt格式，包含以下列：
            # Vehicle_ID, Frame_ID, Total_Frames, Global_Time, Local_X, Local_Y, 
            # Global_X, Global_Y, v_Length, v_Width, v_Class, v_Vel, v_Acc, Lane_ID, etc.
            
            df = pd.read_csv(file_path, delimiter=' ')
            
            # 数据清理和预处理
            df = df.dropna()  # 删除缺失值
            df = df[df['Local_X'].between(*self.area_bounds[:2]) & 
                   df['Local_Y'].between(*self.area_bounds[2:])]  # 区域过滤
            
            print(f"Loaded {len(df)} trajectory points from {file_path}")
            return df
            
        except Exception as e:
            print(f"Error loading NGSIM data: {e}. Using synthetic data.")
            return self._generate_synthetic_trajectory_data()
    
    def _generate_synthetic_trajectory_data(self) -> pd.DataFrame:
        """
        生成合成轨迹数据用于开发和测试
        
        Returns:
            合成的轨迹DataFrame
        """
        print("Generating synthetic vehicle trajectory data...")
        
        data_points = []
        num_vehicles = 50
        frames_per_vehicle = 200
        
        for vehicle_id in range(1, num_vehicles + 1):
            # 随机起始位置
            start_x = np.random.uniform(50, 350)
            start_y = np.random.uniform(50, 350)
            
            # 随机速度和方向
            base_speed = np.random.uniform(10, 30)  # m/s
            direction = np.random.uniform(0, 2*np.pi)
            
            for frame in range(frames_per_vehicle):
                # 添加噪声和方向变化
                noise_x = np.random.normal(0, 2)
                noise_y = np.random.normal(0, 2)
                direction += np.random.normal(0, 0.1)
                
                # 计算位置
                x = start_x + frame * base_speed * np.cos(direction) * 0.1 + noise_x
                y = start_y + frame * base_speed * np.sin(direction) * 0.1 + noise_y
                
                # 边界处理
                x = np.clip(x, 0, 400)
                y = np.clip(y, 0, 400)
                
                # 计算速度
                vx = base_speed * np.cos(direction) + np.random.normal(0, 1)
                vy = base_speed * np.sin(direction) + np.random.normal(0, 1)
                
                data_points.append({
                    'Vehicle_ID': vehicle_id,
                    'Frame_ID': frame,
                    'Global_Time': frame * 0.1,
                    'Local_X': x,
                    'Local_Y': y,
                    'v_Vel': np.sqrt(vx**2 + vy**2),
                    'v_Acc': np.random.normal(0, 0.5),
                    'Lane_ID': np.random.randint(1, 4),
                    'v_Class': np.random.choice([1, 2], p=[0.8, 0.2])  # 1:car, 2:truck
                })
        
        df = pd.DataFrame(data_points)
        print(f"Generated {len(df)} synthetic trajectory points")
        return df
    
    def process_trajectories(self, df: pd.DataFrame) -> List[VehicleTrajectory]:
        """
        处理原始轨迹数据，生成VehicleTrajectory对象
        
        Args:
            df: 原始轨迹DataFrame
            
        Returns:
            VehicleTrajectory对象列表
        """
        trajectories = []
        
        # 按车辆ID分组
        grouped = df.groupby('Vehicle_ID')
        
        for vehicle_id, group in grouped:
            # 按时间排序
            group = group.sort_values('Global_Time')
            
            # 提取轨迹数据
            positions = list(zip(group['Local_X'], group['Local_Y']))
            timestamps = list(group['Global_Time'])
            
            # 计算速度（如果没有直接提供）
            velocities = []
            for i in range(len(positions)):
                if i == 0:
                    vx, vy = 0, 0
                else:
                    dt = timestamps[i] - timestamps[i-1]
                    if dt > 0:
                        dx = positions[i][0] - positions[i-1][0]
                        dy = positions[i][1] - positions[i-1][1]
                        vx, vy = dx/dt, dy/dt
                    else:
                        vx, vy = 0, 0
                velocities.append((vx, vy))
            
            # 分配车辆类型（基于某些规则或随机分配）
            vehicle_type = 'task_vehicle' if np.random.random() < self.task_vehicle_ratio else 'service_vehicle'
            
            trajectory = VehicleTrajectory(
                vehicle_id=vehicle_id,
                positions=positions,
                timestamps=timestamps,
                velocities=velocities,
                vehicle_type=vehicle_type
            )
            
            trajectories.append(trajectory)
        
        self.trajectories = trajectories
        print(f"Processed {len(trajectories)} vehicle trajectories")
        return trajectories
    
    def generate_graph_snapshots(self, time_step: float = 1.0) -> List[Dict]:
        """
        基于车辆轨迹生成图网络快照序列
        
        Args:
            time_step: 时间步长（秒）
            
        Returns:
            图快照信息列表
        """
        if not self.trajectories:
            raise ValueError("No trajectories loaded. Call process_trajectories first.")
        
        # 找到时间范围
        all_times = []
        for traj in self.trajectories:
            all_times.extend(traj.timestamps)
        
        min_time, max_time = min(all_times), max(all_times)
        time_points = np.arange(min_time, max_time, time_step)
        
        graph_snapshots = []
        
        for t in time_points:
            snapshot_info = {
                'timestamp': t,
                'vehicles': [],
                'rsus': [{'id': 0, 'position': [200, 200], 'compute_capacity': 5e8, 'task_queue': []}],
                'uavs': [{'id': 0, 'position': [200, 300], 'compute_capacity': 3e8, 'task_queue': []}]
            }
            
            # 为每个轨迹插值得到t时刻的位置
            for traj in self.trajectories:
                # 找到最接近的时间点
                time_diffs = [abs(ts - t) for ts in traj.timestamps]
                min_idx = np.argmin(time_diffs)
                
                # 如果时间差太大，跳过这个车辆
                if time_diffs[min_idx] > time_step * 2:
                    continue
                
                # 线性插值位置和速度
                if min_idx < len(traj.positions) - 1 and traj.timestamps[min_idx] <= t <= traj.timestamps[min_idx + 1]:
                    # 在两个点之间插值
                    t1, t2 = traj.timestamps[min_idx], traj.timestamps[min_idx + 1]
                    alpha = (t - t1) / (t2 - t1) if t2 > t1 else 0
                    
                    pos1, pos2 = traj.positions[min_idx], traj.positions[min_idx + 1]
                    position = [pos1[0] + alpha * (pos2[0] - pos1[0]), 
                               pos1[1] + alpha * (pos2[1] - pos1[1])]
                    
                    vel1, vel2 = traj.velocities[min_idx], traj.velocities[min_idx + 1]
                    velocity = [vel1[0] + alpha * (vel2[0] - vel1[0]),
                               vel1[1] + alpha * (vel2[1] - vel1[1])]
                else:
                    # 使用最近点的数据
                    position = list(traj.positions[min_idx])
                    velocity = list(traj.velocities[min_idx])
                
                vehicle_info = {
                    'id': traj.vehicle_id,
                    'position': position,
                    'velocity': velocity,
                    'compute_capacity': np.random.uniform(0.5e8, 1e8) if traj.vehicle_type == 'task_vehicle' else np.random.uniform(1e8, 2e8),
                    'task_queue': list(np.random.randint(0, 10, np.random.randint(0, 4))),
                    'type': traj.vehicle_type,
                    'energy_level': np.random.uniform(0.3, 1.0)
                }
                
                snapshot_info['vehicles'].append(vehicle_info)
            
            graph_snapshots.append(snapshot_info)
        
        print(f"Generated {len(graph_snapshots)} graph snapshots")
        return graph_snapshots
    
    def create_training_dataset(self, graph_snapshots: List[Dict], 
                               sequence_length: int = 20) -> GraphDataset:
        """
        创建训练数据集
        
        Args:
            graph_snapshots: 图快照列表
            sequence_length: 序列长度
            
        Returns:
            GraphDataset对象
        """
        from .graph_utils import DynamicVehicleGraph, GraphNeuralNetwork, GraphFeatureExtractor
        
        graph_builder = DynamicVehicleGraph()
        graphs = []
        states = []
        actions = []
        rewards = []
        
        # 滑动窗口生成序列
        for i in range(0, len(graph_snapshots) - sequence_length + 1, sequence_length // 2):
            sequence = graph_snapshots[i:i + sequence_length]
            
            for snapshot in sequence:
                # 构建图数据，修正位置键名
                vehicles_info = {}
                for v in snapshot['vehicles']:
                    v_copy = v.copy()
                    v_copy['pos'] = v_copy.pop('position', [0, 0])  # 将position改为pos
                    vehicles_info[v['id']] = v_copy
                
                rsu_info = {}
                for r in snapshot['rsus']:
                    r_copy = r.copy()
                    r_copy['pos'] = r_copy.pop('position', [0, 0])
                    rsu_info[r['id']] = r_copy
                
                uav_info = {}
                for u in snapshot['uavs']:
                    u_copy = u.copy()
                    u_copy['pos'] = u_copy.pop('position', [0, 0])
                    uav_info[u['id']] = u_copy
                
                try:
                    graph_data = graph_builder.build_graph(vehicles_info, rsu_info, uav_info)
                    graphs.append(graph_data)
                    
                    # 生成对应的状态、动作和奖励（这里使用模拟数据）
                    state = np.random.randn(len(snapshot['vehicles']) * 4 + 4)  # 模拟状态
                    action = np.random.randn(len(snapshot['vehicles']))  # 模拟动作
                    reward = np.random.uniform(-1, 1)  # 模拟奖励
                    
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    
                except Exception as e:
                    print(f"Error creating graph data: {e}")
                    continue
        
        dataset = GraphDataset(
            graphs=graphs,
            states=states,
            actions=actions,
            rewards=rewards,
            metadata={
                'num_samples': len(graphs),
                'sequence_length': sequence_length,
                'timestamp_range': (graph_snapshots[0]['timestamp'], graph_snapshots[-1]['timestamp']),
                'num_vehicles': len(graph_snapshots[0]['vehicles']) if graph_snapshots else 0
            }
        )
        
        print(f"Created training dataset with {len(graphs)} samples")
        return dataset
    
    def save_dataset(self, dataset: GraphDataset, file_path: str):
        """
        保存数据集到文件
        
        Args:
            dataset: 要保存的数据集
            file_path: 保存路径
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {file_path}")
    
    def load_dataset(self, file_path: str) -> GraphDataset:
        """
        从文件加载数据集
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            加载的数据集
        """
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"Dataset loaded from {file_path}")
        return dataset
    
    def get_data_statistics(self, dataset: GraphDataset) -> Dict:
        """
        获取数据集统计信息
        
        Args:
            dataset: 数据集
            
        Returns:
            统计信息字典
        """
        if not dataset.graphs:
            return {}
        
        node_counts = [g.x.shape[0] for g in dataset.graphs]
        edge_counts = [g.edge_index.shape[1] for g in dataset.graphs]
        
        stats = {
            'num_samples': len(dataset.graphs),
            'avg_nodes': np.mean(node_counts),
            'avg_edges': np.mean(edge_counts),
            'node_feature_dim': dataset.graphs[0].x.shape[1],
            'edge_feature_dim': dataset.graphs[0].edge_attr.shape[1] if dataset.graphs[0].edge_attr.numel() > 0 else 0,
            'state_dim': len(dataset.states[0]) if dataset.states else 0,
            'action_dim': len(dataset.actions[0]) if dataset.actions else 0,
            'reward_range': (min(dataset.rewards), max(dataset.rewards)) if dataset.rewards else (0, 0)
        }
        
        return stats


def test_ngsim_processor():
    """测试NGSIM数据处理器"""
    print("Testing NGSIM Data Processor...")
    
    # 创建处理器
    processor = NGSIMDataProcessor()
    
    # 生成合成数据（因为没有真实NGSIM数据）
    df = processor._generate_synthetic_trajectory_data()
    print(f"Generated data shape: {df.shape}")
    
    # 处理轨迹
    trajectories = processor.process_trajectories(df)
    print(f"Number of trajectories: {len(trajectories)}")
    
    # 生成图快照
    snapshots = processor.generate_graph_snapshots(time_step=1.0)
    print(f"Number of snapshots: {len(snapshots)}")
    
    # 创建训练数据集
    dataset = processor.create_training_dataset(snapshots[:100], sequence_length=10)  # 使用前100个快照测试
    print(f"Dataset samples: {dataset.metadata['num_samples']}")
    
    # 获取统计信息
    stats = processor.get_data_statistics(dataset)
    print(f"Dataset statistics: {stats}")
    
    print("NGSIM Data Processor test completed!")


if __name__ == "__main__":
    test_ngsim_processor()