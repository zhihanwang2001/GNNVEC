"""
VEC环境适配工具
用于桥接原始VEC环境与GNN-PPO算法的数据需求
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional, Any

# 添加原始项目路径以导入vec_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from vec_env import ENV
except ImportError:
    print("Warning: vec_env not found. Using mock environment.")
    ENV = None


class VECEnvironmentAdapter:
    """
    VEC环境适配器
    将原始VEC环境适配为支持图神经网络的接口
    """
    
    def __init__(self, num_car: int = 20, num_tcar: int = 15, num_scar: int = 5, 
                 num_task: int = 15, num_uav: int = 1, num_rsu: int = 1):
        """
        初始化VEC环境适配器
        
        Args:
            num_car: 总车辆数
            num_tcar: 任务车辆数
            num_scar: 服务车辆数
            num_task: 任务数量
            num_uav: UAV数量
            num_rsu: RSU数量
        """
        self.num_car = num_car
        self.num_tcar = num_tcar
        self.num_scar = num_scar
        self.num_task = num_task
        self.num_uav = num_uav
        self.num_rsu = num_rsu
        
        # 初始化环境
        if ENV is not None:
            self.env = ENV(num_car, num_tcar, num_scar, num_task, num_uav, num_rsu)
        else:
            self.env = None
            print("Using mock environment for development")
        
        # 车辆位置范围（米）
        self.area_width = 400
        self.area_height = 400
        
        # 通信参数
        self.v2v_range = 300  # V2V通信范围
        self.v2i_range = 400  # V2I通信范围
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        重置环境并返回初始状态和环境信息
        
        Returns:
            state: 传统环境状态向量
            env_info: 用于图构建的环境信息字典
        """
        if self.env is not None:
            state = self.env.reset()
        else:
            # Mock状态用于开发测试
            state_dim = self.num_tcar * 4 + self.num_rsu * 2 + self.num_uav * 2
            state = np.random.randn(state_dim)
        
        # 生成环境信息用于图构建
        env_info = self._extract_environment_info()
        
        return state, env_info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作并返回下一状态、奖励、结束标志和环境信息
        
        Args:
            action: 动作向量
            
        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否结束
            env_info: 环境信息
        """
        if self.env is not None:
            next_state, reward, done = self.env.step(action)
        else:
            # Mock响应用于开发测试
            state_dim = self.num_tcar * 4 + self.num_rsu * 2 + self.num_uav * 2
            next_state = np.random.randn(state_dim)
            reward = float(np.random.randn())  # 确保reward是标量
            done = False
        
        # 提取环境信息
        env_info = self._extract_environment_info()
        
        return next_state, reward, done, env_info
    
    def _extract_environment_info(self) -> Dict:
        """
        从环境中提取图构建所需的信息
        
        Returns:
            环境信息字典
        """
        env_info = {
            'vehicles': [],
            'rsus': [],
            'uavs': []
        }
        
        if self.env is not None and hasattr(self.env, 'car'):
            # 从真实环境提取信息
            for i in range(self.num_car):
                car = self.env.car[i]
                vehicle_info = {
                    'id': i,
                    'position': getattr(car, 'position', [np.random.uniform(0, 400), np.random.uniform(0, 400)]),
                    'compute_capacity': getattr(car, 'fn', 1e8 if i < self.num_tcar else 2e8),
                    'task_queue': getattr(car, 'task_queue', []),
                    'type': 'task_vehicle' if i < self.num_tcar else 'service_vehicle',
                    'energy_level': getattr(car, 'energy', 1.0),
                    'velocity': getattr(car, 'velocity', [0, 0])
                }
                env_info['vehicles'].append(vehicle_info)
            
            # RSU信息
            for i in range(self.num_rsu):
                rsu_info = {
                    'id': i,
                    'position': [200, 200],  # RSU固定位置
                    'compute_capacity': 5e8,
                    'task_queue': [],
                    'coverage_range': self.v2i_range
                }
                env_info['rsus'].append(rsu_info)
            
            # UAV信息
            for i in range(self.num_uav):
                uav_info = {
                    'id': i,
                    'position': [200, 300],  # UAV位置
                    'compute_capacity': 3e8,
                    'task_queue': [],
                    'flight_height': 100
                }
                env_info['uavs'].append(uav_info)
        
        else:
            # 生成模拟环境信息用于开发测试
            env_info = self._generate_mock_environment_info()
        
        return env_info
    
    def _generate_mock_environment_info(self) -> Dict:
        """
        生成模拟的环境信息用于开发和测试
        
        Returns:
            模拟环境信息字典
        """
        env_info = {
            'vehicles': [],
            'rsus': [],
            'uavs': []
        }
        
        # 生成车辆信息
        for i in range(self.num_car):
            # 随机位置但保持一定的聚集性
            cluster_centers = [[100, 100], [300, 100], [200, 300]]
            cluster_center = cluster_centers[np.random.randint(0, len(cluster_centers))]
            position = [
                cluster_center[0] + np.random.normal(0, 50),
                cluster_center[1] + np.random.normal(0, 50)
            ]
            position[0] = np.clip(position[0], 0, self.area_width)
            position[1] = np.clip(position[1], 0, self.area_height)
            
            vehicle_info = {
                'id': i,
                'position': position,
                'compute_capacity': np.random.uniform(0.5e8, 1e8) if i < self.num_tcar else np.random.uniform(1e8, 2e8),
                'task_queue': list(np.random.randint(0, 10, np.random.randint(0, 4))),
                'type': 'task_vehicle' if i < self.num_tcar else 'service_vehicle',
                'energy_level': np.random.uniform(0.3, 1.0),
                'velocity': [np.random.uniform(-20, 20), np.random.uniform(-20, 20)]
            }
            env_info['vehicles'].append(vehicle_info)
        
        # RSU信息
        rsu_positions = [[200, 200]] if self.num_rsu == 1 else [[150, 150], [250, 250]]
        for i in range(self.num_rsu):
            rsu_info = {
                'id': i,
                'position': rsu_positions[i] if i < len(rsu_positions) else [200, 200],
                'compute_capacity': np.random.uniform(3e8, 5e8),
                'task_queue': list(np.random.randint(0, 20, np.random.randint(0, 6))),
                'coverage_range': self.v2i_range
            }
            env_info['rsus'].append(rsu_info)
        
        # UAV信息
        uav_positions = [[200, 300]] if self.num_uav == 1 else [[150, 300], [250, 300]]
        for i in range(self.num_uav):
            uav_info = {
                'id': i,
                'position': uav_positions[i] if i < len(uav_positions) else [200, 300],
                'compute_capacity': np.random.uniform(2e8, 3e8),
                'task_queue': list(np.random.randint(0, 15, np.random.randint(0, 5))),
                'flight_height': np.random.uniform(80, 120)
            }
            env_info['uavs'].append(uav_info)
        
        return env_info
    
    def get_system_metrics(self) -> Dict[str, float]:
        """
        获取系统性能指标
        
        Returns:
            系统性能指标字典
        """
        if self.env is not None:
            metrics = {
                'total_delay': getattr(self.env, 'delay_sum', 0),
                'success_rate': 1 - (getattr(self.env, 'count_wrong', 0) / max(self.num_task * 20, 1)),
                'energy_consumption': getattr(self.env, 'energy_sum', 0),
                'throughput': getattr(self.env, 'throughput', 0)
            }
        else:
            # Mock指标
            metrics = {
                'total_delay': np.random.uniform(10, 100),
                'success_rate': np.random.uniform(0.7, 0.95),
                'energy_consumption': np.random.uniform(100, 1000),
                'throughput': np.random.uniform(50, 200)
            }
        
        return metrics
    
    def seed(self, random_seed: int):
        """设置随机种子"""
        np.random.seed(random_seed)
        if self.env is not None and hasattr(self.env, 'seed'):
            self.env.seed(random_seed)
    
    @property
    def state_space_shape(self) -> Tuple[int]:
        """获取状态空间维度"""
        if self.env is not None:
            return self.env.state_space.shape
        else:
            # Mock状态空间维度
            state_dim = self.num_tcar * 4 + self.num_rsu * 2 + self.num_uav * 2
            return (state_dim,)
    
    @property
    def action_space_shape(self) -> Tuple[int]:
        """获取动作空间维度"""
        if self.env is not None:
            return self.env.action_space.shape
        else:
            # Mock动作空间维度
            action_dim = self.num_task
            return (action_dim,)


def test_environment_adapter():
    """测试VEC环境适配器"""
    print("Testing VEC Environment Adapter...")
    
    # 创建适配器
    adapter = VECEnvironmentAdapter(num_car=10, num_tcar=8, num_scar=2)
    
    # 测试重置
    state, env_info = adapter.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Environment info keys: {env_info.keys()}")
    print(f"Number of vehicles: {len(env_info['vehicles'])}")
    print(f"Number of RSUs: {len(env_info['rsus'])}")
    print(f"Number of UAVs: {len(env_info['uavs'])}")
    
    # 测试步骤执行
    action = np.random.randn(adapter.action_space_shape[0])
    next_state, reward, done, env_info = adapter.step(action)
    print(f"Next state shape: {next_state.shape}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    
    # 测试系统指标
    metrics = adapter.get_system_metrics()
    print(f"System metrics: {metrics}")
    
    print("VEC Environment Adapter test completed!")


if __name__ == "__main__":
    test_environment_adapter()