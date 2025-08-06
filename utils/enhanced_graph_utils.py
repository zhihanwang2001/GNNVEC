"""
增强的图构建工具类
基于深入分析的改进版本，解决GNN-PPO性能不佳的关键问题
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from typing import Dict, List, Tuple, Optional
import math
import random


class EnhancedVehicleGraph:
    """
    增强的车辆网络图构建器
    解决原始图构建逻辑过于简化的问题
    """
    
    def __init__(self, 
                 communication_range: float = 300.0,
                 rsu_range: float = 500.0,
                 enable_realistic_channel: bool = True,
                 enable_load_balancing: bool = True):
        self.communication_range = communication_range
        self.rsu_range = rsu_range
        self.enable_realistic_channel = enable_realistic_channel
        self.enable_load_balancing = enable_load_balancing
        
        # 节点类型编码
        self.node_types = {
            'task_vehicle': 0,
            'service_vehicle': 1, 
            'rsu': 2,
            'uav': 3
        }
        
        # 边类型编码
        self.edge_types = {
            'v2v': 0,
            'v2i': 1, 
            'v2u': 2,
            'i2u': 3
        }
        
    def build_dynamic_graph(self, vehicles: List[Dict], infrastructure: List[Dict], 
                           env_context: Optional[Dict] = None) -> Data:
        """
        构建动态车辆网络图
        
        Args:
            vehicles: 车辆信息列表
            infrastructure: 基础设施信息列表 (RSU, UAV)
            env_context: 环境上下文信息
            
        Returns:
            Data: 图数据对象
        """
        nodes = vehicles + infrastructure
        n_nodes = len(nodes)
        
        if n_nodes == 0:
            return self._create_empty_graph()
        
        # 提取增强的节点特征
        node_features = []
        for i, node in enumerate(nodes):
            features = self._extract_enhanced_node_features(node, nodes, env_context)
            node_features.append(features)
        
        # 构建边和边特征
        edge_indices = []
        edge_features = []
        edge_types = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                node1, node2 = nodes[i], nodes[j]
                
                if self._should_connect_enhanced(node1, node2, env_context):
                    # 添加双向边
                    edge_indices.extend([(i, j), (j, i)])
                    
                    # 计算增强的边特征
                    edge_feat = self._calculate_enhanced_edge_features(node1, node2, env_context)
                    edge_features.extend([edge_feat, edge_feat])
                    
                    # 确定边类型
                    edge_type = self._determine_edge_type(node1, node2)
                    edge_types.extend([edge_type, edge_type])
        
        # 转换为tensor
        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 10), dtype=torch.float)
            edge_type = torch.empty((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
    
    def _extract_enhanced_node_features(self, node: Dict, all_nodes: List[Dict], 
                                      env_context: Optional[Dict]) -> List[float]:
        """
        提取增强的节点特征（20维特征向量）
        """
        pos = node.get('position', [0.0, 0.0])
        node_type = node.get('type', 'task_vehicle')
        
        # 基础特征 (7维)
        base_features = [
            pos[0] / 1000.0,  # 归一化x坐标
            pos[1] / 1000.0,  # 归一化y坐标
            node.get('velocity_x', 0.0) / 50.0,  # 归一化速度
            node.get('velocity_y', 0.0) / 50.0,
            node.get('compute_capacity', 1.0),
            len(node.get('task_queue', [])) / 10.0,  # 归一化任务队列
            self.node_types.get(node_type, 0)
        ]
        
        # 资源状态特征 (4维)
        resource_features = [
            node.get('cpu_usage', 0.5),  # CPU使用率
            node.get('memory_usage', 0.5),  # 内存使用率
            node.get('energy_level', 1.0),  # 能耗水平
            node.get('bandwidth_usage', 0.3)  # 带宽使用率
        ]
        
        # 网络拓扑特征 (4维)
        neighbors = self._count_neighbors(node, all_nodes)
        topo_features = [
            neighbors / 10.0,  # 邻居节点数
            node.get('link_quality_avg', 0.8),  # 平均链路质量
            node.get('connectivity_stability', 0.7),  # 连接稳定性
            node.get('centrality', 0.5)  # 网络中心性
        ]
        
        # 任务相关特征 (3维)
        task_features = [
            node.get('success_rate', 0.8),  # 任务成功率
            node.get('avg_processing_time', 50.0) / 100.0,  # 平均处理时间
            node.get('load_factor', 0.5)  # 负载因子
        ]
        
        # 历史统计特征 (2维)
        history_features = [
            node.get('position_variance', 0.1),  # 位置方差
            node.get('performance_trend', 0.0)  # 性能趋势
        ]
        
        return base_features + resource_features + topo_features + task_features + history_features
    
    def _should_connect_enhanced(self, node1: Dict, node2: Dict, 
                               env_context: Optional[Dict]) -> bool:
        """
        增强的连接判断逻辑
        """
        pos1 = np.array(node1.get('position', [0.0, 0.0]))
        pos2 = np.array(node2.get('position', [0.0, 0.0]))
        distance = np.linalg.norm(pos1 - pos2)
        
        type1 = node1.get('type', 'task_vehicle')
        type2 = node2.get('type', 'task_vehicle')
        
        # 基础距离检查
        if not self._distance_feasible(type1, type2, distance):
            return False
        
        # 信道可用性检查
        if self.enable_realistic_channel:
            if not self._channel_available(pos1, pos2, env_context):
                return False
        
        # 负载均衡检查
        if self.enable_load_balancing:
            if not self._load_balance_check(node1, node2):
                return False
        
        # QoS兼容性检查
        if not self._qos_compatible(node1, node2):
            return False
        
        return True
    
    def _distance_feasible(self, type1: str, type2: str, distance: float) -> bool:
        """距离可行性检查"""
        if 'vehicle' in type1 and 'vehicle' in type2:
            return distance <= self.communication_range
        elif 'rsu' in [type1, type2] or 'uav' in [type1, type2]:
            return distance <= self.rsu_range
        return False
    
    def _channel_available(self, pos1: np.ndarray, pos2: np.ndarray, 
                          env_context: Optional[Dict]) -> bool:
        """信道可用性检查"""
        if env_context is None:
            return True
        
        # 简化的信道模型：考虑干扰和阴影效应
        interference_level = env_context.get('interference_level', 0.1)
        shadowing_factor = env_context.get('shadowing_factor', 0.9)
        
        # 基于距离的路径损耗
        distance = np.linalg.norm(pos1 - pos2)
        path_loss_db = 32.45 + 20 * np.log10(distance/1000) + 20 * np.log10(2.4)  # 2.4GHz
        
        # 信干噪比计算
        signal_power = 20  # dBm
        noise_power = -90  # dBm
        sinr_db = signal_power - path_loss_db - 10*np.log10(1 + interference_level) - noise_power
        
        # 考虑阴影效应
        effective_sinr = sinr_db * shadowing_factor
        
        return effective_sinr > 10  # 10dB阈值
    
    def _load_balance_check(self, node1: Dict, node2: Dict) -> bool:
        """负载均衡检查"""
        load1 = node1.get('cpu_usage', 0.5)
        load2 = node2.get('cpu_usage', 0.5)
        
        # 避免高负载节点互连
        if load1 > 0.8 and load2 > 0.8:
            return False
        
        # 鼓励低负载和高负载节点连接
        load_diff = abs(load1 - load2)
        return load_diff > 0.1
    
    def _qos_compatible(self, node1: Dict, node2: Dict) -> bool:
        """QoS兼容性检查"""
        # 简化的QoS检查：基于节点类型
        type1 = node1.get('type', 'task_vehicle')
        type2 = node2.get('type', 'task_vehicle')
        
        # RSU和UAV提供高QoS服务
        if type1 in ['rsu', 'uav'] or type2 in ['rsu', 'uav']:
            return True
        
        # 服务车辆优先与任务车辆连接
        if type1 == 'service_vehicle' and type2 == 'task_vehicle':
            return True
        if type1 == 'task_vehicle' and type2 == 'service_vehicle':
            return True
        
        # 其他连接基于随机性
        return random.random() > 0.3
    
    def _calculate_enhanced_edge_features(self, node1: Dict, node2: Dict, 
                                        env_context: Optional[Dict]) -> List[float]:
        """
        计算增强的边特征（10维特征向量）
        """
        pos1 = np.array(node1.get('position', [0.0, 0.0]))
        pos2 = np.array(node2.get('position', [0.0, 0.0]))
        distance = np.linalg.norm(pos1 - pos2)
        
        # 物理层特征 (3维)
        path_loss_db = 32.45 + 20 * np.log10(max(distance/1000, 0.001)) + 20 * np.log10(2.4)
        sinr = max(0.0, min(1.0, (40 - path_loss_db) / 40))  # 归一化SINR
        channel_quality = max(0.1, 1.0 - distance / 1000.0)
        
        # 网络层特征 (3维)
        rtt = distance / 300000.0 * 2 * 1000  # RTT in ms
        bandwidth = self._estimate_bandwidth(node1, node2, distance)
        packet_loss = min(0.1, distance / 5000.0)  # 基于距离的丢包率
        
        # 应用层特征 (2维)
        load_factor = (node1.get('cpu_usage', 0.5) + node2.get('cpu_usage', 0.5)) / 2
        qos_score = self._calculate_qos_score(node1, node2)
        
        # 动态特征 (2维)
        relative_velocity = self._calculate_relative_velocity(node1, node2)
        link_stability = max(0.0, 1.0 - relative_velocity / 100.0)  # 基于相对速度的稳定性
        
        return [
            distance / 1000.0,      # 归一化距离
            sinr,                   # 信干噪比
            channel_quality,        # 信道质量
            rtt / 100.0,           # 归一化RTT
            bandwidth / 1e8,        # 归一化带宽
            packet_loss * 10,       # 丢包率
            load_factor,            # 负载因子
            qos_score,              # QoS评分
            relative_velocity / 100.0,  # 归一化相对速度
            link_stability          # 链路稳定性
        ]
    
    def _count_neighbors(self, node: Dict, all_nodes: List[Dict]) -> int:
        """计算邻居节点数"""
        pos = np.array(node.get('position', [0.0, 0.0]))
        neighbors = 0
        
        for other in all_nodes:
            if other is node:
                continue
            other_pos = np.array(other.get('position', [0.0, 0.0]))
            distance = np.linalg.norm(pos - other_pos)
            if distance <= self.communication_range:
                neighbors += 1
                
        return neighbors
    
    def _determine_edge_type(self, node1: Dict, node2: Dict) -> int:
        """确定边类型"""
        type1 = node1.get('type', 'task_vehicle')
        type2 = node2.get('type', 'task_vehicle')
        
        if 'vehicle' in type1 and 'vehicle' in type2:
            return self.edge_types['v2v']
        elif ('vehicle' in type1 and type2 == 'rsu') or (type1 == 'rsu' and 'vehicle' in type2):
            return self.edge_types['v2i']
        elif ('vehicle' in type1 and type2 == 'uav') or (type1 == 'uav' and 'vehicle' in type2):
            return self.edge_types['v2u']
        elif (type1 == 'rsu' and type2 == 'uav') or (type1 == 'uav' and type2 == 'rsu'):
            return self.edge_types['i2u']
        else:
            return 0  # 默认为v2v
    
    def _estimate_bandwidth(self, node1: Dict, node2: Dict, distance: float) -> float:
        """估算可用带宽"""
        base_bandwidth = 100e6  # 100 Mbps
        
        # 基于距离的衰减
        distance_factor = max(0.1, 1.0 - distance / 1000.0)
        
        # 基于节点类型的调整
        type1 = node1.get('type', 'task_vehicle')
        type2 = node2.get('type', 'task_vehicle')
        
        if 'rsu' in [type1, type2] or 'uav' in [type1, type2]:
            base_bandwidth *= 2  # 基础设施提供更高带宽
        
        # 基于负载的调整
        avg_load = (node1.get('cpu_usage', 0.5) + node2.get('cpu_usage', 0.5)) / 2
        load_factor = max(0.3, 1.0 - avg_load)
        
        return base_bandwidth * distance_factor * load_factor
    
    def _calculate_qos_score(self, node1: Dict, node2: Dict) -> float:
        """计算QoS评分"""
        type1 = node1.get('type', 'task_vehicle')
        type2 = node2.get('type', 'task_vehicle')
        
        # 基于节点类型的基础评分
        if 'rsu' in [type1, type2] or 'uav' in [type1, type2]:
            base_score = 0.9
        elif 'service_vehicle' in [type1, type2]:
            base_score = 0.7
        else:
            base_score = 0.5
        
        # 基于性能历史的调整
        perf1 = node1.get('success_rate', 0.8)
        perf2 = node2.get('success_rate', 0.8)
        performance_factor = (perf1 + perf2) / 2
        
        return base_score * performance_factor
    
    def _calculate_relative_velocity(self, node1: Dict, node2: Dict) -> float:
        """计算相对速度"""
        v1 = np.array([node1.get('velocity_x', 0.0), node1.get('velocity_y', 0.0)])
        v2 = np.array([node2.get('velocity_x', 0.0), node2.get('velocity_y', 0.0)])
        
        relative_v = v1 - v2
        return np.linalg.norm(relative_v)
    
    def _create_empty_graph(self) -> Data:
        """创建空图"""
        x = torch.zeros((1, 20), dtype=torch.float)  # 20维节点特征
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 10), dtype=torch.float)  # 10维边特征
        edge_type = torch.empty((0,), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)


class HeterogeneousGAT(nn.Module):
    """
    异构图注意力网络
    支持不同类型的边进行差异化处理
    """
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, 
                 output_dim: int = 256, num_edge_types: int = 4, 
                 num_layers: int = 3, num_heads: int = 8):
        super(HeterogeneousGAT, self).__init__()
        
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.output_dim = output_dim
        
        # 为每种边类型创建独立的GAT层
        self.edge_type_convs = nn.ModuleList()
        for _ in range(num_edge_types):
            layers = nn.ModuleList()
            for i in range(num_layers):
                in_dim = input_dim if i == 0 else hidden_dim * num_heads
                out_dim = hidden_dim
                concat = True if i < num_layers - 1 else False
                
                layers.append(GATConv(in_dim, out_dim, heads=num_heads, 
                                    concat=concat, edge_dim=10, dropout=0.1))
            self.edge_type_convs.append(layers)
        
        # 边类型权重学习
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types) / num_edge_types)
        
        # 层次化图池化
        self.hierarchical_pooling = HierarchicalGraphPooling(
            hidden_dim * num_heads if num_layers > 1 else hidden_dim, 
            output_dim
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, edge_type = data.x, data.edge_index, data.edge_attr, data.edge_type
        
        if edge_index.size(1) == 0:
            # 处理空图情况
            return torch.zeros(1, self.output_dim, device=x.device)
        
        # 为每种边类型分别处理
        type_outputs = []
        
        for edge_t in range(self.num_edge_types):
            mask = (edge_type == edge_t)
            if mask.sum() == 0:
                continue
                
            # 提取该类型的边
            edge_subset = edge_index[:, mask]
            edge_attr_subset = edge_attr[mask] if edge_attr is not None else None
            
            # 通过对应的GAT层
            h = x
            for layer in self.edge_type_convs[edge_t]:
                h = layer(h, edge_subset, edge_attr_subset)
                h = F.elu(h)
                h = F.dropout(h, training=self.training)
            
            type_outputs.append(h)
        
        if len(type_outputs) == 0:
            # 所有边类型都为空
            h = torch.zeros_like(x[:, :self.output_dim])
        else:
            # 加权融合不同边类型的输出
            weights = F.softmax(self.edge_type_weights[:len(type_outputs)], dim=0)
            h = sum(w * output for w, output in zip(weights, type_outputs))
        
        # 层次化图池化
        graph_embedding = self.hierarchical_pooling(h)
        
        return graph_embedding


class HierarchicalGraphPooling(nn.Module):
    """
    层次化图池化模块
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super(HierarchicalGraphPooling, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 多级池化策略
        self.attention_pooling = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
        self.set2set_pooling = nn.LSTM(input_dim, input_dim // 2, batch_first=True, bidirectional=True)
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 2),  # mean + max + attention
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, batch=None) -> torch.Tensor:
        if x.size(0) == 0:
            return torch.zeros(1, self.output_dim, device=x.device)
        
        # 平均池化
        mean_pool = torch.mean(x, dim=0)
        
        # 最大池化
        max_pool = torch.max(x, dim=0)[0]
        
        # 注意力池化
        x_expanded = x.unsqueeze(0)  # (1, N, D)
        attn_output, _ = self.attention_pooling(x_expanded, x_expanded, x_expanded)
        attention_pool = torch.mean(attn_output, dim=1).squeeze(0)  # (D,)
        
        # 融合所有池化结果
        combined = torch.cat([mean_pool, max_pool, attention_pool], dim=0)
        graph_embedding = self.fusion_network(combined)
        
        return graph_embedding.unsqueeze(0)  # (1, output_dim)


class EnhancedGraphFeatureExtractor:
    """
    增强的图特征提取器
    """
    
    def __init__(self):
        self.graph_builder = EnhancedVehicleGraph()
        
    def extract_graph_from_env(self, env_state: np.ndarray, env_info: Dict) -> Data:
        """
        从环境中提取增强的图结构
        """
        # 解析环境信息
        vehicles = []
        infrastructure = []
        
        # 从env_info中提取车辆信息
        if 'vehicles' in env_info:
            vehicles = env_info['vehicles']
        else:
            # 如果没有详细信息，基于env_state创建
            vehicles = self._create_vehicles_from_state(env_state, env_info)
            
        # 提取基础设施信息
        if 'infrastructure' in env_info:
            infrastructure = env_info['infrastructure']
        else:
            infrastructure = self._create_infrastructure_from_env(env_info)
        
        # 构建环境上下文
        env_context = {
            'interference_level': np.random.uniform(0.05, 0.15),
            'shadowing_factor': np.random.uniform(0.8, 0.95),
            'current_time': env_info.get('current_time', 0),
            'weather_factor': env_info.get('weather_factor', 1.0)
        }
        
        return self.graph_builder.build_dynamic_graph(vehicles, infrastructure, env_context)
    
    def _create_vehicles_from_state(self, env_state: np.ndarray, env_info: Dict) -> List[Dict]:
        """基于环境状态创建车辆信息"""
        vehicles = []
        
        num_cars = env_info.get('num_car', 20)
        num_tcar = env_info.get('num_tcar', 15)
        num_scar = env_info.get('num_scar', 5)
        
        # 创建任务车辆
        for i in range(num_tcar):
            vehicle = {
                'id': f'tcar_{i}',
                'type': 'task_vehicle',
                'position': [
                    np.random.uniform(0, 400),
                    np.random.uniform(0, 400)
                ],
                'velocity_x': np.random.uniform(-20, 20),
                'velocity_y': np.random.uniform(-20, 20),
                'compute_capacity': np.random.uniform(0.5, 1.0),
                'task_queue': ['task'] * np.random.randint(0, 8),
                'cpu_usage': np.random.uniform(0.3, 0.9),
                'memory_usage': np.random.uniform(0.2, 0.8),
                'energy_level': np.random.uniform(0.5, 1.0),
                'bandwidth_usage': np.random.uniform(0.1, 0.6),
                'success_rate': np.random.uniform(0.7, 0.95),
                'link_quality_avg': np.random.uniform(0.6, 0.9),
                'connectivity_stability': np.random.uniform(0.5, 0.8)
            }
            vehicles.append(vehicle)
        
        # 创建服务车辆
        for i in range(num_scar):
            vehicle = {
                'id': f'scar_{i}',
                'type': 'service_vehicle',
                'position': [
                    np.random.uniform(0, 400),
                    np.random.uniform(0, 400)
                ],
                'velocity_x': np.random.uniform(-15, 15),
                'velocity_y': np.random.uniform(-15, 15),
                'compute_capacity': np.random.uniform(1.5, 2.5),
                'task_queue': ['task'] * np.random.randint(0, 5),
                'cpu_usage': np.random.uniform(0.1, 0.6),
                'memory_usage': np.random.uniform(0.1, 0.5),
                'energy_level': np.random.uniform(0.7, 1.0),
                'bandwidth_usage': np.random.uniform(0.2, 0.5),
                'success_rate': np.random.uniform(0.8, 0.98),
                'link_quality_avg': np.random.uniform(0.7, 0.95),
                'connectivity_stability': np.random.uniform(0.6, 0.9)
            }
            vehicles.append(vehicle)
            
        return vehicles
    
    def _create_infrastructure_from_env(self, env_info: Dict) -> List[Dict]:
        """创建基础设施信息"""
        infrastructure = []
        
        num_rsu = env_info.get('num_rsu', 1)
        num_uav = env_info.get('num_uav', 1)
        
        # 创建RSU
        for i in range(num_rsu):
            rsu = {
                'id': f'rsu_{i}',
                'type': 'rsu',
                'position': [200, 200],  # RSU通常位于中心位置
                'velocity_x': 0.0,
                'velocity_y': 0.0,
                'compute_capacity': 5.0,  # RSU有强大的计算能力
                'task_queue': ['task'] * np.random.randint(0, 10),
                'cpu_usage': np.random.uniform(0.2, 0.7),
                'memory_usage': np.random.uniform(0.1, 0.4),
                'energy_level': 1.0,  # RSU有稳定的电源
                'bandwidth_usage': np.random.uniform(0.3, 0.7),
                'success_rate': 0.95,
                'link_quality_avg': 0.9,
                'connectivity_stability': 0.95
            }
            infrastructure.append(rsu)
        
        # 创建UAV
        for i in range(num_uav):
            uav = {
                'id': f'uav_{i}',
                'type': 'uav',
                'position': [
                    np.random.uniform(100, 300),
                    np.random.uniform(100, 300)
                ],
                'velocity_x': np.random.uniform(-10, 10),
                'velocity_y': np.random.uniform(-10, 10),
                'compute_capacity': 3.0,  # UAV有中等计算能力
                'task_queue': ['task'] * np.random.randint(0, 6),
                'cpu_usage': np.random.uniform(0.2, 0.6),
                'memory_usage': np.random.uniform(0.1, 0.5),
                'energy_level': np.random.uniform(0.6, 0.9),
                'bandwidth_usage': np.random.uniform(0.2, 0.6),
                'success_rate': 0.9,
                'link_quality_avg': 0.85,
                'connectivity_stability': 0.8
            }
            infrastructure.append(uav)
            
        return infrastructure