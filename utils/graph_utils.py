import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, GraphSAGE, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import math
from typing import List, Tuple, Dict, Optional

class DynamicVehicleGraph:
    """
    动态车辆网络图管理类
    用于构建和维护VEC系统中的车辆网络拓扑
    """
    
    def __init__(self, communication_range: float = 300.0):
        self.communication_range = communication_range
        self.node_types = {
            'task_vehicle': 0,
            'service_vehicle': 1, 
            'rsu': 2,
            'uav': 3
        }
        
    def build_graph(self, vehicles_info: Dict, rsu_info: Dict, uav_info: Dict) -> Data:
        """
        构建动态车辆网络图
        
        Args:
            vehicles_info: 车辆信息字典 {id: {pos, compute_capacity, task_queue, type}}
            rsu_info: RSU信息字典 {id: {pos, compute_capacity, task_queue}}
            uav_info: UAV信息字典 {id: {pos, compute_capacity, task_queue}}
        
        Returns:
            torch_geometric.data.Data: 图数据对象
        """
        nodes = []
        node_features = []
        positions = []
        node_types = []
        
        # 添加车辆节点
        for vehicle_id, info in vehicles_info.items():
            nodes.append(f"vehicle_{vehicle_id}")
            positions.append(info['pos'])
            
            # 节点特征: [pos_x, pos_y, compute_capacity, task_queue_len, node_type]
            features = [
                info['pos'][0], info['pos'][1],
                info['compute_capacity'],
                len(info['task_queue']),
                self.node_types[info['type']]
            ]
            node_features.append(features)
            node_types.append(info['type'])
        
        # 添加RSU节点
        for rsu_id, info in rsu_info.items():
            nodes.append(f"rsu_{rsu_id}")
            positions.append(info['pos'])
            
            features = [
                info['pos'][0], info['pos'][1],
                info['compute_capacity'],
                len(info['task_queue']),
                self.node_types['rsu']
            ]
            node_features.append(features)
            node_types.append('rsu')
            
        # 添加UAV节点
        for uav_id, info in uav_info.items():
            nodes.append(f"uav_{uav_id}")
            positions.append(info['pos'])
            
            features = [
                info['pos'][0], info['pos'][1],
                info['compute_capacity'], 
                len(info['task_queue']),
                self.node_types['uav']
            ]
            node_features.append(features)
            node_types.append('uav')
        
        # 构建邻接关系和边特征
        edge_indices = []
        edge_features = []
        
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i != j:
                    distance = self._calculate_distance(pos_i, pos_j)
                    
                    # 根据不同节点类型判断是否连接
                    if self._should_connect(node_types[i], node_types[j], distance):
                        edge_indices.append([i, j])
                        
                        # 边特征: [distance, channel_quality, bandwidth, link_lifetime]
                        edge_feature = self._calculate_edge_features(
                            pos_i, pos_j, node_types[i], node_types[j], distance
                        )
                        edge_features.append(edge_feature)
        
        # 转换为torch tensor
        x = torch.FloatTensor(node_features)
        edge_index = torch.LongTensor(edge_indices).t().contiguous() if edge_indices else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.FloatTensor(edge_features) if edge_features else torch.zeros((0, 4))
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算两点之间的欧几里得距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _should_connect(self, type1: str, type2: str, distance: float) -> bool:
        """判断两个节点是否应该连接"""
        # V2V连接
        if type1 in ['task_vehicle', 'service_vehicle'] and type2 in ['task_vehicle', 'service_vehicle']:
            return distance <= self.communication_range
        
        # V2I连接 (车辆到RSU)
        if (type1 in ['task_vehicle', 'service_vehicle'] and type2 == 'rsu') or \
           (type2 in ['task_vehicle', 'service_vehicle'] and type1 == 'rsu'):
            return distance <= 400  # RSU覆盖范围
        
        # RSU到UAV连接 (有线或高质量无线连接)
        if (type1 == 'rsu' and type2 == 'uav') or (type2 == 'rsu' and type1 == 'uav'):
            return True  # 假设RSU和UAV总是连接的
        
        return False
    
    def _calculate_edge_features(self, pos1: List[float], pos2: List[float], 
                               type1: str, type2: str, distance: float) -> List[float]:
        """计算边特征"""
        # 信道质量 (距离越近质量越好)
        channel_quality = max(0.1, 1.0 - distance / 500.0)
        
        # 带宽 (根据连接类型确定)
        if type1 == 'rsu' and type2 == 'uav' or type2 == 'rsu' and type1 == 'uav':
            bandwidth = 2e8  # RSU-UAV高带宽
        elif type1 == 'rsu' or type2 == 'rsu':
            bandwidth = 5e7  # V2I带宽
        else:
            bandwidth = 2e7  # V2V带宽
        
        # 链路生存时间 (简化计算，实际应考虑移动性)
        if type1 in ['task_vehicle', 'service_vehicle'] and type2 in ['task_vehicle', 'service_vehicle']:
            # V2V链路，考虑相对移动
            link_lifetime = max(1.0, (self.communication_range - distance) / 20.0)
        else:
            # 固定基础设施，链路相对稳定
            link_lifetime = 10.0
        
        return [distance, channel_quality, bandwidth, link_lifetime]


class GraphNeuralNetwork(nn.Module):
    """
    图神经网络编码器
    用于将车辆网络图编码为固定维度的特征向量
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 128, 
                 gnn_type: str = 'GAT', num_layers: int = 2):
        super(GraphNeuralNetwork, self).__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        if gnn_type == 'GAT':
            self.convs = nn.ModuleList([
                GATConv(input_dim if i == 0 else hidden_dim * 4, 
                       hidden_dim, heads=4, concat=True, edge_dim=4)
                for i in range(num_layers)
            ])
            final_hidden_dim = hidden_dim * 4  # GAT with 4 heads
        elif gnn_type == 'GCN':
            self.convs = nn.ModuleList([
                GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
            final_hidden_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # 改进的图级别特征提取 - 适应多池化策略
        # 输入维度 = final_hidden_dim * 3 (mean + max + attention)
        mlp_input_dim = final_hidden_dim * 3
        self.graph_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: torch_geometric.data.Data对象
        
        Returns:
            torch.Tensor: 图级别的嵌入向量 [batch_size, output_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 多层GNN编码
        for i, conv in enumerate(self.convs):
            if self.gnn_type == 'GAT':
                x = conv(x, edge_index, edge_attr)
            else:  # GCN
                x = conv(x, edge_index)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # 改进的图级别池化 - 结合多种池化策略
        if x.size(0) == 0:
            # 处理空图情况
            graph_embedding = torch.zeros(self.output_dim, device=x.device)
        else:
            # 多种池化策略的组合
            mean_pool = torch.mean(x, dim=0)  # 平均池化
            max_pool = torch.max(x, dim=0)[0]  # 最大池化
            
            # 如果节点数量足够，加入注意力池化权重
            if x.size(0) > 1:
                attention_weights = F.softmax(
                    torch.sum(x * mean_pool.unsqueeze(0), dim=1), dim=0
                )
                attention_pool = torch.sum(x * attention_weights.unsqueeze(1), dim=0)
                
                # 融合三种池化结果
                combined_embedding = torch.cat([mean_pool, max_pool, attention_pool], dim=0)
            else:
                # 单节点情况
                combined_embedding = torch.cat([mean_pool, max_pool, mean_pool], dim=0)
            
            # 通过MLP生成最终的图嵌入
            graph_embedding = self.graph_mlp(combined_embedding)
        
        return graph_embedding.unsqueeze(0)  # 添加batch维度


class GraphFeatureExtractor:
    """
    图特征提取器
    从VEC环境状态中提取图结构信息
    """
    
    def __init__(self):
        self.graph_builder = DynamicVehicleGraph()
        
    def extract_graph_from_env(self, env_state: np.ndarray, env_info: Dict) -> Data:
        """
        从VEC环境状态中提取图结构
        
        Args:
            env_state: 环境状态向量
            env_info: 环境额外信息 (车辆位置、任务队列等)
        
        Returns:
            Data: 图数据对象
        """
        # 解析环境信息构建图数据结构
        vehicles_info = {}
        rsu_info = {}
        uav_info = {}
        
        # 从env_info中提取信息构建图
        # 这里需要根据实际的env实现来调整
        for i, vehicle in enumerate(env_info.get('vehicles', [])):
            vehicles_info[i] = {
                'pos': vehicle['position'],
                'compute_capacity': vehicle['compute_capacity'],
                'task_queue': vehicle.get('task_queue', []),
                'type': vehicle['type']  # 'task_vehicle' or 'service_vehicle'
            }
        
        for i, rsu in enumerate(env_info.get('rsus', [])):
            rsu_info[i] = {
                'pos': rsu['position'],
                'compute_capacity': rsu['compute_capacity'],
                'task_queue': rsu.get('task_queue', [])
            }
            
        for i, uav in enumerate(env_info.get('uavs', [])):
            uav_info[i] = {
                'pos': uav['position'],
                'compute_capacity': uav['compute_capacity'],
                'task_queue': uav.get('task_queue', [])
            }
        
        return self.graph_builder.build_graph(vehicles_info, rsu_info, uav_info)


def test_graph_construction():
    """测试图构建功能"""
    print("Testing graph construction...")
    
    # 模拟车辆信息
    vehicles_info = {
        0: {'pos': [100, 100], 'compute_capacity': 1e8, 'task_queue': [1, 2], 'type': 'task_vehicle'},
        1: {'pos': [150, 120], 'compute_capacity': 2e8, 'task_queue': [], 'type': 'service_vehicle'},
        2: {'pos': [200, 150], 'compute_capacity': 1.5e8, 'task_queue': [3], 'type': 'task_vehicle'}
    }
    
    rsu_info = {
        0: {'pos': [175, 125], 'compute_capacity': 5e8, 'task_queue': [4, 5, 6]}
    }
    
    uav_info = {
        0: {'pos': [175, 200], 'compute_capacity': 3e8, 'task_queue': [7]}
    }
    
    # 构建图
    graph_builder = DynamicVehicleGraph()
    graph_data = graph_builder.build_graph(vehicles_info, rsu_info, uav_info)
    
    print(f"Graph nodes: {graph_data.x.shape}")
    print(f"Graph edges: {graph_data.edge_index.shape}")
    print(f"Edge features: {graph_data.edge_attr.shape}")
    
    # 测试GNN编码
    gnn = GraphNeuralNetwork(input_dim=5, output_dim=128)
    graph_embedding = gnn(graph_data)
    print(f"Graph embedding: {graph_embedding.shape}")
    
    print("Graph construction test completed!")


if __name__ == "__main__":
    test_graph_construction()