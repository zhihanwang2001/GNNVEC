"""
独立的GNN网络模块
可复用的图卷积、注意力机制等组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv
from torch_geometric.data import Data
from typing import Optional, List


class GraphAttentionNetwork(nn.Module):
    """
    图注意力网络（GAT）
    使用多头注意力机制学习图结构特征
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super(GraphAttentionNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT层
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.layer_norms.append(nn.LayerNorm(output_dim))
            else:
                self.layer_norms.append(nn.LayerNorm(hidden_dim * heads))
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.layer_norms[i](x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GraphConvNetwork(nn.Module):
    """
    图卷积网络（GCN）
    标准的图卷积操作
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super(GraphConvNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(output_dim))
            else:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GraphSAGENetwork(nn.Module):
    """
    GraphSAGE网络
    适用于大规模图的归纳学习
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1, aggr: str = 'mean'):
        super(GraphSAGENetwork, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # SAGE层
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggr))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggr))
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class TemporalGraphNetwork(nn.Module):
    """
    时序图网络
    处理动态变化的图结构
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 sequence_length: int = 10, gnn_type: str = 'GAT'):
        super(TemporalGraphNetwork, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # 选择GNN类型
        if gnn_type == 'GAT':
            self.gnn = GraphAttentionNetwork(input_dim, hidden_dim, hidden_dim)
        elif gnn_type == 'GCN':
            self.gnn = GraphConvNetwork(input_dim, hidden_dim, hidden_dim)
        elif gnn_type == 'SAGE':
            self.gnn = GraphSAGENetwork(input_dim, hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # LSTM用于时序建模
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, graph_sequence: List[Data]) -> torch.Tensor:
        """
        处理图序列
        
        Args:
            graph_sequence: 图序列列表
            
        Returns:
            时序图特征
        """
        # 对序列中每个图进行GNN编码
        graph_embeddings = []
        for graph in graph_sequence:
            node_features = self.gnn(graph)  # [N, hidden_dim]
            # 全局平均池化得到图级别特征
            graph_embedding = torch.mean(node_features, dim=0)  # [hidden_dim]
            graph_embeddings.append(graph_embedding)
        
        # 堆叠为序列 [seq_len, hidden_dim]
        sequence_embeddings = torch.stack(graph_embeddings, dim=0)
        sequence_embeddings = sequence_embeddings.unsqueeze(0)  # [1, seq_len, hidden_dim]
        
        # LSTM处理时序信息
        lstm_out, _ = self.lstm(sequence_embeddings)
        
        # 使用最后一个时间步的输出
        final_embedding = lstm_out[0, -1, :]  # [hidden_dim]
        
        # 输出层
        output = self.output_layer(final_embedding)
        
        return output.unsqueeze(0)  # [1, output_dim]


class GraphTransformerNetwork(nn.Module):
    """
    图Transformer网络
    使用Transformer机制处理图数据
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super(GraphTransformerNetwork, self).__init__()
        
        self.num_layers = num_layers
        
        # Graph Transformer层
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(
                input_dim if _ == 0 else hidden_dim,
                hidden_dim,
                heads=heads,
                dropout=dropout
            ))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = self.output_layer(x)
        return x


class MultiScaleGraphNetwork(nn.Module):
    """
    多尺度图网络
    结合局部和全局图特征
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MultiScaleGraphNetwork, self).__init__()
        
        # 局部特征提取（1-hop邻居）
        self.local_conv = GCNConv(input_dim, hidden_dim)
        
        # 全局特征提取（2-hop邻居）
        self.global_conv1 = GCNConv(input_dim, hidden_dim)
        self.global_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 特征融合
        self.fusion_layer = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        # 局部特征
        local_features = F.relu(self.local_conv(x, edge_index))
        
        # 全局特征
        global_features = F.relu(self.global_conv1(x, edge_index))
        global_features = F.relu(self.global_conv2(global_features, edge_index))
        
        # 特征融合
        combined_features = torch.cat([local_features, global_features], dim=-1)
        output = self.fusion_layer(combined_features)
        
        return output


def test_gnn_modules():
    """测试GNN模块"""
    print("Testing GNN Modules...")
    
    # 创建测试图数据
    num_nodes = 10
    num_features = 5
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    data = Data(x=x, edge_index=edge_index)
    
    # 测试GAT
    print("Testing GraphAttentionNetwork...")
    gat = GraphAttentionNetwork(num_features, 16, 32)
    gat_out = gat(data)
    print(f"GAT output shape: {gat_out.shape}")
    
    # 测试GCN
    print("Testing GraphConvNetwork...")
    gcn = GraphConvNetwork(num_features, 16, 32)
    gcn_out = gcn(data)
    print(f"GCN output shape: {gcn_out.shape}")
    
    # 测试SAGE
    print("Testing GraphSAGENetwork...")
    sage = GraphSAGENetwork(num_features, 16, 32)
    sage_out = sage(data)
    print(f"SAGE output shape: {sage_out.shape}")
    
    # 测试Transformer
    print("Testing GraphTransformerNetwork...")
    transformer = GraphTransformerNetwork(num_features, 16, 32)
    transformer_out = transformer(data)
    print(f"Transformer output shape: {transformer_out.shape}")
    
    print("All GNN modules tested successfully!")


if __name__ == "__main__":
    test_gnn_modules()