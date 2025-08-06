# GNN_PPO_VEC: 图神经网络增强的车联网边缘计算任务调度

## 🚀 快速开始

### 环境安装
```bash
# 1. 进入项目目录
cd /Users/harry./Desktop/VCORA-code/GNN_PPO_VEC

# 2. 安装依赖
pip install torch torch_geometric numpy pandas matplotlib pyyaml tqdm

# 3. 验证安装 - 测试所有组件
python main.py --mode test_components
```

### 基本使用

#### 🎯 训练模型（推荐）
```bash
# 使用默认配置训练GNN-PPO模型
python main.py --mode train

# 训练过程会显示：
# Episode: 0, Reward: -X.XX, Delay: X.XX, Success Rate: 0.XXX
# 模型自动保存到 saved_models/ 目录
```

#### 📊 评估模型
```bash
# 评估训练好的模型
python main.py --mode test --model saved_models/GNN_PPO_model.pth
```

#### 🔧 自定义配置训练
```bash
# 修改 configs/default.yaml 配置文件，然后运行：
python main.py --mode train --config configs/default.yaml
```

#### 📈 处理NGSIM数据
```bash
# 处理车辆轨迹数据（可选，系统会自动生成合成数据）
python main.py --mode process_data --data_path data/ngsim/trajectory.csv --output_dir results/
```

## 🆚 完整对比实验指南

### 第一步：训练算法对比

#### 1. 训练GNN-PPO算法（已完成）
你已经成功训练了GNN-PPO，日志文件在：`experiment_plot_new/GNN_PPO_GAT_15_5.log`

#### 2. 训练PPO基线模型
```bash
# 训练传统PPO作为对比基线（500个episodes，完整训练）
python main.py --mode train_baseline
```
**输出**：
- 模型文件：`saved_models/PPO_baseline_final.pth`
- 日志文件：`experiment_plot_new/PPO_baseline_15_5.log`

#### 3. 自动化对比训练（推荐）
```bash
# 一键运行完整对比：PPO基线 + GNN-PPO
python main.py --mode compare
```
**说明**：会自动训练两个算法并保存所有结果，确保使用相同随机种子保证公平对比

### 第二步：结果分析和可视化

#### 1. 生成完整可视化图表
```bash
# 生成所有对比图表：训练曲线、性能对比、收敛分析
python main.py --mode visualize
```

**自动查找日志文件**：
- PPO基线：`experiment_plot_new/PPO_baseline_15_5.log`
- GNN-PPO：`experiment_plot_new/GNN_PPO_GAT_15_5.log`

**手动指定日志文件**：
```bash
python experiments/visualize.py --baseline_log experiment_plot_new/PPO_baseline_15_5.log --gnn_log experiment_plot_new/GNN_PPO_GAT_15_5.log --output_dir plots/
```

**生成的图表**：
- `plots/training_comparison.png` - 训练过程对比（奖励、延迟、成功率、收敛分析）
- `plots/performance_comparison.png` - 最终性能柱状图对比
- `plots/convergence_analysis.png` - 详细收敛性分析（方差、学习率、综合得分）
- `plots/experiment_summary.txt` - 文本格式实验报告

#### 2. 模型性能评估
```bash
# 加载训练好的模型进行深度评估（100个episodes测试）
python main.py --mode evaluate
```

**评估内容**：
- 统计显著性检验（t-test, Mann-Whitney U test）
- 效应量分析（Cohen's d）
- 稳定性分析（标准差对比）
- 生成`plots/evaluation_report.txt`

### 第三步：实验报告解读

#### 关键对比指标

1. **训练效果对比**：
   - 奖励收敛速度和最终值
   - 延迟降低幅度
   - 成功率提升程度

2. **算法稳定性**：
   - 训练过程方差分析
   - 奖励波动程度
   - 收敛稳定性

3. **统计显著性**：
   - p值 < 0.05 表示差异显著
   - Cohen's d > 0.5 表示中等效应量
   - Cohen's d > 0.8 表示大效应量

#### 预期结果模式

**GNN-PPO优势场景**：
- ✅ 更快收敛（利用图结构信息）
- ✅ 更低延迟（空间相关性优化）
- ✅ 更高成功率（全局协调决策）
- ✅ 更稳定训练（结构化特征）

**可能的权衡**：
- ❓ 计算开销更高（GNN前向传播）
- ❓ 内存使用更多（图数据存储）

### 第四步：论文写作数据

**完整实验后，你将获得**：

1. **数值对比表格**：
   ```
   指标            PPO基线    GNN-PPO    改进幅度
   最终平均奖励    X.XXX      Y.YYY      +Z.Z%
   最终平均延迟    XX.XX      YY.YY      -Z.Z%
   最终成功率      0.XXX      0.YYY      +Z.Z%
   ```

2. **高质量图表**：
   - 训练曲线对比图（4子图）
   - 性能柱状图（3指标）
   - 收敛分析图（4维度）

3. **统计证据**：
   - t检验结果和p值
   - 效应量计算
   - 置信区间

4. **技术亮点**：
   - 图神经网络建模创新
   - 空间相关性利用
   - 多层次特征融合

### 📋 完整实验检查清单

- [ ] **步骤1**：GNN-PPO训练完成 ✅（已完成）
- [ ] **步骤2**：PPO基线训练 (`python main.py --mode train_baseline`)
- [ ] **步骤3**：生成对比图表 (`python main.py --mode visualize`)
- [ ] **步骤4**：性能评估 (`python main.py --mode evaluate`)
- [ ] **步骤5**：检查`plots/`目录所有图表
- [ ] **步骤6**：阅读实验总结报告

### 💡 重要提示
- **首次使用**：直接运行 `python main.py --mode train` 即可开始训练
- **Mock环境**：系统会自动使用模拟环境，无需额外的VEC环境文件
- **GPU支持**：如果有CUDA GPU，程序会自动使用GPU加速训练
- **日志输出**：训练日志保存在 `experiment_plot_new/` 目录

---

## 📋 项目概述

这是对原始VEC论文的重大技术创新重构，从简单的PPO应用转向真正的算法创新。核心创新是使用图神经网络（GNN）建模车辆网络的动态拓扑结构，结合强化学习（RL）实现智能任务调度决策。

### 🎯 技术创新点
1. **动态图建模**: 实时构建和更新车辆-RSU-UAV网络拓扑图
2. **GNN特征学习**: 提取空间相关性和网络结构特征
3. **图增强RL**: 将图嵌入融入PPO决策过程
4. **理论分析**: GNN-PPO收敛性证明和复杂度分析

## 📁 项目结构

```
GNN_PPO_VEC/
├── README.md              # 项目主文档 (本文件)
├── requirements.txt       # Python依赖清单
├── main.py               # 主入口文件
├── utils/                # 工具模块
│   ├── __init__.py
│   ├── graph_utils.py    # 图构建和GNN编码工具
│   ├── env_utils.py      # 环境适配工具
│   └── data_utils.py     # 数据处理工具
├── models/               # 模型实现
│   ├── __init__.py
│   ├── gnn_ppo.py       # GNN增强的PPO算法
│   ├── gnn_modules.py   # GNN网络模块
│   └── ppo_base.py      # PPO基础实现
├── experiments/          # 实验脚本
│   ├── __init__.py
│   ├── train.py         # 训练脚本
│   ├── evaluate.py      # 评估脚本
│   └── visualize.py     # 结果可视化
├── configs/             # 配置文件
│   ├── default.yaml     # 默认配置
│   └── gnn_config.yaml  # GNN特定配置
├── docs/                # 详细文档
│   ├── algorithm.md     # 算法设计文档
│   ├── theory.md        # 理论分析文档
│   └── experiments.md   # 实验说明文档
└── saved_models/        # 模型保存目录
    └── checkpoints/
```

## 🔧 核心模块说明

### 1. utils/ - 工具模块

#### `graph_utils.py`
- **功能**: 动态车辆网络图构建和GNN编码
- **核心类**:
  - `DynamicVehicleGraph`: 车辆网络拓扑构建
  - `GraphNeuralNetwork`: GNN编码器（支持GAT/GCN）
  - `GraphFeatureExtractor`: 从环境状态提取图特征
- **用途**: 将VEC系统建模为动态图，提取空间相关性特征

#### `env_utils.py`
- **功能**: VEC环境适配和数据接口
- **用途**: 桥接原始环境与GNN模型的数据需求

#### `data_utils.py`
- **功能**: NGSIM数据处理和图数据生成
- **用途**: 处理车辆轨迹数据，构建训练和测试数据集

### 2. models/ - 模型实现

#### `gnn_ppo.py`
- **功能**: GNN增强的PPO主算法
- **核心类**:
  - `GNN_PPO`: 主算法类，整合图特征和RL决策
  - `GNNActor`: 基于图嵌入的策略网络
  - `GNNCritic`: 基于图嵌入的价值网络
- **创新**: 将图神经网络嵌入到PPO的Actor-Critic架构中

#### `gnn_modules.py`
- **功能**: 独立的GNN网络模块
- **用途**: 可复用的图卷积、注意力机制等组件

#### `ppo_base.py`
- **功能**: 传统PPO基础实现
- **用途**: 对比基准和算法验证

### 3. experiments/ - 实验脚本

#### `train.py`
- **功能**: GNN-PPO训练主脚本
- **特点**: 支持大规模车辆网络训练，可配置的超参数

#### `evaluate.py`
- **功能**: 模型性能评估
- **指标**: 任务完成率、平均时延、能耗效率等

#### `visualize.py`
- **功能**: 结果可视化和图网络可视化
- **输出**: 训练曲线、网络拓扑、性能对比图

### 4. configs/ - 配置管理

#### `default.yaml`
- **内容**: 基础训练参数、环境设置、模型超参数
- **用途**: 统一的参数管理，便于实验重现

#### `gnn_config.yaml`
- **内容**: GNN特定参数（网络类型、层数、注意力头数等）
- **用途**: GNN架构的灵活配置

### 5. docs/ - 详细文档

#### `algorithm.md`
- **内容**: GNN-PPO算法详细设计，伪代码，架构图
- **用途**: 算法理解和论文写作参考

#### `theory.md`
- **内容**: 数学推导、收敛性证明、复杂度分析
- **用途**: 理论分析和审稿人质疑回应

#### `experiments.md`
- **内容**: 实验设计、参数设置、结果分析
- **用途**: 实验重现和结果解释

## 🚀 快速开始

### 1. 环境设置
```bash
# 安装依赖
pip install -r requirements.txt

# 安装torch_geometric
pip install torch_geometric
```

### 2. 训练模型
```bash
# 使用默认配置训练
python main.py --mode train

# 使用自定义配置
python main.py --mode train --config configs/gnn_config.yaml
```

### 3. 评估模型
```bash
# 评估训练好的模型
python main.py --mode test --model saved_models/gnn_ppo_best.pth
```

## 📊 技术规格

### GNN架构
- **支持类型**: GAT, GCN, GraphSAGE
- **节点特征**: 位置、计算能力、任务队列、节点类型
- **边特征**: 距离、信道质量、带宽、链路生存时间
- **图级特征**: 全局平均池化 + MLP

### RL配置
- **算法**: PPO with Clipped Objective
- **网络**: Actor-Critic with Graph Fusion
- **状态空间**: 传统状态 + 图嵌入向量
- **动作空间**: 连续任务调度决策

### 实验设置
- **车辆规模**: 50-500辆
- **网络拓扑**: 动态变化，实时更新
- **训练回合**: 500轮，每轮20个时隙
- **对比算法**: 传统PPO, 最新GNN-VEC方法

## 🔄 开发状态

- [x] 项目结构搭建
- [x] 核心算法实现
- [ ] 依赖安装和测试
- [ ] 环境适配和数据接口
- [ ] 大规模实验验证
- [ ] 理论分析和证明
- [ ] 论文重写

## 📞 维护说明

本项目是VEC论文的重大重构版本，每次代码修改都会同步更新相关文档。请确保：
1. 代码修改后及时更新对应模块的说明
2. 新增功能需要在相应的docs/文件中记录
3. 实验结果需要记录在experiments.md中
4. 理论分析更新需要同步到theory.md

---

**版本**: v1.0  
**最后更新**: 2025-08-06  
**状态**: 开发中 (Week 1-2 实现阶段)