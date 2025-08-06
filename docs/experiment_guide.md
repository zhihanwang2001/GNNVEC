# GNN-PPO完整对比实验指南

## 🎯 实验目标

证明图神经网络增强的PPO算法在车联网边缘计算任务调度中的优越性，为论文提供强有力的实验支撑。

## 📊 完整实验矩阵

### 对比算法列表

| 算法名称 | 描述 | 配置文件 | 预期优势 |
|---------|------|----------|----------|
| **PPO基线** | 传统PPO，无图结构信息 | `configs/ppo_baseline.yaml` | 计算简单、稳定 |
| **GNN-PPO (GAT)** | 使用图注意力网络 | `configs/default.yaml` | 空间相关性、注意力权重 |
| **GNN-PPO (GCN)** | 使用图卷积网络 | `configs/gnn_gcn.yaml` | 图结构建模、计算效率 |

### 核心创新验证点

1. **图建模有效性**: GNN-PPO vs PPO基线
2. **GNN架构对比**: GAT vs GCN
3. **收敛速度**: 图结构是否加速学习
4. **最终性能**: 延迟、成功率、奖励
5. **稳定性**: 训练过程方差分析

## 🧪 详细实验步骤

### Phase 1: 基础对比实验

#### 1.1 训练PPO基线
```bash
# 500个episodes完整训练
python main.py --mode train_baseline
```
**预期时间**: ~2-3小时
**输出文件**:
- `saved_models/PPO_baseline_final.pth`
- `experiment_plot_new/PPO_baseline_15_5.log`

#### 1.2 训练GNN-PPO (GAT)
```bash
# 已完成，确认文件存在
ls experiment_plot_new/GNN_PPO_GAT_15_5.log
```

#### 1.3 训练GNN-PPO (GCN)
```bash
python main.py --mode train --config configs/gnn_gcn.yaml
```
**预期时间**: ~2-3小时
**输出文件**:
- `saved_models/GNN_PPO_GCN_model.pth`
- `experiment_plot_new/GNN_PPO_GCN_15_5.log`

### Phase 2: 高级对比实验

#### 2.1 不同随机种子验证
```bash
# 修改configs中的random_seed为不同值(如1234, 5678, 9999)
# 重新训练各算法，验证结果稳定性
```

#### 2.2 不同环境规模测试
```bash
# 修改environment配置:
# - 小规模: num_car=10, num_tcar=8, num_scar=2
# - 大规模: num_car=30, num_tcar=20, num_scar=10
```

### Phase 3: 结果分析和可视化

#### 3.1 生成对比图表
```bash
python experiments/visualize.py \
  --baseline_log experiment_plot_new/PPO_baseline_15_5.log \
  --gnn_log experiment_plot_new/GNN_PPO_GAT_15_5.log \
  --output_dir plots/gat_vs_baseline/

python experiments/visualize.py \
  --baseline_log experiment_plot_new/PPO_baseline_15_5.log \
  --gnn_log experiment_plot_new/GNN_PPO_GCN_15_5.log \
  --output_dir plots/gcn_vs_baseline/
```

#### 3.2 三算法综合对比
```bash
# 需要实现三算法对比的可视化脚本
python experiments/visualize_multi.py \
  --logs PPO_baseline:experiment_plot_new/PPO_baseline_15_5.log \
         GNN_GAT:experiment_plot_new/GNN_PPO_GAT_15_5.log \
         GNN_GCN:experiment_plot_new/GNN_PPO_GCN_15_5.log \
  --output_dir plots/comprehensive/
```

#### 3.3 统计显著性检验
```bash
python experiments/evaluate.py --comprehensive
```

## 📈 预期实验结果

### 定量指标对比

| 指标 | PPO基线 | GNN-GAT | GNN-GCN | GAT改进 | GCN改进 |
|------|---------|---------|---------|---------|---------|
| 最终奖励 | X.XX | Y.YY | Z.ZZ | +A.A% | +B.B% |
| 平均延迟 | XX.X | YY.Y | ZZ.Z | -A.A% | -B.B% |
| 成功率 | 0.XXX | 0.YYY | 0.ZZZ | +A.A% | +B.B% |
| 收敛速度(episodes) | ~XXX | ~YYY | ~ZZZ | -AA episodes | -BB episodes |

### 定性分析要点

1. **图建模优势**:
   - GNN算法应在空间相关性强的场景表现更好
   - 延迟指标改善最明显（空间优化效果）

2. **GAT vs GCN**:
   - GAT理论上在注意力权重学习上更强
   - GCN计算效率更高，可能收敛更稳定

3. **收敛特性**:
   - GNN算法初期可能波动更大（图特征学习）
   - 后期收敛更稳定（结构化信息约束）

## 📊 实验数据收集清单

### 训练过程数据
- [ ] 每个算法500个episodes的完整训练日志
- [ ] 奖励、延迟、成功率的逐episode记录
- [ ] 训练时间和计算资源使用情况

### 性能评估数据
- [ ] 最后100个episodes的统计均值和方差
- [ ] 最佳性能记录（峰值指标）
- [ ] 收敛稳定性指标

### 可视化材料
- [ ] 训练曲线对比图（4子图布局）
- [ ] 最终性能柱状图（3指标对比）
- [ ] 收敛分析图（稳定性、学习率等）
- [ ] 综合对比雷达图

### 统计分析结果
- [ ] t检验p值和置信区间
- [ ] 效应量计算（Cohen's d）
- [ ] 方差分析结果
- [ ] 非参数检验结果（Mann-Whitney U）

## 🎯 论文写作要点

### 实验部分结构

1. **Experimental Setup**
   - 环境配置和参数设置
   - 对比算法选择理由
   - 评估指标定义

2. **Baseline Comparison**
   - PPO vs GNN-PPO核心对比
   - 统计显著性分析
   - 改进幅度量化

3. **Ablation Study**
   - GAT vs GCN架构对比
   - 图特征重要性分析
   - 超参数敏感性分析

4. **Scalability Analysis**
   - 不同网络规模测试
   - 计算复杂度对比
   - 内存使用分析

### 关键图表标准

1. **Figure 1**: Training Convergence Comparison
   - 4子图：奖励、延迟、成功率、综合得分
   - 清晰的图例和标签
   - 置信区间或误差棒

2. **Figure 2**: Final Performance Comparison
   - 柱状图展示最终指标
   - 数值标注和改进百分比
   - 统计显著性标记

3. **Table 1**: Quantitative Results Summary
   - 完整数值对比表格
   - 均值±标准差格式
   - 统计检验结果

## 🚨 注意事项

1. **公平性保证**:
   - 所有算法使用相同随机种子
   - 相同的环境参数和网络架构
   - 统一的训练episodes数量

2. **重现性要求**:
   - 详细记录所有配置参数
   - 保存随机种子和环境状态
   - 版本控制所有代码修改

3. **统计严谨性**:
   - 多次运行验证结果稳定性
   - 正确的统计检验方法选择
   - 合理的置信水平设置

---

**实验完成后，你将拥有发表高质量论文所需的全部实验证据！**