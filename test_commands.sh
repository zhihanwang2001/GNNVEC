#!/bin/bash
# 测试所有主要命令模式是否能正常解析参数

echo "🧪 测试 GNN_PPO_VEC 命令行接口"
echo "================================"

echo -e "\n1. 测试 --help 命令"
python main.py --help

echo -e "\n2. 测试 test_components 模式"
python main.py --mode test_components

echo -e "\n3. 检查可视化模式参数解析（不执行）"
python main.py --mode visualize --output_dir test_plots --verbose

echo -e "\n4. 检查训练基线模式参数解析（不执行）"  
python main.py --mode train_baseline --verbose

echo -e "\n5. 检查对比模式参数解析（不执行）"
python main.py --mode compare --verbose

echo -e "\n✅ 命令行接口测试完成"