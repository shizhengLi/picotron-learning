# Picotron 功能扩展项目

## 项目概述

本项目是基于Picotron分布式训练框架的功能扩展，旨在将Picotron从教育性质的工具发展为生产级别的分布式训练系统。



## 项目优化


目前的测试代码举例：

```c
python -m pytest tests/unit/test_parallel/test_moe_basic.py -v --tb=short

```

## 主要功能

### 核心并行策略
- **MoE并行**: 支持混合专家模型的分布式训练
- **3D并行优化**: 优化张量、流水线、数据并行的协同工作
- **自适应混合精度**: 动态调整训练精度以提高效率

### 智能系统
- **智能内存管理**: 自动化的内存分配、整理和优化
- **高级训练策略**: 课程学习和元学习支持
- **分布式存储**: 高效的检查点和数据缓存系统

### 监控和工具
- **实时监控**: 全面的性能监控和异常检测
- **自动调优**: 基于机器学习的自动参数优化
- **开发工具**: 分布式调试器和性能分析器

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
import torch
from src.parallel import MoEParallelStrategy
from src.optimization import AdaptiveMixedPrecision

# 创建模型
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 10)
)

# 应用MoE并行
moe_strategy = MoEParallelStrategy(
    num_experts=8,
    expert_parallel_size=2
)
moe_strategy.setup_expert_parallel(model)

# 应用自适应混合精度
amp = AdaptiveMixedPrecision(model, {})
amp.setup_adaptive_precision()

# 训练模型
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    # 训练代码...
    pass
```

## 项目结构

```
project/
├── src/                    # 源代码
├── tests/                  # 测试代码
├── summary/                # 技术文档
├── configs/                # 配置文件
├── examples/               # 示例代码
└── requirements.txt        # 依赖文件
```

## 开发指南

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/

# 生成覆盖率报告
pytest --cov=src tests/
```

### 代码格式化

```bash
# 格式化代码
black src/ tests/

# 检查代码风格
flake8 src/ tests/

# 类型检查
mypy src/
```

## 文档

详细的技术文档和实现说明请参考 `summary/` 目录：

- [MoE并行实现](summary/01-moe并行实现.md)
- [3D并行优化实现](summary/02-3d并行优化实现.md)
- [自适应混合精度实现](summary/03-自适应混合精度实现.md)
- [智能内存管理实现](summary/04-智能内存管理实现.md)
- [基础监控系统实现](summary/05-基础监控系统实现.md)

## 贡献指南

1. Fork 本项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues
- Email: picotron@example.com