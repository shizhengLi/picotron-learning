"""
Picotron 功能扩展项目
======================

这是一个基于Picotron的分布式训练框架功能扩展项目。

主要功能模块：
- MoE并行策略
- 3D并行优化
- 自适应混合精度
- 智能内存管理
- 高级训练策略
- 分布式存储系统
- 监控和诊断系统
- 开发工具
"""

__version__ = "0.1.0"
__author__ = "Picotron Team"

from .parallel import *
from .optimization import *
from .training import *
from .storage import *
from .monitoring import *
from .tools import *