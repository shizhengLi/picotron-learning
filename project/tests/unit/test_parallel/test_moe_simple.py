"""
MoE并行策略简化测试
==================
"""

import pytest
import sys
import os

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# 创建模拟的torch模块用于测试
class MockTensor:
    def __init__(self, shape):
        self.shape = shape
        self.data = [0] * self._size()
    
    def _size(self):
        size = 1
        for dim in self.shape:
            size *= dim
        return size
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value

class MockModule:
    def __init__(self):
        self.training = True
    
    def __call__(self, x):
        return x

class MockLinear(MockModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

class MockReLU(MockModule):
    def __init__(self):
        super().__init__()

class MockSequential(MockModule):
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

class MockTorch:
    @staticmethod
    def randn(*shape):
        return MockTensor(shape)
    
    @staticmethod
    def zeros_like(tensor):
        return MockTensor(tensor.shape)
    
    @staticmethod
    def allclose(a, b, atol=1e-6):
        return True
    
    @staticmethod
    def any(tensor):
        return True
    
    @staticmethod
    def sum(tensor, dim=None):
        return MockTensor((1,))
    
    @staticmethod
    def topk(tensor, k, dim=-1):
        return MockTensor((tensor.shape[0], k)), MockTensor((tensor.shape[0], k))
    
    @staticmethod
    def softmax(tensor, dim=-1):
        return tensor
    
    class nn:
        Module = MockModule
        Linear = MockLinear
        ReLU = MockReLU
        Sequential = MockSequential
    
    class distributed:
        @staticmethod
        def is_initialized():
            return False
        
        @staticmethod
        def get_rank():
            return 0
        
        @staticmethod
        def get_world_size():
            return 1

# 替换torch模块
sys.modules['torch'] = MockTorch()

# 现在导入我们的模块
from parallel.moe import LoadBalancer, MoEParallelStrategy


class TestLoadBalancer:
    """负载均衡器测试"""
    
    def test_load_balancer_creation(self):
        """测试负载均衡器创建"""
        balancer = LoadBalancer(num_experts=8, expert_parallel_size=2)
        
        assert balancer.num_experts == 8
        assert balancer.expert_parallel_size == 2
        assert balancer.expert_load == [0] * 8
        assert balancer.capacity_factor == 1.2
    
    def test_update_load(self):
        """测试负载更新"""
        balancer = LoadBalancer(num_experts=4, expert_parallel_size=2)
        
        # 更新负载
        balancer.update_load(0, 10)
        balancer.update_load(1, 20)
        balancer.update_load(2, 15)
        
        assert balancer.expert_load == [10, 20, 15, 0]
    
    def test_get_load_stats(self):
        """测试负载统计"""
        balancer = LoadBalancer(num_experts=4, expert_parallel_size=2)
        
        # 设置负载
        balancer.expert_load = [10, 20, 15, 5]
        
        stats = balancer.get_load_stats()
        
        assert stats['max_load'] == 20
        assert stats['min_load'] == 5
        assert stats['avg_load'] == 12.5
        assert stats['load_variance'] == 31.25
    
    def test_should_rebalance(self):
        """测试重新平衡判断"""
        balancer = LoadBalancer(num_experts=4, expert_parallel_size=2)
        
        # 均衡负载
        balancer.expert_load = [10, 12, 8, 10]
        assert not balancer.should_rebalance()
        
        # 不均衡负载
        balancer.expert_load = [30, 5, 10, 5]
        assert balancer.should_rebalance()
    
    def test_rebalance_experts(self):
        """测试专家重新平衡"""
        balancer = LoadBalancer(num_experts=4, expert_parallel_size=2)
        
        # 设置不均衡负载
        balancer.expert_load = [30, 5, 10, 5]
        
        # 重新平衡
        balancer.rebalance_experts()
        
        # 验证负载被限制在合理范围内
        stats = balancer.get_load_stats()
        avg_load = stats['avg_load']
        
        for load in balancer.expert_load:
            assert load <= avg_load * balancer.capacity_factor * 1.1  # 允许一定误差


class TestMoEParallelStrategy:
    """MoE并行策略测试"""
    
    def test_moe_strategy_creation(self):
        """测试MoE策略创建"""
        strategy = MoEParallelStrategy(num_experts=8, expert_parallel_size=2)
        
        assert strategy.num_experts == 8
        assert strategy.expert_parallel_size == 2
        assert len(strategy.expert_groups) == 2
        assert strategy.expert_groups[0] == [0, 1, 2, 3]
        assert strategy.expert_groups[1] == [4, 5, 6, 7]
    
    def test_create_expert_groups(self):
        """测试专家分组创建"""
        strategy = MoEParallelStrategy(num_experts=12, expert_parallel_size=3)
        
        groups = strategy.create_expert_groups()
        
        assert len(groups) == 3
        assert groups[0] == [0, 1, 2, 3]
        assert groups[1] == [4, 5, 6, 7]
        assert groups[2] == [8, 9, 10, 11]
    
    def test_create_expert_groups_uneven(self):
        """测试不均匀专家分组"""
        strategy = MoEParallelStrategy(num_experts=10, expert_parallel_size=3)
        
        groups = strategy.create_expert_groups()
        
        assert len(groups) == 3
        assert groups[0] == [0, 1, 2, 3]  # 4 experts
        assert groups[1] == [4, 5, 6, 7]  # 4 experts
        assert groups[2] == [8, 9]         # 2 experts (remainder)
    
    def test_distributed_setup(self):
        """测试分布式设置"""
        strategy = MoEParallelStrategy(num_experts=8, expert_parallel_size=2)
        
        assert strategy.rank == 0
        assert strategy.world_size == 1
        assert strategy.expert_group is None
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 单专家情况
        strategy = MoEParallelStrategy(num_experts=1, expert_parallel_size=1)
        assert len(strategy.expert_groups) == 1
        assert strategy.expert_groups[0] == [0]
        
        # 专家数量少于并行大小
        strategy = MoEParallelStrategy(num_experts=2, expert_parallel_size=4)
        assert len(strategy.expert_groups) == 4
        assert strategy.expert_groups[0] == [0]
        assert strategy.expert_groups[1] == [1]
        assert strategy.expert_groups[2] == []
        assert strategy.expert_groups[3] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])