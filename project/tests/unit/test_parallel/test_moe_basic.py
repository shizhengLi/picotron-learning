"""
MoE并行策略单元测试（简化版）
=======================
"""

import pytest
import sys
import os

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

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
        
        # 验证负载被重新平衡
        # 原来的负载 [30, 5, 10, 5]
        # 平均负载 = 12.5
        # 容量因子 = 1.2, 阈值 = 15
        # 30 > 15 -> 调整为15
        # 5 < 6.25 (0.5 * 12.5) -> 调整为10 (0.8 * 12.5)
        # 10 在正常范围内 -> 保持不变
        # 5 < 6.25 -> 调整为10
        expected_load = [15, 10, 10, 10]
        assert balancer.expert_load == expected_load
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空负载
        balancer = LoadBalancer(num_experts=0, expert_parallel_size=1)
        stats = balancer.get_load_stats()
        assert stats['max_load'] == 0
        assert stats['avg_load'] == 0
        
        # 单专家
        balancer = LoadBalancer(num_experts=1, expert_parallel_size=1)
        balancer.update_load(0, 100)
        stats = balancer.get_load_stats()
        assert stats['max_load'] == 100
        assert stats['min_load'] == 100


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
        # 10个专家，3个组：每组3个，余数1个
        # 组0: [0, 1, 2, 3] (4个)
        # 组1: [4, 5, 6] (3个)
        # 组2: [7, 8, 9] (3个)
        assert groups[0] == [0, 1, 2, 3]  # 4 experts
        assert groups[1] == [4, 5, 6]     # 3 experts
        assert groups[2] == [7, 8, 9]     # 3 experts
    
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
        
        # 零专家情况
        strategy = MoEParallelStrategy(num_experts=0, expert_parallel_size=2)
        assert len(strategy.expert_groups) == 2
        assert strategy.expert_groups[0] == []
        assert strategy.expert_groups[1] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])