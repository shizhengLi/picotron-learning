"""
MoE并行策略单元测试
=================
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from parallel.moe import ExpertLayer, Router, MoEParallelStrategy, MoEModel, LoadBalancer


class TestExpertLayer:
    """专家层测试"""
    
    def test_expert_layer_creation(self):
        """测试专家层创建"""
        expert = ExpertLayer(input_dim=64, output_dim=32, expert_id=0)
        
        assert expert.expert_id == 0
        assert expert.expert_group is None
        assert isinstance(expert.network, nn.Sequential)
        assert len(expert.network) == 3
    
    def test_expert_layer_forward(self):
        """测试专家层前向传播"""
        expert = ExpertLayer(input_dim=64, output_dim=32, expert_id=0)
        
        # 创建测试输入
        x = torch.randn(10, 64)
        
        # 前向传播
        output = expert(x)
        
        # 验证输出形状
        assert output.shape == (10, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestRouter:
    """路由器测试"""
    
    def test_router_creation(self):
        """测试路由器创建"""
        router = Router(input_dim=64, num_experts=8, k=2)
        
        assert router.num_experts == 8
        assert router.k == 2
        assert isinstance(router.gate, nn.Linear)
    
    def test_router_forward(self):
        """测试路由器前向传播"""
        router = Router(input_dim=64, num_experts=4, k=2)
        
        # 创建测试输入
        x = torch.randn(10, 64)
        
        # 前向传播
        weights, indices = router(x)
        
        # 验证输出形状
        assert weights.shape == (10, 2)
        assert indices.shape == (10, 2)
        
        # 验证权重范围
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
        
        # 验证权重和为1
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
        
        # 验证索引范围
        assert torch.all(indices >= 0)
        assert torch.all(indices < 4)
    
    def test_router_forward_single_expert(self):
        """测试单专家路由"""
        router = Router(input_dim=64, num_experts=4, k=1)
        
        x = torch.randn(5, 64)
        weights, indices = router(x)
        
        assert weights.shape == (5, 1)
        assert indices.shape == (5, 1)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(5))


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


class TestMoEModel:
    """MoE模型测试"""
    
    def test_moe_model_creation(self):
        """测试MoE模型创建"""
        model = MoEModel(
            input_dim=64, 
            output_dim=10, 
            num_experts=4, 
            hidden_dim=32, 
            k=2
        )
        
        assert model.num_experts == 4
        assert model.k == 2
        assert isinstance(model.router, Router)
        assert len(model.experts) == 4
        assert isinstance(model.output_layer, nn.Linear)
    
    def test_moe_model_forward(self):
        """测试MoE模型前向传播"""
        model = MoEModel(
            input_dim=64, 
            output_dim=10, 
            num_experts=4, 
            hidden_dim=32, 
            k=2
        )
        
        # 创建测试输入
        x = torch.randn(5, 10, 64)  # (batch_size, seq_len, input_dim)
        
        # 前向传播
        output = model(x)
        
        # 验证输出形状
        assert output.shape == (5, 10, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_moe_model_forward_different_batch_sizes(self):
        """测试不同批量大小的前向传播"""
        model = MoEModel(
            input_dim=32, 
            output_dim=5, 
            num_experts=3, 
            hidden_dim=16, 
            k=1
        )
        
        # 测试不同批量大小
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 20, 32)
            output = model(x)
            assert output.shape == (batch_size, 20, 5)
    
    def test_moe_model_forward_different_k(self):
        """测试不同k值的前向传播"""
        model = MoEModel(
            input_dim=32, 
            output_dim=5, 
            num_experts=4, 
            hidden_dim=16, 
            k=3
        )
        
        x = torch.randn(4, 10, 32)
        output = model(x)
        
        assert output.shape == (4, 10, 5)
    
    def test_moe_model_training_step(self):
        """测试MoE模型训练步骤"""
        model = MoEModel(
            input_dim=32, 
            output_dim=5, 
            num_experts=3, 
            hidden_dim=16, 
            k=2
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        # 模拟训练步骤
        x = torch.randn(8, 15, 32)
        target = torch.randn(8, 15, 5)
        
        # 前向传播
        output = model(x)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证损失是标量
        assert loss.shape == ()
        assert loss.item() > 0
        
        # 验证梯度存在
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])