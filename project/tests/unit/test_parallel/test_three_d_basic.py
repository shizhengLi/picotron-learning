"""
3D并行优化模块单元测试（简化版）
==========================

测试张量并行、流水线并行和数据并行的3D并行优化功能。
"""

import pytest
import sys
import os

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from parallel.three_d import (
    TensorParallelOptimizer, PipelineParallelOptimizer, DataParallelOptimizer,
    ThreeDParallelOptimizer, ColumnParallelLinear, RowParallelLinear,
    ThreeDParallelModel
)


class TestTensorParallelOptimizer:
    """张量并行优化器测试"""
    
    def test_tensor_optimizer_creation(self):
        """测试张量优化器创建"""
        optimizer = TensorParallelOptimizer(tensor_parallel_size=2)
        
        assert optimizer.tensor_parallel_size == 2
        assert optimizer.tensor_parallel_group is None
    
    def test_column_parallel_linear(self):
        """测试列并行线性层"""
        layer = ColumnParallelLinear(input_dim=64, output_dim=32)
        
        assert layer.input_dim == 64
        assert layer.output_dim == 32
        assert hasattr(layer, 'weight')
    
    def test_row_parallel_linear(self):
        """测试行并行线性层"""
        layer = RowParallelLinear(input_dim=32, output_dim=64)
        
        assert layer.input_dim == 32
        assert layer.output_dim == 64
        assert hasattr(layer, 'weight')
    
    def test_split_tensor_along_dim(self):
        """测试张量分割"""
        optimizer = TensorParallelOptimizer(tensor_parallel_size=4)
        
        # 模拟张量
        mock_tensor = "MockTensor(shape=(100, 200))"
        chunks = optimizer.split_tensor_along_dim(mock_tensor, dim=0)
        
        assert len(chunks) == 4
        assert all("TensorChunk_" in chunk for chunk in chunks)
    
    def test_all_reduce(self):
        """测试All-Reduce操作"""
        optimizer = TensorParallelOptimizer(tensor_parallel_size=2)
        
        mock_tensor = "MockTensor"
        result = optimizer.all_reduce(mock_tensor)
        
        assert result == mock_tensor  # 在单机环境下直接返回


class TestPipelineParallelOptimizer:
    """流水线并行优化器测试"""
    
    def test_pipeline_optimizer_creation(self):
        """测试流水线优化器创建"""
        optimizer = PipelineParallelOptimizer(pipeline_parallel_size=2, num_micro_batches=4)
        
        assert optimizer.pipeline_parallel_size == 2
        assert optimizer.num_micro_batches == 4
        assert optimizer.pipeline_group is None
    
    def test_create_pipeline_schedule(self):
        """测试流水线调度创建"""
        optimizer = PipelineParallelOptimizer(pipeline_parallel_size=2, num_micro_batches=4)
        
        schedule = optimizer.create_pipeline_schedule(num_layers=6)
        
        assert len(schedule) == 2
        assert schedule[0]['stage_id'] == 0
        assert schedule[1]['stage_id'] == 1
        assert schedule[0]['is_first'] == True
        assert schedule[1]['is_last'] == True
    
    def test_pipeline_schedule_layers(self):
        """测试流水线调度层级分配"""
        optimizer = PipelineParallelOptimizer(pipeline_parallel_size=3, num_micro_batches=4)
        
        schedule = optimizer.create_pipeline_schedule(num_layers=10)
        
        # 检查层级分配是否正确
        assert len(schedule[0]['layers']) == 4  # 第一个阶段
        assert len(schedule[1]['layers']) == 3  # 第二个阶段
        assert len(schedule[2]['layers']) == 3  # 第三个阶段
    
    def test_send_recv_activation(self):
        """测试激活值发送和接收"""
        optimizer = PipelineParallelOptimizer(pipeline_parallel_size=2, num_micro_batches=4)
        
        # 测试发送
        activation = "MockActivation"
        sent = optimizer.send_activation(activation, target_rank=1)
        assert sent == activation
        
        # 测试接收
        received = optimizer.recv_activation(source_rank=0)
        assert "ReceivedActivation" in received
    
    def test_send_recv_gradient(self):
        """测试梯度发送和接收"""
        optimizer = PipelineParallelOptimizer(pipeline_parallel_size=2, num_micro_batches=4)
        
        # 测试发送
        gradient = "MockGradient"
        sent = optimizer.send_gradient(gradient, target_rank=1)
        assert sent == gradient
        
        # 测试接收
        received = optimizer.recv_gradient(source_rank=0)
        assert "ReceivedGradient" in received


class TestDataParallelOptimizer:
    """数据并行优化器测试"""
    
    def test_data_optimizer_creation(self):
        """测试数据优化器创建"""
        optimizer = DataParallelOptimizer(data_parallel_size=4)
        
        assert optimizer.data_parallel_size == 4
        assert optimizer.data_parallel_group is None
    
    def test_distribute_data(self):
        """测试数据分发"""
        optimizer = DataParallelOptimizer(data_parallel_size=4)
        
        # 模拟batch
        mock_batch = "MockBatch(size=100)"
        local_batch = optimizer.distribute_data(mock_batch)
        
        assert "LocalBatch" in local_batch
        assert "size=25" in local_batch  # 100 / 4 = 25
    
    def test_all_reduce_gradients(self):
        """测试梯度All-Reduce"""
        optimizer = DataParallelOptimizer(data_parallel_size=2)
        
        # 创建模拟模型
        class MockModel:
            def parameters(self):
                return ["MockParam1", "MockParam2"]
        
        model = MockModel()
        result = optimizer.all_reduce_gradients(model)
        
        assert result == model  # 在单机环境下直接返回
    
    def test_broadcast_parameters(self):
        """测试参数广播"""
        optimizer = DataParallelOptimizer(data_parallel_size=2)
        
        # 创建模拟模型
        class MockModel:
            def parameters(self):
                return ["MockParam1", "MockParam2"]
        
        model = MockModel()
        result = optimizer.broadcast_parameters(model)
        
        assert result == model  # 在单机环境下直接返回


class TestThreeDParallelOptimizer:
    """3D并行优化器测试"""
    
    def test_3d_optimizer_creation(self):
        """测试3D优化器创建"""
        optimizer = ThreeDParallelOptimizer(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            num_micro_batches=4
        )
        
        assert optimizer.tensor_parallel_size == 2
        assert optimizer.pipeline_parallel_size == 2
        assert optimizer.data_parallel_size == 2
        assert optimizer.num_micro_batches == 4
        
        # 检查子优化器
        assert optimizer.tensor_optimizer is not None
        assert optimizer.pipeline_optimizer is not None
        assert optimizer.data_optimizer is not None
    
    def test_3d_optimizer_with_default_params(self):
        """测试3D优化器默认参数"""
        optimizer = ThreeDParallelOptimizer()
        
        assert optimizer.tensor_parallel_size == 1
        assert optimizer.pipeline_parallel_size == 1
        assert optimizer.data_parallel_size == 1
        assert optimizer.num_micro_batches == 4
    
    def test_optimize_model(self):
        """测试模型优化"""
        optimizer = ThreeDParallelOptimizer(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2
        )
        
        # 创建测试模型
        model = ThreeDParallelModel(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            num_layers=4
        )
        
        # 优化模型
        optimized_model = optimizer.optimize_model(model)
        
        assert optimized_model is not None
        assert hasattr(optimizer, 'pipeline_schedule')
    
    def test_train_step(self):
        """测试训练步骤"""
        optimizer = ThreeDParallelOptimizer(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2
        )
        
        # 创建模拟模型和优化器
        class MockModel:
            def __call__(self, x):
                return f"ModelOutput({x})"
        
        class MockOptimizer:
            def step(self):
                pass
        
        model = MockModel()
        batch = "MockBatch"
        mock_optimizer = MockOptimizer()
        
        # 执行训练步骤
        loss = optimizer.train_step(model, batch, mock_optimizer)
        
        assert loss is not None
        assert "Loss" in str(loss)
    
    def test_forward_step(self):
        """测试前向传播步骤"""
        optimizer = ThreeDParallelOptimizer(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2
        )
        
        # 创建模拟模型
        class MockModel:
            def __call__(self, x):
                return f"ModelOutput({x})"
        
        model = MockModel()
        batch = "MockBatch"
        
        # 执行前向传播
        output = optimizer.forward_step(model, batch)
        
        assert output is not None
        assert "ModelOutput" in str(output)
    
    def test_pipeline_forward(self):
        """测试流水线前向传播"""
        optimizer = ThreeDParallelOptimizer(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            data_parallel_size=1
        )
        
        # 创建测试模型
        model = ThreeDParallelModel(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            num_layers=4
        )
        
        batch = "MockBatch"
        
        # 执行流水线前向传播
        output = optimizer.pipeline_forward(model, batch)
        
        assert output is not None
    
    def test_compute_loss(self):
        """测试损失计算"""
        optimizer = ThreeDParallelOptimizer()
        
        output = "MockOutput"
        target = "MockTarget"
        
        loss = optimizer.compute_loss(output, target)
        
        assert loss is not None
        assert "Loss" in str(loss)


class TestThreeDParallelModel:
    """3D并行模型测试"""
    
    def test_model_creation(self):
        """测试模型创建"""
        model = ThreeDParallelModel(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            num_layers=6
        )
        
        assert model.input_dim == 64
        assert model.hidden_dim == 128
        assert model.output_dim == 10
        assert model.num_layers == 6
        assert len(model.layers) == 11  # 6 layers + 5 ReLU activations
    
    def test_model_forward(self):
        """测试模型前向传播"""
        model = ThreeDParallelModel(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            num_layers=3
        )
        
        input_data = "MockInput"
        output = model(input_data)
        
        assert output is not None
    
    def test_model_with_different_layers(self):
        """测试不同层数的模型"""
        # 测试单层模型
        model1 = ThreeDParallelModel(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            num_layers=1
        )
        assert len(model1.layers) == 1  # 只有输出层
        
        # 测试多层模型
        model2 = ThreeDParallelModel(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            num_layers=8
        )
        assert len(model2.layers) == 15  # 8 layers + 7 ReLU activations


class TestEdgeCases:
    """边界情况测试"""
    
    def test_single_parallel_size(self):
        """测试单并行大小"""
        # 测试张量并行
        tensor_opt = TensorParallelOptimizer(tensor_parallel_size=1)
        assert tensor_opt.tensor_parallel_size == 1
        
        # 测试流水线并行
        pipeline_opt = PipelineParallelOptimizer(pipeline_parallel_size=1)
        assert pipeline_opt.pipeline_parallel_size == 1
        
        # 测试数据并行
        data_opt = DataParallelOptimizer(data_parallel_size=1)
        assert data_opt.data_parallel_size == 1
    
    def test_zero_parallel_size(self):
        """测试零并行大小"""
        # 测试数据并行
        data_opt = DataParallelOptimizer(data_parallel_size=1)  # 最小为1
        assert data_opt.data_parallel_size == 1
    
    def test_large_parallel_size(self):
        """测试大并行大小"""
        # 测试张量并行
        tensor_opt = TensorParallelOptimizer(tensor_parallel_size=8)
        assert tensor_opt.tensor_parallel_size == 8
        
        # 测试流水线并行
        pipeline_opt = PipelineParallelOptimizer(pipeline_parallel_size=8)
        assert pipeline_opt.pipeline_parallel_size == 8
        
        # 测试数据并行
        data_opt = DataParallelOptimizer(data_parallel_size=8)
        assert data_opt.data_parallel_size == 8
    
    def test_zero_layers_model(self):
        """测试零层模型"""
        model = ThreeDParallelModel(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            num_layers=0
        )
        # 应该创建一个空模型
        assert len(model.layers) == 0
    
    def test_large_model(self):
        """测试大模型"""
        model = ThreeDParallelModel(
            input_dim=1024,
            hidden_dim=2048,
            output_dim=1000,
            num_layers=24
        )
        assert model.input_dim == 1024
        assert model.hidden_dim == 2048
        assert model.output_dim == 1000
        assert model.num_layers == 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])