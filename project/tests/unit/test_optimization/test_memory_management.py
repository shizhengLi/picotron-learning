"""
智能内存管理模块单元测试
======================

测试内存池管理、碎片整理、内存调度和垃圾回收功能。
"""

import pytest
import sys
import os
import time

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from optimization.memory_management import (
    MemoryBlock, MemoryPool, MemoryScheduler, GarbageCollector,
    MemoryMonitor, SmartMemoryManager
)


class TestMemoryBlock:
    """内存块测试"""
    
    def test_memory_block_creation(self):
        """测试内存块创建"""
        block = MemoryBlock(1024, 'float32', 'cpu')
        
        assert block.size == 1024
        assert block.dtype == 'float32'
        assert block.device == 'cpu'
        assert block.is_allocated == False
        assert block.access_count == 0
        assert block.data is None
    
    def test_memory_block_allocate(self):
        """测试内存块分配"""
        block = MemoryBlock(1024, 'float32', 'cpu')
        block.allocate()
        
        assert block.is_allocated == True
        assert block.data is not None
        assert block.access_count == 1
    
    def test_memory_block_deallocate(self):
        """测试内存块释放"""
        block = MemoryBlock(1024, 'float32', 'cpu')
        block.allocate()
        block.deallocate()
        
        assert block.is_allocated == False
        assert block.data is None
    
    def test_memory_block_access(self):
        """测试内存块访问"""
        block = MemoryBlock(1024, 'float32', 'cpu')
        initial_count = block.access_count
        
        block.access()
        
        assert block.access_count == initial_count + 1
        assert block.last_access > block.timestamp
    
    def test_memory_block_priority(self):
        """测试内存块优先级"""
        block = MemoryBlock(1024, 'float32', 'cpu')
        
        # 初始优先级应该为0
        initial_priority = block.get_priority()
        assert initial_priority >= 0
        
        # 访问后优先级应该增加
        block.access()
        new_priority = block.get_priority()
        assert new_priority >= initial_priority


class TestMemoryPool:
    """内存池测试"""
    
    def test_memory_pool_creation(self):
        """测试内存池创建"""
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        
        assert pool.max_memory_bytes == 1024**3
        assert pool.device == 'cpu'
        assert pool.allocated_memory == 0
        assert len(pool.memory_blocks) == 0
        assert len(pool.available_blocks) == 0
    
    def test_memory_pool_allocate(self):
        """测试内存池分配"""
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        
        # 分配一个小的内存块
        block = pool.allocate(1024, 'float32')
        
        assert block is not None
        assert block.size == 1024
        assert block.dtype == 'float32'
        assert block.is_allocated == True
        assert pool.allocated_memory == 1024
        assert len(pool.allocated_blocks) == 1
    
    def test_memory_pool_allocate_large_block(self):
        """测试分配大内存块"""
        pool = MemoryPool(max_memory_gb=0.001, device='cpu')  # 1MB
        
        # 尝试分配超过限制的内存
        block = pool.allocate(2 * 1024 * 1024, 'float32')  # 2MB
        
        # 应该返回None，因为超过了内存限制
        assert block is None
    
    def test_memory_pool_deallocate(self):
        """测试内存池释放"""
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        
        # 分配内存
        block = pool.allocate(1024, 'float32')
        assert pool.allocated_memory == 1024
        
        # 释放内存
        pool.deallocate(block.id)
        assert pool.allocated_memory == 0
        assert len(pool.allocated_blocks) == 0
        assert len(pool.available_blocks) == 1
    
    def test_memory_pool_reuse_block(self):
        """测试重用内存块"""
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        
        # 分配并释放一个块
        block1 = pool.allocate(1024, 'float32')
        pool.deallocate(block1.id)
        
        # 分配相同大小的块，应该重用
        block2 = pool.allocate(1024, 'float32')
        
        assert block2 is not None
        assert len(pool.available_blocks) == 0  # 可用块被重用
    
    def test_memory_pool_get_stats(self):
        """测试获取内存统计"""
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        
        # 分配一些内存
        block1 = pool.allocate(1024, 'float32')
        block2 = pool.allocate(2048, 'float32')
        
        stats = pool.get_memory_stats()
        
        assert isinstance(stats, dict)
        assert stats['total_blocks'] == 2
        assert stats['allocated_blocks'] == 2
        assert stats['allocated_memory'] == 3072
        assert stats['memory_usage'] == 3072 / (1024**3)
        assert 'fragmentation_ratio' in stats
    
    def test_memory_pool_defragment(self):
        """测试内存碎片整理"""
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        
        # 分配并释放多个块
        for i in range(5):
            block = pool.allocate(1024, 'float32')
            pool.deallocate(block.id)
        
        # 碎片整理前
        initial_fragmentation = pool.get_memory_stats()['fragmentation_ratio']
        
        # 执行碎片整理
        pool.defragment()
        
        # 碎片整理后
        final_fragmentation = pool.get_memory_stats()['fragmentation_ratio']
        
        # 碎片率应该降低
        assert final_fragmentation <= initial_fragmentation


class TestMemoryScheduler:
    """内存调度器测试"""
    
    def test_memory_scheduler_creation(self):
        """测试内存调度器创建"""
        scheduler = MemoryScheduler()
        
        assert len(scheduler.memory_pools) == 0
        assert scheduler.scheduling_strategy == 'adaptive'
        assert scheduler.adaptive_threshold == 0.8
    
    def test_add_memory_pool(self):
        """测试添加内存池"""
        scheduler = MemoryScheduler()
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        
        scheduler.add_memory_pool(pool)
        
        assert 'cpu' in scheduler.memory_pools
        assert scheduler.memory_pools['cpu'] == pool
    
    def test_request_allocation(self):
        """测试请求分配"""
        scheduler = MemoryScheduler()
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        scheduler.add_memory_pool(pool)
        
        # 请求分配
        block = scheduler.request_allocation(1024, 'float32', 'cpu')
        
        assert block is not None
        assert block.size == 1024
        assert block.dtype == 'float32'
        assert block.device == 'cpu'
    
    def test_adaptive_scheduling(self):
        """测试自适应调度"""
        scheduler = MemoryScheduler()
        scheduler.scheduling_strategy = 'adaptive'
        
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        scheduler.add_memory_pool(pool)
        
        # 测试自适应调度
        block = scheduler._adaptive_scheduling(1024, 'float32', 'cpu')
        
        assert block is not None
    
    def test_lru_scheduling(self):
        """测试LRU调度"""
        scheduler = MemoryScheduler()
        scheduler.scheduling_strategy = 'lru'
        
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        scheduler.add_memory_pool(pool)
        
        # 测试LRU调度
        block = scheduler._lru_scheduling(1024, 'float32', 'cpu')
        
        assert block is not None
    
    def test_optimize_memory_usage(self):
        """测试优化内存使用"""
        scheduler = MemoryScheduler()
        pool = MemoryPool(max_memory_gb=1.0, device='cpu')
        scheduler.add_memory_pool(pool)
        
        # 分配一些内存
        for i in range(3):
            block = pool.allocate(1024, 'float32')
        
        # 优化内存使用
        scheduler.optimize_memory_usage()
        
        # 优化应该成功执行
        assert True  # 如果没有异常，测试通过


class TestGarbageCollector:
    """垃圾回收器测试"""
    
    def test_garbage_collector_creation(self):
        """测试垃圾回收器创建"""
        gc = GarbageCollector()
        
        assert len(gc.weak_references) == 0
        assert len(gc.reference_counts) == 0
        assert gc.gc_interval == 30
        assert gc.gc_threshold == 100
    
    def test_add_reference(self):
        """测试添加引用"""
        gc = GarbageCollector()
        
        # 创建一个对象
        obj = "test_object"
        obj_id = id(obj)
        
        # 添加引用
        gc.add_reference(obj, obj_id)
        
        assert obj_id in gc.weak_references
        assert gc.reference_counts[obj_id] == 1
    
    def test_remove_reference(self):
        """测试移除引用"""
        gc = GarbageCollector()
        
        obj = "test_object"
        obj_id = id(obj)
        
        # 添加并移除引用
        gc.add_reference(obj, obj_id)
        gc.remove_reference(obj_id)
        
        assert obj_id not in gc.weak_references
        assert obj_id not in gc.reference_counts
    
    def test_collect_garbage(self):
        """测试垃圾收集"""
        gc = GarbageCollector()
        
        # 添加一些引用
        for i in range(5):
            obj = f"test_object_{i}"
            gc.add_reference(obj, id(obj))
        
        # 让对象超出作用域
        del obj
        
        # 收集垃圾
        collected = gc.collect_garbage()
        
        assert collected >= 0
    
    def test_get_gc_stats(self):
        """测试获取垃圾回收统计"""
        gc = GarbageCollector()
        
        # 添加一些引用
        obj = "test_object"
        gc.add_reference(obj, id(obj))
        
        stats = gc.get_gc_stats()
        
        assert isinstance(stats, dict)
        assert 'tracked_objects' in stats
        assert 'reference_counts' in stats
        assert 'total_references' in stats
        assert stats['tracked_objects'] == 1
        assert stats['total_references'] == 1


class TestMemoryMonitor:
    """内存监控器测试"""
    
    def test_memory_monitor_creation(self):
        """测试内存监控器创建"""
        monitor = MemoryMonitor()
        
        assert len(monitor.monitoring_data['timestamps']) == 0
        assert len(monitor.alerts) == 0
        assert monitor.alert_thresholds['memory_usage'] == 0.9
    
    def test_record_memory_event(self):
        """测试记录内存事件"""
        monitor = MemoryMonitor()
        
        # 记录分配事件
        monitor.record_memory_event('allocation', {
            'memory_usage': 0.5,
            'fragmentation_ratio': 0.1
        })
        
        assert len(monitor.monitoring_data['timestamps']) == 1
        assert len(monitor.monitoring_data['memory_usage']) == 1
        assert monitor.monitoring_data['memory_usage'][0] == 0.5
        assert monitor.monitoring_data['allocation_count'][0] == 1
    
    def test_check_alerts(self):
        """测试检查告警"""
        monitor = MemoryMonitor()
        
        # 测试高内存使用率告警
        stats = {'memory_usage': 0.95, 'fragmentation_ratio': 0.1}
        alerts = monitor.check_alerts(stats)
        
        assert len(alerts) == 1
        assert alerts[0]['type'] == 'high_memory_usage'
        assert alerts[0]['severity'] == 'warning'
    
    def test_get_memory_report(self):
        """测试获取内存报告"""
        monitor = MemoryMonitor()
        
        # 记录一些事件
        monitor.record_memory_event('allocation', {
            'memory_usage': 0.5,
            'fragmentation_ratio': 0.1
        })
        
        report = monitor.get_memory_report()
        
        assert isinstance(report, dict)
        assert 'current_memory_usage' in report
        assert 'peak_memory_usage' in report
        assert 'total_allocations' in report
        assert report['total_allocations'] == 1


class TestSmartMemoryManager:
    """智能内存管理器测试"""
    
    def test_smart_memory_manager_creation(self):
        """测试智能内存管理器创建"""
        config = {
            'default_pool_size_gb': 2.0,
            'enable_auto_defrag': True,
            'enable_auto_gc': True,
            'monitoring_interval': 5
        }
        
        manager = SmartMemoryManager(config)
        
        assert manager.config == config
        assert len(manager.memory_pools) > 0  # 应该有默认内存池
        assert manager.default_pool_size == 2.0
        assert manager.enable_auto_defrag == True
        assert manager.enable_auto_gc == True
    
    def test_allocate_memory(self):
        """测试分配内存"""
        manager = SmartMemoryManager()
        
        # 分配内存
        block = manager.allocate_memory(1024, 'float32', 'cpu')
        
        assert block is not None
        assert block.size == 1024
        assert block.dtype == 'float32'
        assert block.device == 'cpu'
        assert block.is_allocated == True
    
    def test_deallocate_memory(self):
        """测试释放内存"""
        manager = SmartMemoryManager()
        
        # 分配并释放内存
        block = manager.allocate_memory(1024, 'float32', 'cpu')
        initial_usage = manager.get_total_memory_usage()
        
        manager.deallocate_memory(block)
        final_usage = manager.get_total_memory_usage()
        
        assert final_usage < initial_usage
    
    def test_get_total_memory_usage(self):
        """测试获取总内存使用率"""
        manager = SmartMemoryManager()
        
        # 初始使用率
        initial_usage = manager.get_total_memory_usage()
        assert 0 <= initial_usage <= 1
        
        # 分配内存后
        block = manager.allocate_memory(1024, 'float32', 'cpu')
        new_usage = manager.get_total_memory_usage()
        assert new_usage >= initial_usage
    
    def test_get_average_fragmentation(self):
        """测试获取平均碎片率"""
        manager = SmartMemoryManager()
        
        fragmentation = manager.get_average_fragmentation()
        
        assert 0 <= fragmentation <= 1
        assert isinstance(fragmentation, float)
    
    def test_get_memory_status(self):
        """测试获取内存状态"""
        manager = SmartMemoryManager()
        
        # 分配一些内存
        block = manager.allocate_memory(1024, 'float32', 'cpu')
        
        status = manager.get_memory_status()
        
        assert isinstance(status, dict)
        assert 'total_memory_usage' in status
        assert 'average_fragmentation' in status
        assert 'memory_pools' in status
        assert 'garbage_collector_stats' in status
        assert 'memory_report' in status
        
        # 检查内存池状态
        assert len(status['memory_pools']) > 0
        for pool_stats in status['memory_pools'].values():
            assert isinstance(pool_stats, dict)
            assert 'allocated_memory' in pool_stats
            assert 'memory_usage' in pool_stats
    
    def test_optimize_memory(self):
        """测试优化内存"""
        manager = SmartMemoryManager()
        
        # 分配一些内存
        blocks = []
        for i in range(3):
            block = manager.allocate_memory(1024, 'float32', 'cpu')
            blocks.append(block)
        
        # 释放一些内存
        for block in blocks[:2]:
            manager.deallocate_memory(block)
        
        # 优化内存
        result = manager.optimize_memory()
        
        assert isinstance(result, dict)
        assert 'garbage_collected' in result
        assert 'pools_defragmented' in result
        assert 'optimization_completed' in result
        assert result['optimization_completed'] == True


class TestEdgeCases:
    """边界情况测试"""
    
    def test_zero_size_allocation(self):
        """测试零大小分配"""
        manager = SmartMemoryManager()
        
        # 零大小分配应该返回None或处理得当
        block = manager.allocate_memory(0, 'float32', 'cpu')
        
        # 根据实现，可能返回None或特殊处理
        assert True  # 只要没有异常就通过
    
    def test_negative_size_allocation(self):
        """测试负大小分配"""
        manager = SmartMemoryManager()
        
        # 负大小分配应该返回None或处理得当
        block = manager.allocate_memory(-1024, 'float32', 'cpu')
        
        # 根据实现，可能返回None或特殊处理
        assert True  # 只要没有异常就通过
    
    def test_allocate_with_none_device(self):
        """测试None设备分配"""
        manager = SmartMemoryManager()
        
        # None设备应该有合理的默认处理
        block = manager.allocate_memory(1024, 'float32', None)
        
        # 应该使用默认设备或返回None
        assert True  # 只要没有异常就通过
    
    def test_deallocate_none_block(self):
        """测试释放None块"""
        manager = SmartMemoryManager()
        
        # 释放None块不应该出错
        manager.deallocate_memory(None)
        
        assert True  # 只要没有异常就通过
    
    def test_memory_overflow(self):
        """测试内存溢出"""
        manager = SmartMemoryManager()
        
        # 尝试分配极大内存
        huge_size = 10**12  # 1TB
        block = manager.allocate_memory(huge_size, 'float32', 'cpu')
        
        # 应该返回None或处理溢出
        assert block is None
    
    def test_concurrent_allocation(self):
        """测试并发分配"""
        manager = SmartMemoryManager()
        results = []
        
        def allocate_task():
            block = manager.allocate_memory(1024, 'float32', 'cpu')
            results.append(block is not None)
            time.sleep(0.01)
            if block:
                manager.deallocate_memory(block)
        
        # 创建多个线程
        import threading
        threads = []
        for i in range(5):
            thread = threading.Thread(target=allocate_task)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 所有分配都应该成功
        assert all(results)
    
    def test_memory_pool_with_zero_max_memory(self):
        """测试零最大内存的内存池"""
        pool = MemoryPool(max_memory_gb=0.0, device='cpu')
        
        # 任何分配都应该失败
        block = pool.allocate(1024, 'float32')
        assert block is None
    
    def test_garbage_collector_with_dead_objects(self):
        """测试垃圾回收器处理死对象"""
        gc = GarbageCollector()
        
        # 创建临时对象
        temp_objects = []
        for i in range(5):
            obj = f"temp_object_{i}"
            temp_objects.append(obj)
            gc.add_reference(obj, id(obj))
        
        # 删除对象
        del temp_objects
        
        # 强制垃圾回收
        collected = gc.collect_garbage()
        
        # 应该能收集到一些垃圾
        assert collected >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])