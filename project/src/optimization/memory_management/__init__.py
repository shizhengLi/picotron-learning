"""
智能内存管理模块
================

实现高效的内存管理策略，包括内存池管理、碎片整理、内存调度和垃圾回收。
"""

try:
    import torch
    import torch.cuda as cuda
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import gc
    HAS_GC = True
except ImportError:
    HAS_GC = False

from typing import Dict, List, Any, Optional, Tuple, Union, Set
import threading
import time
import logging
import weakref
from collections import defaultdict, deque
import heapq


class MemoryBlock:
    """内存块类"""
    
    def __init__(self, size: int, dtype: str, device: str = 'cpu'):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.timestamp = time.time()
        self.access_count = 0
        self.last_access = self.timestamp
        self.is_allocated = False
        self.data = None
        self.id = id(self)
        
    def allocate(self):
        """分配内存"""
        if not self.is_allocated:
            if HAS_TORCH and self.device.startswith('cuda'):
                self.data = torch.empty(self.size, dtype=self._get_torch_dtype(), device=self.device)
            elif HAS_TORCH:
                self.data = torch.empty(self.size, dtype=self._get_torch_dtype(), device='cpu')
            else:
                self.data = [0] * self.size
            self.is_allocated = True
            self.timestamp = time.time()
            self.access_count += 1
            self.last_access = self.timestamp
            
    def deallocate(self):
        """释放内存"""
        if self.is_allocated:
            self.data = None
            self.is_allocated = False
            
    def access(self):
        """访问内存块"""
        self.access_count += 1
        self.last_access = time.time()
        
    def _get_torch_dtype(self):
        """获取PyTorch数据类型"""
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'int32': torch.int32,
            'int64': torch.int64,
            'bool': torch.bool
        }
        return dtype_map.get(self.dtype, torch.float32)
        
    def get_priority(self):
        """获取内存块优先级（用于LRU算法）"""
        # 综合考虑访问频率和最近访问时间
        frequency_score = self.access_count
        recency_score = 1.0 / (1.0 + time.time() - self.last_access)
        return frequency_score * recency_score


class MemoryPool:
    """内存池管理器"""
    
    def __init__(self, max_memory_gb: float = 8.0, device: str = 'cpu'):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.device = device
        self.allocated_memory = 0
        self.memory_blocks = {}  # id -> MemoryBlock
        self.available_blocks = []  # 优先队列
        self.allocated_blocks = {}  # id -> MemoryBlock
        self.block_size_groups = defaultdict(list)  # size -> [MemoryBlock]
        
        # 内存统计
        self.allocation_history = deque(maxlen=1000)
        self.fragmentation_history = deque(maxlen=100)
        
        # 配置参数
        self.min_block_size = 1024  # 1KB
        self.max_block_size = 1024**3  # 1GB
        self.fragmentation_threshold = 0.3  # 30%碎片率阈值
        
        # 锁保护并发访问
        self.lock = threading.Lock()
        
    def allocate(self, size: int, dtype: str = 'float32') -> Optional[MemoryBlock]:
        """分配内存块"""
        with self.lock:
            # 检查内存限制
            if self.allocated_memory + size > self.max_memory_bytes:
                # 尝试释放内存
                if not self._free_memory(size):
                    return None
            
            # 尝试重用现有块
            block = self._reuse_block(size, dtype)
            if block:
                block.allocate()
                self.allocated_blocks[block.id] = block
                self.allocation_history.append({
                    'action': 'allocate',
                    'size': size,
                    'timestamp': time.time(),
                    'type': 'reuse'
                })
                return block
            
            # 创建新块
            block = MemoryBlock(size, dtype, self.device)
            block.allocate()
            
            # 更新状态
            self.memory_blocks[block.id] = block
            self.allocated_blocks[block.id] = block
            self.block_size_groups[size].append(block)
            self.allocated_memory += size
            
            self.allocation_history.append({
                'action': 'allocate',
                'size': size,
                'timestamp': time.time(),
                'type': 'new'
            })
            
            return block
    
    def deallocate(self, block_id: int):
        """释放内存块"""
        with self.lock:
            if block_id in self.allocated_blocks:
                block = self.allocated_blocks[block_id]
                block.deallocate()
                del self.allocated_blocks[block_id]
                self.allocated_memory -= block.size
                
                # 添加到可用块列表
                heapq.heappush(self.available_blocks, (-block.get_priority(), block.id, block))
                
                self.allocation_history.append({
                    'action': 'deallocate',
                    'size': block.size,
                    'timestamp': time.time()
                })
    
    def _reuse_block(self, size: int, dtype: str) -> Optional[MemoryBlock]:
        """重用现有内存块"""
        # 查找大小相近的块
        tolerance = max(size * 0.1, 1024)  # 10%容差或1KB
        
        # 遍历可用块
        for i, (_, block_id, block) in enumerate(self.available_blocks):
            if abs(block.size - size) <= tolerance and block.dtype == dtype:
                # 从可用块中移除
                self.available_blocks.pop(i)
                heapq.heapify(self.available_blocks)
                return block
        
        return None
    
    def _free_memory(self, required_size: int) -> bool:
        """释放内存以满足需求"""
        # 按LRU策略释放内存
        freed = 0
        blocks_to_free = []
        
        # 从分配的块中选择最不常用的
        allocated_list = list(self.allocated_blocks.values())
        allocated_list.sort(key=lambda x: x.get_priority())
        
        for block in allocated_list:
            if freed >= required_size:
                break
            
            blocks_to_free.append(block)
            freed += block.size
        
        # 释放选中的块
        for block in blocks_to_free:
            self.deallocate(block.id)
        
        return freed >= required_size
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        with self.lock:
            total_blocks = len(self.memory_blocks)
            allocated_blocks = len(self.allocated_blocks)
            available_blocks = len(self.available_blocks)
            
            # 计算碎片率
            if available_blocks > 1:
                available_sizes = [block.size for _, _, block in self.available_blocks]
                total_available = sum(available_sizes)
                max_contiguous = max(available_sizes) if available_sizes else 0
                fragmentation = 1.0 - (max_contiguous / total_available) if total_available > 0 else 0.0
            else:
                fragmentation = 0.0
            
            return {
                'total_blocks': total_blocks,
                'allocated_blocks': allocated_blocks,
                'available_blocks': available_blocks,
                'allocated_memory': self.allocated_memory,
                'max_memory': self.max_memory_bytes,
                'memory_usage': self.allocated_memory / self.max_memory_bytes,
                'fragmentation_ratio': fragmentation,
                'device': self.device
            }
    
    def defragment(self):
        """内存碎片整理"""
        with self.lock:
            if len(self.available_blocks) < 2:
                return
            
            # 合并相邻大小的块
            size_groups = defaultdict(list)
            for _, _, block in self.available_blocks:
                size_groups[block.size].append(block)
            
            # 重新整理可用块列表
            self.available_blocks = []
            for size, blocks in size_groups.items():
                # 保留一些块以满足不同大小的需求
                keep_count = min(3, len(blocks))
                for block in blocks[:keep_count]:
                    heapq.heappush(self.available_blocks, (-block.get_priority(), block.id, block))
                
                # 释放多余的块
                for block in blocks[keep_count:]:
                    del self.memory_blocks[block.id]
            
            self.fragmentation_history.append({
                'timestamp': time.time(),
                'fragmentation_ratio': self.get_memory_stats()['fragmentation_ratio']
            })


class MemoryScheduler:
    """内存调度器"""
    
    def __init__(self):
        self.memory_pools = {}  # device -> MemoryPool
        self.allocation_requests = deque()
        self.scheduler_thread = None
        self.is_running = False
        
        # 调度策略
        self.scheduling_strategy = 'adaptive'  # 'fifo', 'lru', 'adaptive'
        self.adaptive_threshold = 0.8  # 内存使用率阈值
        
    def add_memory_pool(self, pool: MemoryPool):
        """添加内存池"""
        self.memory_pools[pool.device] = pool
        
    def request_allocation(self, size: int, dtype: str, device: str = 'cpu') -> Optional[MemoryBlock]:
        """请求内存分配"""
        # 选择最佳内存池
        pool = self._select_memory_pool(size, device)
        if not pool:
            return None
        
        # 尝试分配
        block = pool.allocate(size, dtype)
        if block:
            return block
        
        # 如果分配失败，尝试调度策略
        return self._schedule_allocation(size, dtype, device)
    
    def _select_memory_pool(self, size: int, device: str) -> Optional[MemoryPool]:
        """选择最佳内存池"""
        # 优先选择指定设备的内存池
        if device in self.memory_pools:
            pool = self.memory_pools[device]
            if pool.allocated_memory + size <= pool.max_memory_bytes:
                return pool
        
        # 如果指定设备不可用，尝试其他设备
        for pool_device, pool in self.memory_pools.items():
            if pool.allocated_memory + size <= pool.max_memory_bytes:
                return pool
        
        return None
    
    def _schedule_allocation(self, size: int, dtype: str, device: str) -> Optional[MemoryBlock]:
        """调度内存分配"""
        if self.scheduling_strategy == 'adaptive':
            return self._adaptive_scheduling(size, dtype, device)
        elif self.scheduling_strategy == 'lru':
            return self._lru_scheduling(size, dtype, device)
        else:
            return self._fifo_scheduling(size, dtype, device)
    
    def _adaptive_scheduling(self, size: int, dtype: str, device: str) -> Optional[MemoryBlock]:
        """自适应调度"""
        # 根据内存使用情况选择策略
        for pool in self.memory_pools.values():
            usage_ratio = pool.allocated_memory / pool.max_memory_bytes
            
            if usage_ratio < self.adaptive_threshold:
                # 内存使用率较低，直接分配
                block = pool.allocate(size, dtype)
                if block:
                    return block
            else:
                # 内存使用率高，尝试释放内存
                if pool._free_memory(size):
                    block = pool.allocate(size, dtype)
                    if block:
                        return block
        
        return None
    
    def _lru_scheduling(self, size: int, dtype: str, device: str) -> Optional[MemoryBlock]:
        """LRU调度"""
        # 找到最不常用的内存池
        best_pool = None
        best_priority = float('inf')
        
        for pool in self.memory_pools.values():
            if pool.allocated_memory + size <= pool.max_memory_bytes:
                # 计算池的优先级（基于平均访问频率）
                if pool.allocated_blocks:
                    avg_priority = sum(block.get_priority() for block in pool.allocated_blocks.values()) / len(pool.allocated_blocks)
                else:
                    avg_priority = 0
                
                if avg_priority < best_priority:
                    best_priority = avg_priority
                    best_pool = pool
        
        if best_pool:
            return best_pool.allocate(size, dtype)
        
        return None
    
    def _fifo_scheduling(self, size: int, dtype: str, device: str) -> Optional[MemoryBlock]:
        """FIFO调度"""
        # 简单的FIFO策略
        for pool in self.memory_pools.values():
            if pool.allocated_memory + size <= pool.max_memory_bytes:
                return pool.allocate(size, dtype)
        
        return None
    
    def optimize_memory_usage(self):
        """优化内存使用"""
        for pool in self.memory_pools.values():
            stats = pool.get_memory_stats()
            
            # 如果碎片率过高，进行整理
            if stats['fragmentation_ratio'] > pool.fragmentation_threshold:
                pool.defragment()
            
            # 如果内存使用率过高，主动释放一些内存
            if stats['memory_usage'] > 0.9:
                pool._free_memory(int(pool.max_memory_bytes * 0.1))


class GarbageCollector:
    """垃圾回收器"""
    
    def __init__(self):
        self.weak_references = {}  # object_id -> weakref
        self.reference_counts = defaultdict(int)
        self.gc_thread = None
        self.is_running = False
        self.gc_interval = 30  # 30秒
        self.gc_threshold = 100  # 100个对象
        
    def add_reference(self, obj: Any, obj_id: int):
        """添加对象引用"""
        try:
            weak_ref = weakref.ref(obj)
            self.weak_references[obj_id] = weak_ref
            self.reference_counts[obj_id] += 1
        except:
            pass
    
    def remove_reference(self, obj_id: int):
        """移除对象引用"""
        if obj_id in self.reference_counts:
            self.reference_counts[obj_id] -= 1
            if self.reference_counts[obj_id] <= 0:
                del self.reference_counts[obj_id]
                if obj_id in self.weak_references:
                    del self.weak_references[obj_id]
    
    def collect_garbage(self) -> int:
        """收集垃圾"""
        collected_count = 0
        
        # 检查弱引用
        dead_objects = []
        for obj_id, weak_ref in self.weak_references.items():
            if weak_ref() is None:
                dead_objects.append(obj_id)
        
        # 清理死对象
        for obj_id in dead_objects:
            del self.weak_references[obj_id]
            if obj_id in self.reference_counts:
                del self.reference_counts[obj_id]
            collected_count += 1
        
        # Python垃圾回收
        if HAS_GC:
            collected_count += gc.collect()
        
        return collected_count
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """获取垃圾回收统计"""
        return {
            'tracked_objects': len(self.weak_references),
            'reference_counts': dict(self.reference_counts),
            'total_references': sum(self.reference_counts.values()),
            'gc_interval': self.gc_interval,
            'gc_threshold': self.gc_threshold
        }


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.monitoring_data = {
            'timestamps': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'allocation_count': deque(maxlen=1000),
            'deallocation_count': deque(maxlen=1000),
            'fragmentation_ratio': deque(maxlen=1000)
        }
        
        self.alerts = []
        self.alert_thresholds = {
            'memory_usage': 0.9,  # 90%
            'fragmentation_ratio': 0.5,  # 50%
            'allocation_failure_rate': 0.1  # 10%
        }
        
    def record_memory_event(self, event_type: str, data: Dict[str, Any]):
        """记录内存事件"""
        timestamp = time.time()
        
        if event_type == 'allocation':
            self.monitoring_data['timestamps'].append(timestamp)
            self.monitoring_data['memory_usage'].append(data.get('memory_usage', 0))
            self.monitoring_data['allocation_count'].append(1)
            self.monitoring_data['deallocation_count'].append(0)
            self.monitoring_data['fragmentation_ratio'].append(data.get('fragmentation_ratio', 0))
            
        elif event_type == 'deallocation':
            if self.monitoring_data['timestamps']:
                self.monitoring_data['timestamps'].append(timestamp)
                self.monitoring_data['memory_usage'].append(data.get('memory_usage', 0))
                self.monitoring_data['allocation_count'].append(0)
                self.monitoring_data['deallocation_count'].append(1)
                self.monitoring_data['fragmentation_ratio'].append(data.get('fragmentation_ratio', 0))
    
    def check_alerts(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警"""
        new_alerts = []
        
        # 内存使用率告警
        if stats.get('memory_usage', 0) > self.alert_thresholds['memory_usage']:
            new_alerts.append({
                'type': 'high_memory_usage',
                'severity': 'warning',
                'message': f'High memory usage: {stats["memory_usage"]:.2%}',
                'timestamp': time.time()
            })
        
        # 碎片率告警
        if stats.get('fragmentation_ratio', 0) > self.alert_thresholds['fragmentation_ratio']:
            new_alerts.append({
                'type': 'high_fragmentation',
                'severity': 'warning',
                'message': f'High fragmentation: {stats["fragmentation_ratio"]:.2%}',
                'timestamp': time.time()
            })
        
        self.alerts.extend(new_alerts)
        return new_alerts
    
    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存报告"""
        if not self.monitoring_data['timestamps']:
            return {'status': 'no_data'}
        
        return {
            'current_memory_usage': self.monitoring_data['memory_usage'][-1] if self.monitoring_data['memory_usage'] else 0,
            'peak_memory_usage': max(self.monitoring_data['memory_usage']) if self.monitoring_data['memory_usage'] else 0,
            'avg_memory_usage': sum(self.monitoring_data['memory_usage']) / len(self.monitoring_data['memory_usage']) if self.monitoring_data['memory_usage'] else 0,
            'total_allocations': sum(self.monitoring_data['allocation_count']),
            'total_deallocations': sum(self.monitoring_data['deallocation_count']),
            'current_fragmentation': self.monitoring_data['fragmentation_ratio'][-1] if self.monitoring_data['fragmentation_ratio'] else 0,
            'recent_alerts': self.alerts[-10:] if self.alerts else [],
            'monitoring_period': len(self.monitoring_data['timestamps'])
        }


class SmartMemoryManager:
    """智能内存管理器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化组件
        self.memory_pools = {}
        self.memory_scheduler = MemoryScheduler()
        self.garbage_collector = GarbageCollector()
        self.memory_monitor = MemoryMonitor()
        
        # 默认配置
        self.default_pool_size = self.config.get('default_pool_size_gb', 4.0)
        self.enable_auto_defrag = self.config.get('enable_auto_defrag', True)
        self.enable_auto_gc = self.config.get('enable_auto_gc', True)
        self.monitoring_interval = self.config.get('monitoring_interval', 10)
        
        # 初始化默认内存池
        self._initialize_default_pools()
        
        # 启动后台任务
        self._start_background_tasks()
        
    def _initialize_default_pools(self):
        """初始化默认内存池"""
        # CPU内存池
        cpu_pool = MemoryPool(self.default_pool_size, 'cpu')
        self.memory_pools['cpu'] = cpu_pool
        self.memory_scheduler.add_memory_pool(cpu_pool)
        
        # 如果有GPU，创建GPU内存池
        if HAS_TORCH and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_pool = MemoryPool(gpu_memory * 0.8, f'cuda:{i}')  # 使用80%的GPU内存
                self.memory_pools[f'cuda:{i}'] = gpu_pool
                self.memory_scheduler.add_memory_pool(gpu_pool)
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 启动内存优化线程
        def optimization_loop():
            while True:
                time.sleep(self.monitoring_interval)
                
                # 内存优化
                self.memory_scheduler.optimize_memory_usage()
                
                # 垃圾回收
                if self.enable_auto_gc:
                    collected = self.garbage_collector.collect_garbage()
                    if collected > 0:
                        logging.info(f"Garbage collected: {collected} objects")
                
                # 自动碎片整理
                if self.enable_auto_defrag:
                    for pool in self.memory_pools.values():
                        stats = pool.get_memory_stats()
                        if stats['fragmentation_ratio'] > 0.3:
                            pool.defragment()
                            logging.info(f"Memory defragmented on {pool.device}")
        
        import threading
        optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        optimization_thread.start()
    
    def allocate_memory(self, size: int, dtype: str = 'float32', device: str = 'cpu') -> Optional[MemoryBlock]:
        """分配内存"""
        # 记录分配事件
        block = self.memory_scheduler.request_allocation(size, dtype, device)
        
        if block:
            self.memory_monitor.record_memory_event('allocation', {
                'memory_usage': self.get_total_memory_usage(),
                'fragmentation_ratio': self.get_average_fragmentation()
            })
            
            # 添加垃圾回收引用
            self.garbage_collector.add_reference(block, block.id)
        
        return block
    
    def deallocate_memory(self, block: MemoryBlock):
        """释放内存"""
        if block:
            # 移除垃圾回收引用
            self.garbage_collector.remove_reference(block.id)
            
            # 释放内存
            self.memory_pools[block.device].deallocate(block.id)
            
            # 记录释放事件
            self.memory_monitor.record_memory_event('deallocation', {
                'memory_usage': self.get_total_memory_usage(),
                'fragmentation_ratio': self.get_average_fragmentation()
            })
    
    def get_total_memory_usage(self) -> float:
        """获取总内存使用率"""
        total_allocated = sum(pool.allocated_memory for pool in self.memory_pools.values())
        total_max = sum(pool.max_memory_bytes for pool in self.memory_pools.values())
        
        return total_allocated / total_max if total_max > 0 else 0.0
    
    def get_average_fragmentation(self) -> float:
        """获取平均碎片率"""
        if not self.memory_pools:
            return 0.0
        
        total_fragmentation = sum(pool.get_memory_stats()['fragmentation_ratio'] for pool in self.memory_pools.values())
        return total_fragmentation / len(self.memory_pools)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """获取内存状态"""
        status = {
            'total_memory_usage': self.get_total_memory_usage(),
            'average_fragmentation': self.get_average_fragmentation(),
            'memory_pools': {},
            'garbage_collector_stats': self.garbage_collector.get_gc_stats(),
            'memory_report': self.memory_monitor.get_memory_report()
        }
        
        # 添加各个内存池的状态
        for device, pool in self.memory_pools.items():
            status['memory_pools'][device] = pool.get_memory_stats()
        
        return status
    
    def optimize_memory(self):
        """优化内存"""
        # 强制垃圾回收
        collected = self.garbage_collector.collect_garbage()
        
        # 内存碎片整理
        for pool in self.memory_pools.values():
            pool.defragment()
        
        # 内存调度优化
        self.memory_scheduler.optimize_memory_usage()
        
        return {
            'garbage_collected': collected,
            'pools_defragmented': len(self.memory_pools),
            'optimization_completed': True
        }


__all__ = [
    'MemoryBlock', 'MemoryPool', 'MemoryScheduler', 'GarbageCollector',
    'MemoryMonitor', 'SmartMemoryManager'
]