#!/usr/bin/env python3
"""
智能内存管理模块简单演示
======================

快速演示内存管理模块的核心功能
"""

import sys
import os
import time

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from optimization.memory_management import SmartMemoryManager, MemoryPool


def quick_demo():
    """快速演示"""
    print("智能内存管理模块演示")
    print("=" * 40)
    
    # 创建内存管理器
    print("1. 创建内存管理器...")
    manager = SmartMemoryManager({
        'default_pool_size_gb': 0.1,  # 100MB
        'enable_auto_defrag': True,
        'enable_auto_gc': True
    })
    
    # 分配内存
    print("2. 分配内存块...")
    blocks = []
    for i in range(3):
        block = manager.allocate_memory(1024 * 1024, 'float32', 'cpu')  # 1MB
        if block:
            blocks.append(block)
            print(f"   分配块 {i+1}: {block.size/1024/1024:.1f}MB")
    
    # 显示状态
    print("3. 内存状态:")
    status = manager.get_memory_status()
    print(f"   总内存使用率: {status['total_memory_usage']:.2%}")
    print(f"   平均碎片率: {status['average_fragmentation']:.2%}")
    
    # 释放内存
    print("4. 释放内存...")
    for i, block in enumerate(blocks):
        manager.deallocate_memory(block)
        print(f"   释放块 {i+1}")
    
    # 最终状态
    print("5. 最终状态:")
    status = manager.get_memory_status()
    print(f"   总内存使用率: {status['total_memory_usage']:.2%}")
    print(f"   平均碎片率: {status['average_fragmentation']:.2%}")
    
    print("\n演示完成!")


def memory_pool_demo():
    """内存池演示"""
    print("\n内存池演示")
    print("=" * 40)
    
    # 创建内存池
    pool = MemoryPool(max_memory_gb=0.01, device='cpu')  # 10MB
    print(f"创建内存池: {pool.max_memory_bytes/1024/1024:.1f}MB")
    
    # 分配内存
    print("分配内存块...")
    blocks = []
    for i in range(5):
        block = pool.allocate(1024 * 512, 'float32')  # 512KB
        if block:
            blocks.append(block)
            print(f"   分配块 {i+1}: {block.size/1024:.1f}KB")
        else:
            print(f"   分配块 {i+1}: 失败")
    
    # 显示统计
    print("内存池统计:")
    stats = pool.get_memory_stats()
    print(f"   已分配块: {stats['allocated_blocks']}")
    print(f"   内存使用: {stats['allocated_memory']/1024:.1f}KB")
    print(f"   使用率: {stats['memory_usage']:.2%}")
    print(f"   碎片率: {stats['fragmentation_ratio']:.2%}")
    
    # 释放内存
    print("释放内存...")
    for block in blocks:
        pool.deallocate(block.id)
    
    print("内存池演示完成!")


if __name__ == "__main__":
    try:
        quick_demo()
        memory_pool_demo()
    except Exception as e:
        print(f"演示出错: {str(e)}")
        import traceback
        traceback.print_exc()