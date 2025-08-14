#!/usr/bin/env python3
"""
基础监控系统简单演示
================
"""

import sys
import os
import time

# 添加源代码路径
sys.path.insert(0, 'src')

try:
    from monitoring.basic_monitoring import MetricsCollector, AlertManager, AnomalyDetector
    print("✅ 基础监控系统导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def test_metrics_collector():
    """测试指标收集器"""
    print("\n📊 测试指标收集器")
    print("-" * 30)
    
    collector = MetricsCollector()
    
    # 添加指标
    collector.add_metric("test_metric", 42.0)
    collector.add_metric("cpu_usage", 75.5)
    collector.add_metric("memory_usage", 60.2)
    
    # 获取指标
    latest = collector.get_latest_metric("test_metric")
    print(f"最新指标: {latest.value if latest else 'None'}")
    
    # 获取摘要
    summary = collector.get_metrics_summary()
    print(f"指标数量: {len(summary)}")
    for name, info in summary.items():
        print(f"  {name}: {info['latest_value']:.1f}")
    
    return collector


def test_alert_manager():
    """测试告警管理器"""
    print("\n🚨 测试告警管理器")
    print("-" * 30)
    
    alert_manager = AlertManager()
    collector = MetricsCollector()
    
    # 添加告警规则
    rule = {
        'metric_name': 'cpu_usage',
        'condition': 'greater_than',
        'threshold': 70.0,
        'level': 'warning',
        'message': 'CPU使用率过高: {value}%'
    }
    alert_manager.add_alert_rule("high_cpu", rule)
    
    # 添加正常指标
    collector.add_metric("cpu_usage", 50.0)
    alerts = alert_manager.check_alerts(collector)
    print(f"正常情况告警数: {len(alerts)}")
    
    # 添加告警指标
    collector.add_metric("cpu_usage", 85.0)
    alerts = alert_manager.check_alerts(collector)
    print(f"告警情况告警数: {len(alerts)}")
    
    if alerts:
        print(f"告警消息: {alerts[0].message}")
    
    return alert_manager, collector


def test_anomaly_detector():
    """测试异常检测器"""
    print("\n🔍 测试异常检测器")
    print("-" * 30)
    
    detector = AnomalyDetector()
    collector = MetricsCollector()
    
    # 添加正常数据
    for i in range(10):
        collector.add_metric("normal_data", 10.0 + i * 0.1)
    
    # 添加异常数据
    collector.add_metric("normal_data", 100.0)
    
    # 检测异常
    anomalies = detector.detect_anomalies(collector)
    print(f"检测到异常数: {len(anomalies)}")
    
    if anomalies:
        for anomaly in anomalies[:3]:  # 显示前3个
            print(f"  异常值: {anomaly['value']:.1f} (算法: {anomaly['algorithm']})")
    
    return detector


def main():
    """主函数"""
    print("🔧 基础监控系统简单测试")
    print("=" * 40)
    
    try:
        # 测试指标收集器
        collector = test_metrics_collector()
        
        # 测试告警管理器
        alert_manager, alert_collector = test_alert_manager()
        
        # 测试异常检测器
        detector = test_anomaly_detector()
        
        print("\n✅ 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()