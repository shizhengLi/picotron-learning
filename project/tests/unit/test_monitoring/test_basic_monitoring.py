"""
基础监控系统单元测试
=================

测试监控系统的各个组件，包括指标收集、异常检测、告警管理和可视化功能。
"""

import sys
import os
import unittest
import time
import threading
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from monitoring.basic_monitoring import (
        MetricsCollector, AnomalyDetector, AlertManager, BasicVisualization,
        BasicMonitoringSystem, MetricType, AlertLevel, MetricData, Alert
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestMetricData(unittest.TestCase):
    """测试指标数据类"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
    
    def test_metric_data_creation(self):
        """测试指标数据创建"""
        metric = MetricData(
            name="test_metric",
            value=42.0,
            timestamp=time.time(),
            tags={"env": "test"},
            metric_type=MetricType.GAUGE
        )
        
        self.assertEqual(metric.name, "test_metric")
        self.assertEqual(metric.value, 42.0)
        self.assertEqual(metric.metric_type, MetricType.GAUGE)
        self.assertEqual(metric.tags["env"], "test")
    
    def test_metric_data_default_values(self):
        """测试指标数据默认值"""
        metric = MetricData("test", 1.0)
        
        self.assertEqual(metric.name, "test")
        self.assertEqual(metric.value, 1.0)
        self.assertEqual(metric.metric_type, MetricType.GAUGE)
        self.assertEqual(metric.tags, {})


class TestMetricsCollector(unittest.TestCase):
    """测试指标收集器"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.collector = MetricsCollector(max_history_size=10)
    
    def test_add_metric(self):
        """测试添加指标"""
        self.collector.add_metric("test_metric", 10.5)
        
        history = self.collector.get_metric("test_metric")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].value, 10.5)
        self.assertEqual(history[0].name, "test_metric")
    
    def test_add_metric_with_tags(self):
        """测试添加带标签的指标"""
        tags = {"host": "server1", "env": "prod"}
        self.collector.add_metric("cpu_usage", 75.5, tags=tags)
        
        history = self.collector.get_metric("cpu_usage")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].tags, tags)
    
    def test_get_metric_with_time_filter(self):
        """测试按时间过滤获取指标"""
        now = time.time()
        
        # 添加过去的数据
        self.collector.add_metric("old_metric", 1.0)
        time.sleep(0.1)
        
        # 添加现在的数据
        self.collector.add_metric("new_metric", 2.0)
        recent_time = time.time() - 0.05
        
        # 只获取最近的数据
        recent_history = self.collector.get_metric("new_metric", recent_time)
        self.assertEqual(len(recent_history), 1)
        self.assertEqual(recent_history[0].value, 2.0)
    
    def test_get_latest_metric(self):
        """测试获取最新指标"""
        self.collector.add_metric("test", 1.0)
        time.sleep(0.01)
        self.collector.add_metric("test", 2.0)
        
        latest = self.collector.get_latest_metric("test")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.value, 2.0)
    
    def test_get_latest_metric_nonexistent(self):
        """测试获取不存在的最新指标"""
        latest = self.collector.get_latest_metric("nonexistent")
        self.assertIsNone(latest)
    
    def test_history_size_limit(self):
        """测试历史记录大小限制"""
        # 添加超过限制的数据
        for i in range(15):
            self.collector.add_metric("limited_metric", float(i))
        
        history = self.collector.get_metric("limited_metric")
        self.assertEqual(len(history), 10)  # 应该限制为10
        self.assertEqual(history[0].value, 5.0)  # 最早的应该是5
        self.assertEqual(history[-1].value, 14.0)  # 最新的应该是14
    
    def test_metrics_summary(self):
        """测试指标摘要"""
        self.collector.add_metric("metric1", 10.0)
        self.collector.add_metric("metric2", 20.0)
        
        summary = self.collector.get_metrics_summary()
        
        self.assertIn("metric1", summary)
        self.assertIn("metric2", summary)
        self.assertEqual(summary["metric1"]["latest_value"], 10.0)
        self.assertEqual(summary["metric2"]["latest_value"], 20.0)
    
    def test_export_metrics_json(self):
        """测试导出JSON格式指标"""
        self.collector.add_metric("test_metric", 42.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.collector.export_metrics(temp_file, 'json')
            
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            self.assertIn("test_metric", data)
            self.assertEqual(len(data["test_metric"]), 1)
            self.assertEqual(data["test_metric"][0]["value"], 42.0)
        
        finally:
            os.unlink(temp_file)
    
    def test_start_stop_collection(self):
        """测试启动和停止收集"""
        # 启动收集
        self.collector.start_collection(0.1)
        self.assertTrue(self.collector.is_collecting)
        
        # 等待一些收集
        time.sleep(0.5)
        
        # 停止收集
        self.collector.stop_collection()
        self.assertFalse(self.collector.is_collecting)
    
    def test_custom_collector(self):
        """测试自定义收集器"""
        def custom_collector(collector):
            collector.add_metric("custom", 123.0)
        
        self.collector.add_custom_collector("test_custom", custom_collector)
        
        # 启动收集以触发自定义收集器
        self.collector.start_collection(0.1)
        time.sleep(0.2)
        self.collector.stop_collection()
        
        # 检查自定义指标是否被收集
        custom_metric = self.collector.get_latest_metric("custom")
        self.assertIsNotNone(custom_metric)
        self.assertEqual(custom_metric.value, 123.0)
    
    def test_collect_system_metrics(self):
        """测试收集系统指标"""
        # 这个测试主要确保方法不会抛出异常
        self.collector.collect_system_metrics()
        
        # 检查是否有系统指标被收集（如果有psutil）
        cpu_metric = self.collector.get_latest_metric("system_cpu_usage")
        if cpu_metric is not None:
            self.assertIsInstance(cpu_metric.value, (int, float))
            self.assertGreaterEqual(cpu_metric.value, 0)
            self.assertLessEqual(cpu_metric.value, 100)
    
    def test_collect_training_metrics(self):
        """测试收集训练指标"""
        # 创建模拟的模型和优化器
        mock_model = Mock()
        mock_optimizer = Mock()
        
        # 测试无PyTorch的情况
        self.collector.collect_training_metrics(
            mock_model, mock_optimizer, 0.5, 32, 0.001
        )
        
        # 检查指标是否被添加
        loss_metric = self.collector.get_latest_metric("training_loss")
        lr_metric = self.collector.get_latest_metric("training_learning_rate")
        
        self.assertIsNotNone(loss_metric)
        self.assertEqual(loss_metric.value, 0.5)
        self.assertIsNotNone(lr_metric)
        self.assertEqual(lr_metric.value, 0.001)


class TestAnomalyDetector(unittest.TestCase):
    """测试异常检测器"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.detector = AnomalyDetector(window_size=5, sensitivity=2.0)
        self.collector = MetricsCollector()
    
    def test_z_score_detection(self):
        """测试Z-Score异常检测"""
        # 添加正常数据
        for i in range(10):
            self.collector.add_metric("test", 10.0 + i * 0.1)
        
        # 添加异常数据
        self.collector.add_metric("test", 100.0)  # 明显异常
        
        anomalies = self.detector.detect_anomalies(self.collector, ["test"])
        
        self.assertGreater(len(anomalies), 0)
        self.assertEqual(anomalies[0]["metric_name"], "test")
        self.assertEqual(anomalies[0]["algorithm"], "z_score")
    
    def test_iqr_detection(self):
        """测试IQR异常检测"""
        # 添加数据，包含异常值
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100是异常值
        for val in values:
            self.collector.add_metric("iqr_test", val)
        
        anomalies = self.detector.detect_anomalies(self.collector, ["iqr_test"])
        
        # 应该检测到异常
        iqr_anomalies = [a for a in anomalies if a["algorithm"] == "iqr"]
        self.assertGreater(len(iqr_anomalies), 0)
    
    def test_moving_average_detection(self):
        """测试移动平均异常检测"""
        # 添加稳定数据，然后突然变化
        for i in range(20):
            value = 10.0 if i < 15 else 50.0  # 突然从10变成50
            self.collector.add_metric("ma_test", value)
        
        anomalies = self.detector.detect_anomalies(self.collector, ["ma_test"])
        
        ma_anomalies = [a for a in anomalies if a["algorithm"] == "moving_average"]
        self.assertGreater(len(ma_anomalies), 0)
    
    def test_insufficient_data(self):
        """测试数据不足的情况"""
        # 只添加少量数据
        for i in range(3):
            self.collector.add_metric("small_test", i)
        
        anomalies = self.detector.detect_anomalies(self.collector, ["small_test"])
        
        # 应该没有异常检测（数据不足）
        self.assertEqual(len(anomalies), 0)
    
    def test_update_baseline(self):
        """测试更新基线"""
        # 添加一些数据
        for i in range(20):
            self.collector.add_metric("baseline_test", 10.0 + i * 0.5)
        
        # 更新基线
        self.detector.update_baseline(self.collector, ["baseline_test"])
        
        # 检查基线是否被更新
        self.assertIn("baseline_test", self.detector.baseline_stats)
        baseline = self.detector.baseline_stats["baseline_test"]
        
        self.assertIn("mean", baseline)
        self.assertIn("std", baseline)
        self.assertIn("min", baseline)
        self.assertIn("max", baseline)
    
    def test_anomaly_summary(self):
        """测试异常摘要"""
        # 添加一些异常
        for i in range(10):
            self.collector.add_metric("summary_test", 10.0)
        self.collector.add_metric("summary_test", 100.0)  # 异常值
        
        # 检测异常
        self.detector.detect_anomalies(self.collector, ["summary_test"])
        
        # 获取摘要
        summary = self.detector.get_anomaly_summary()
        
        self.assertIn("total_anomalies", summary)
        self.assertIn("recent_anomalies", summary)
        self.assertIn("baseline_stats", summary)
        self.assertGreater(summary["total_anomalies"], 0)


class TestAlertManager(unittest.TestCase):
    """测试告警管理器"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.alert_manager = AlertManager()
        self.collector = MetricsCollector()
    
    def test_add_alert_rule(self):
        """测试添加告警规则"""
        rule = {
            'metric_name': 'test_metric',
            'condition': 'greater_than',
            'threshold': 80.0,
            'level': AlertLevel.WARNING,
            'message': '测试告警: {value}'
        }
        
        self.alert_manager.add_alert_rule("test_rule", rule)
        
        self.assertIn("test_rule", self.alert_manager.alert_rules)
        self.assertEqual(self.alert_manager.alert_rules["test_rule"]["threshold"], 80.0)
    
    def test_add_alert_rule_missing_fields(self):
        """测试添加缺少字段的告警规则"""
        rule = {
            'metric_name': 'test_metric',
            # 缺少condition字段
            'threshold': 80.0
        }
        
        with self.assertRaises(ValueError):
            self.alert_manager.add_alert_rule("invalid_rule", rule)
    
    def test_remove_alert_rule(self):
        """测试移除告警规则"""
        rule = {
            'metric_name': 'test_metric',
            'condition': 'greater_than',
            'threshold': 80.0,
            'level': AlertLevel.WARNING,
            'message': '测试告警: {value}'
        }
        
        self.alert_manager.add_alert_rule("test_rule", rule)
        self.assertIn("test_rule", self.alert_manager.alert_rules)
        
        self.alert_manager.remove_alert_rule("test_rule")
        self.assertNotIn("test_rule", self.alert_manager.alert_rules)
    
    def test_check_alerts_triggered(self):
        """测试检查告警触发"""
        # 添加告警规则
        rule = {
            'metric_name': 'test_metric',
            'condition': 'greater_than',
            'threshold': 50.0,
            'level': AlertLevel.WARNING,
            'message': '值过高: {value}'
        }
        self.alert_manager.add_alert_rule("high_value", rule)
        
        # 添加触发告警的指标
        self.collector.add_metric("test_metric", 75.0)
        
        # 检查告警
        alerts = self.alert_manager.check_alerts(self.collector)
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].name, "high_value")
        self.assertEqual(alerts[0].level, AlertLevel.WARNING)
        self.assertIn("75.0", alerts[0].message)
    
    def test_check_alerts_not_triggered(self):
        """测试检查告警未触发"""
        # 添加告警规则
        rule = {
            'metric_name': 'test_metric',
            'condition': 'greater_than',
            'threshold': 50.0,
            'level': AlertLevel.WARNING,
            'message': '值过高: {value}'
        }
        self.alert_manager.add_alert_rule("high_value", rule)
        
        # 添加未触发告警的指标
        self.collector.add_metric("test_metric", 25.0)
        
        # 检查告警
        alerts = self.alert_manager.check_alerts(self.collector)
        
        self.assertEqual(len(alerts), 0)
    
    def test_spike_detection(self):
        """测试突变检测"""
        # 添加突变检测规则
        rule = {
            'metric_name': 'spike_metric',
            'condition': 'spike',
            'threshold': 2.0,  # 2倍变化
            'level': AlertLevel.INFO,
            'message': '检测到突变: {value}'
        }
        self.alert_manager.add_alert_rule("spike_rule", rule)
        
        # 添加稳定数据，然后突变
        self.collector.add_metric("spike_metric", 10.0)
        time.sleep(0.01)
        self.collector.add_metric("spike_metric", 25.0)  # 2.5倍变化
        
        # 检查告警
        alerts = self.alert_manager.check_alerts(self.collector)
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].name, "spike_rule")
    
    def test_get_active_alerts(self):
        """测试获取活动告警"""
        # 添加告警规则
        rule = {
            'metric_name': 'test_metric',
            'condition': 'greater_than',
            'threshold': 50.0,
            'level': AlertLevel.WARNING,
            'message': '值过高: {value}'
        }
        self.alert_manager.add_alert_rule("test_rule", rule)
        
        # 触发告警
        self.collector.add_metric("test_metric", 75.0)
        self.alert_manager.check_alerts(self.collector)
        
        # 获取活动告警
        active_alerts = self.alert_manager.get_active_alerts()
        
        self.assertEqual(len(active_alerts), 1)
        self.assertFalse(active_alerts[0].resolved)
    
    def test_resolve_alert(self):
        """测试解决告警"""
        # 添加告警规则
        rule = {
            'metric_name': 'test_metric',
            'condition': 'greater_than',
            'threshold': 50.0,
            'level': AlertLevel.WARNING,
            'message': '值过高: {value}'
        }
        self.alert_manager.add_alert_rule("test_rule", rule)
        
        # 触发告警
        self.collector.add_metric("test_metric", 75.0)
        alerts = self.alert_manager.check_alerts(self.collector)
        
        # 解决告警
        alert_id = alerts[0].id
        self.alert_manager.resolve_alert(alert_id)
        
        # 检查告警状态
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 0)
        
        # 检查告警历史
        alert_history = self.alert_manager.get_alert_history()
        resolved_alert = next((a for a in alert_history if a.id == alert_id), None)
        self.assertIsNotNone(resolved_alert)
        self.assertTrue(resolved_alert.resolved)
    
    def test_alert_summary(self):
        """测试告警摘要"""
        # 添加多个告警规则
        rules = [
            {
                'metric_name': 'cpu_usage',
                'condition': 'greater_than',
                'threshold': 80.0,
                'level': AlertLevel.WARNING,
                'message': 'CPU使用率过高: {value}'
            },
            {
                'metric_name': 'memory_usage',
                'condition': 'greater_than',
                'threshold': 90.0,
                'level': AlertLevel.ERROR,
                'message': '内存使用率过高: {value}'
            }
        ]
        
        for i, rule in enumerate(rules):
            self.alert_manager.add_alert_rule(f"rule_{i}", rule)
        
        # 触发告警
        self.collector.add_metric("cpu_usage", 85.0)
        self.collector.add_metric("memory_usage", 95.0)
        self.alert_manager.check_alerts(self.collector)
        
        # 获取摘要
        summary = self.alert_manager.get_alert_summary()
        
        self.assertIn("total_active_alerts", summary)
        self.assertIn("alerts_by_level", summary)
        self.assertIn("alerts_by_rule", summary)
        self.assertEqual(summary["total_active_alerts"], 2)
        self.assertIn("warning", summary["alerts_by_level"])
        self.assertIn("error", summary["alerts_by_level"])
    
    def test_notification_handler(self):
        """测试通知处理器"""
        received_alerts = []
        
        def test_handler(alert):
            received_alerts.append(alert)
        
        # 添加通知处理器
        self.alert_manager.add_notification_handler("test", test_handler)
        
        # 添加告警规则
        rule = {
            'metric_name': 'test_metric',
            'condition': 'greater_than',
            'threshold': 50.0,
            'level': AlertLevel.WARNING,
            'message': '测试告警: {value}'
        }
        self.alert_manager.add_alert_rule("test_rule", rule)
        
        # 触发告警
        self.collector.add_metric("test_metric", 75.0)
        self.alert_manager.check_alerts(self.collector)
        
        # 检查是否收到通知
        self.assertEqual(len(received_alerts), 1)
        self.assertEqual(received_alerts[0].name, "test_rule")


class TestBasicVisualization(unittest.TestCase):
    """测试基础可视化"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.detector = AnomalyDetector()
        self.visualization = BasicVisualization(self.collector, self.alert_manager, self.detector)
    
    def test_console_dashboard(self):
        """测试控制台仪表板"""
        # 添加一些测试数据
        self.collector.add_metric("system_cpu_usage", 45.5)
        self.collector.add_metric("system_memory_usage", 65.2)
        self.collector.add_metric("training_loss", 0.5)
        
        # 启动控制台仪表板（短暂运行）
        def run_dashboard():
            self.visualization.start_console_dashboard(0.1)
        
        dashboard_thread = threading.Thread(target=run_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # 等待一小段时间
        time.sleep(0.5)
        
        # 停止仪表板
        self.visualization.stop_dashboard()
    
    def test_generate_report(self):
        """测试生成报告"""
        # 添加一些测试数据
        for i in range(10):
            self.collector.add_metric("test_metric", float(i))
        
        # 生成报告
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.visualization.generate_report(temp_file)
            
            # 检查文件是否创建
            self.assertTrue(os.path.exists(temp_file))
            
            # 检查报告内容
            with open(temp_file, 'r') as f:
                report = json.load(f)
            
            self.assertIn("report_time", report)
            self.assertIn("metrics_summary", report)
            self.assertIn("alert_summary", report)
            self.assertIn("anomaly_summary", report)
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_system_info(self):
        """测试系统信息获取"""
        info = self.visualization._get_system_info()
        
        self.assertIn("platform", info)
        self.assertIn("python_version", info)
        self.assertIn("timestamp", info)


class TestBasicMonitoringSystem(unittest.TestCase):
    """测试基础监控系统"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.system = BasicMonitoringSystem({
            'max_history_size': 100,
            'anomaly_window_size': 10,
            'anomaly_sensitivity': 2.0,
            'monitoring_interval': 0.1
        })
    
    def tearDown(self):
        if hasattr(self, 'system') and self.system.is_monitoring:
            self.system.stop_monitoring()
    
    def test_system_initialization(self):
        """测试系统初始化"""
        self.assertIsNotNone(self.system.metrics_collector)
        self.assertIsNotNone(self.system.anomaly_detector)
        self.assertIsNotNone(self.system.alert_manager)
        self.assertIsNotNone(self.system.visualization)
        self.assertFalse(self.system.is_monitoring)
    
    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        # 启动监控
        self.system.start_monitoring()
        self.assertTrue(self.system.is_monitoring)
        
        # 等待一小段时间
        time.sleep(0.5)
        
        # 停止监控
        self.system.stop_monitoring()
        self.assertFalse(self.system.is_monitoring)
    
    def test_add_custom_alert_rule(self):
        """测试添加自定义告警规则"""
        rule = {
            'metric_name': 'custom_metric',
            'condition': 'greater_than',
            'threshold': 100.0,
            'level': AlertLevel.WARNING,
            'message': '自定义告警: {value}'
        }
        
        self.system.add_custom_alert_rule("custom_rule", rule)
        
        self.assertIn("custom_rule", self.system.alert_manager.alert_rules)
    
    def test_add_custom_metric_collector(self):
        """测试添加自定义指标收集器"""
        def custom_collector(collector):
            collector.add_metric("custom_test", 42.0)
        
        self.system.add_custom_metric_collector("test_custom", custom_collector)
        
        # 启动监控以触发自定义收集器
        self.system.start_monitoring()
        time.sleep(0.5)
        self.system.stop_monitoring()
        
        # 检查自定义指标是否被收集
        custom_metric = self.system.metrics_collector.get_latest_metric("custom_test")
        self.assertIsNotNone(custom_metric)
        self.assertEqual(custom_metric.value, 42.0)
    
    def test_get_system_status(self):
        """测试获取系统状态"""
        # 添加一些测试数据
        self.system.metrics_collector.add_metric("test_status", 1.0)
        
        status = self.system.get_system_status()
        
        self.assertIn("is_monitoring", status)
        self.assertIn("metrics_summary", status)
        self.assertIn("alert_summary", status)
        self.assertIn("anomaly_summary", status)
        self.assertIn("timestamp", status)
        self.assertIn("test_status", status["metrics_summary"])
    
    def test_export_data(self):
        """测试导出数据"""
        # 添加一些测试数据
        self.system.metrics_collector.add_metric("export_test", 123.0)
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 导出数据
            self.system.export_data(temp_dir)
            
            # 检查文件是否创建
            metrics_file = os.path.join(temp_dir, "metrics.json")
            report_file = os.path.join(temp_dir, "monitoring_report.json")
            
            self.assertTrue(os.path.exists(metrics_file))
            self.assertTrue(os.path.exists(report_file))
            
            # 检查指标文件内容
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            self.assertIn("export_test", metrics_data)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.system = BasicMonitoringSystem({
            'monitoring_interval': 0.1
        })
    
    def tearDown(self):
        if hasattr(self, 'system') and self.system.is_monitoring:
            self.system.stop_monitoring()
    
    def test_full_monitoring_workflow(self):
        """测试完整的监控工作流"""
        # 启动监控
        self.system.start_monitoring()
        
        # 模拟训练过程
        for i in range(10):
            # 添加训练指标
            loss = 1.0 / (i + 1)  # 损失递减
            self.system.metrics_collector.collect_training_metrics(
                Mock(), Mock(), loss, 32, 0.001
            )
            
            # 添加系统指标
            self.system.metrics_collector.add_metric("system_cpu_usage", 50.0 + i * 2)
            self.system.metrics_collector.add_metric("system_memory_usage", 60.0 + i)
            
            time.sleep(0.1)
        
        # 等待监控处理
        time.sleep(0.5)
        
        # 检查系统状态
        status = self.system.get_system_status()
        
        self.assertTrue(status["is_monitoring"])
        self.assertIn("training_loss", status["metrics_summary"])
        self.assertIn("system_cpu_usage", status["metrics_summary"])
        self.assertIn("system_memory_usage", status["metrics_summary"])
        
        # 停止监控
        self.system.stop_monitoring()
    
    def test_alert_workflow(self):
        """测试告警工作流"""
        # 添加告警规则
        rule = {
            'metric_name': 'high_cpu',
            'condition': 'greater_than',
            'threshold': 80.0,
            'level': AlertLevel.WARNING,
            'message': 'CPU使用率过高: {value}%'
        }
        self.system.add_custom_alert_rule("cpu_alert", rule)
        
        # 启动监控
        self.system.start_monitoring()
        
        # 触发告警
        self.system.metrics_collector.add_metric("high_cpu", 85.0)
        
        # 等待告警处理
        time.sleep(0.5)
        
        # 检查告警状态
        status = self.system.get_system_status()
        alert_summary = status["alert_summary"]
        
        self.assertGreater(alert_summary["total_active_alerts"], 0)
        
        # 解决告警（添加正常值）
        self.system.metrics_collector.add_metric("high_cpu", 40.0)
        time.sleep(0.5)
        
        # 检查告警是否解决
        active_alerts = self.system.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 0)
        
        # 停止监控
        self.system.stop_monitoring()
    
    def test_anomaly_detection_workflow(self):
        """测试异常检测工作流"""
        # 启动监控
        self.system.start_monitoring()
        
        # 添加正常数据
        for i in range(20):
            self.system.metrics_collector.add_metric("normal_metric", 10.0 + i * 0.1)
        
        # 添加异常数据
        self.system.metrics_collector.add_metric("normal_metric", 100.0)
        
        # 等待异常检测
        time.sleep(0.5)
        
        # 检查异常检测结果
        status = self.system.get_system_status()
        anomaly_summary = status["anomaly_summary"]
        
        self.assertGreater(anomaly_summary["total_anomalies"], 0)
        
        # 停止监控
        self.system.stop_monitoring()


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)