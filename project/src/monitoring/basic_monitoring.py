"""
基础监控系统
===========

实现完整的监控系统，包括指标收集、异常检测、告警管理和可视化界面。
"""

try:
    import torch
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
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import threading
    import time
    import queue
    import json
    import logging
    from typing import Dict, List, Any, Optional, Tuple, Union
    from dataclasses import dataclass, field
    from enum import Enum
    import weakref
    import traceback
    import os
    import sys
except ImportError as e:
    print(f"Missing required dependency: {e}")
    sys.exit(1)


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"      # 计数器
    GAUGE = "gauge"          # 仪表盘
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"      # 摘要


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """指标数据"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """告警信息"""
    id: str
    name: str
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    current_value: float
    threshold: float
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history_size: int = 10000):
        self.metrics_history: Dict[str, List[MetricData]] = {}
        self.max_history_size = max_history_size
        self.lock = threading.Lock()
        self.is_collecting = False
        self.collection_thread = None
        self.collection_interval = 1.0  # 1秒
        self.custom_collectors = {}
        
        # 初始化系统指标
        self._init_system_metrics()
    
    def _init_system_metrics(self):
        """初始化系统指标"""
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'gpu_memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
            'process_count': 0
        }
    
    def add_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                   tags: Optional[Dict[str, str]] = None):
        """添加指标"""
        metric = MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metric_type=metric_type
        )
        
        with self.lock:
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            
            self.metrics_history[name].append(metric)
            
            # 限制历史数据大小
            if len(self.metrics_history[name]) > self.max_history_size:
                self.metrics_history[name] = self.metrics_history[name][-self.max_history_size:]
    
    def get_metric(self, name: str, since: Optional[float] = None) -> List[MetricData]:
        """获取指标历史数据"""
        with self.lock:
            if name not in self.metrics_history:
                return []
            
            history = self.metrics_history[name]
            if since is None:
                return history
            
            return [m for m in history if m.timestamp >= since]
    
    def get_latest_metric(self, name: str) -> Optional[MetricData]:
        """获取最新指标值"""
        history = self.get_metric(name)
        return history[-1] if history else None
    
    def collect_system_metrics(self):
        """收集系统指标"""
        if not HAS_PSUTIL:
            return
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.add_metric('system_cpu_usage', cpu_percent, MetricType.GAUGE)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self.add_metric('system_memory_usage', memory.percent, MetricType.GAUGE)
            self.add_metric('system_memory_used', memory.used, MetricType.GAUGE)
            self.add_metric('system_memory_total', memory.total, MetricType.GAUGE)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            self.add_metric('system_disk_usage', disk.percent, MetricType.GAUGE)
            
            # 网络IO
            net_io = psutil.net_io_counters()
            self.add_metric('system_network_bytes_sent', net_io.bytes_sent, MetricType.COUNTER)
            self.add_metric('system_network_bytes_recv', net_io.bytes_recv, MetricType.COUNTER)
            
            # 进程数
            process_count = len(psutil.pids())
            self.add_metric('system_process_count', process_count, MetricType.GAUGE)
            
            # GPU指标（如果有PyTorch）
            if HAS_TORCH and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_allocated(i)
                    try:
                        # 尝试获取总内存
                        if hasattr(torch.cuda, 'memory_properties'):
                            gpu_memory_total = torch.cuda.memory_properties(i).total_memory
                        else:
                            # 使用旧的API
                            gpu_memory_total = torch.cuda.get_device_properties(i).total_memory
                        gpu_usage = gpu_memory / gpu_memory_total * 100
                        
                        self.add_metric(f'gpu_{i}_memory_usage', gpu_usage, MetricType.GAUGE,
                                      tags={'device': f'cuda:{i}'})
                        self.add_metric(f'gpu_{i}_memory_allocated', gpu_memory, MetricType.GAUGE,
                                      tags={'device': f'cuda:{i}'})
                        self.add_metric(f'gpu_{i}_memory_total', gpu_memory_total, MetricType.GAUGE,
                                      tags={'device': f'cuda:{i}'})
                    except Exception as gpu_e:
                        logging.warning(f"Error collecting GPU metrics for device {i}: {gpu_e}")
        
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
    
    def collect_training_metrics(self, model, optimizer, loss, batch_size, 
                               learning_rate: Optional[float] = None):
        """收集训练指标"""
        if HAS_TORCH and hasattr(loss, 'item'):
            # 损失值
            self.add_metric('training_loss', loss.item(), MetricType.GAUGE)
            
            # 学习率
            if learning_rate is None and optimizer is not None:
                for param_group in optimizer.param_groups:
                    lr = param_group.get('lr', 0.0)
                    self.add_metric('training_learning_rate', lr, MetricType.GAUGE,
                                  tags={'param_group': param_group.get('name', 'default')})
            else:
                self.add_metric('training_learning_rate', learning_rate or 0.0, MetricType.GAUGE)
            
            # 梯度统计
            if optimizer is not None:
                total_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.add_metric('training_gradient_norm', total_norm, MetricType.GAUGE)
            
            # 参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.add_metric('model_total_parameters', total_params, MetricType.GAUGE)
            self.add_metric('model_trainable_parameters', trainable_params, MetricType.GAUGE)
            
            # 批次大小
            self.add_metric('training_batch_size', batch_size, MetricType.GAUGE)
        
        else:
            # 简化版本
            self.add_metric('training_loss', float(loss), MetricType.GAUGE)
            self.add_metric('training_learning_rate', learning_rate or 0.0, MetricType.GAUGE)
            self.add_metric('training_batch_size', batch_size, MetricType.GAUGE)
    
    def add_custom_collector(self, name: str, collector_func):
        """添加自定义收集器"""
        self.custom_collectors[name] = collector_func
    
    def start_collection(self, interval: float = 1.0):
        """开始收集指标"""
        self.collection_interval = interval
        self.is_collecting = True
        
        if self.collection_thread is None or not self.collection_thread.is_alive():
            self.collection_thread = threading.Thread(target=self._collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
    
    def stop_collection(self):
        """停止收集指标"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
    
    def _collection_loop(self):
        """收集循环"""
        while self.is_collecting:
            try:
                # 收集系统指标
                self.collect_system_metrics()
                
                # 收集自定义指标
                for name, collector in self.custom_collectors.items():
                    try:
                        collector(self)
                    except Exception as e:
                        logging.error(f"Error in custom collector {name}: {e}")
                
                time.sleep(self.collection_interval)
            
            except Exception as e:
                logging.error(f"Error in collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {}
        
        with self.lock:
            for name, history in self.metrics_history.items():
                if history:
                    latest = history[-1]
                    summary[name] = {
                        'latest_value': latest.value,
                        'timestamp': latest.timestamp,
                        'count': len(history),
                        'metric_type': latest.metric_type.value
                    }
        
        return summary
    
    def export_metrics(self, filename: str, format: str = 'json'):
        """导出指标数据"""
        if format == 'json':
            data = {}
            with self.lock:
                for name, history in self.metrics_history.items():
                    data[name] = [
                        {
                            'value': m.value,
                            'timestamp': m.timestamp,
                            'tags': m.tags,
                            'type': m.metric_type.value
                        }
                        for m in history
                    ]
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'csv':
            import csv
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'value', 'timestamp', 'tags', 'type'])
                
                with self.lock:
                    for name, history in self.metrics_history.items():
                        for m in history:
                            writer.writerow([
                                name, m.value, m.timestamp, 
                                json.dumps(m.tags), m.metric_type.value
                            ])


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity  # 标准差倍数
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.anomaly_history: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        # 异常检测算法
        self.algorithms = {
            'z_score': self._z_score_detection,
            'iqr': self._iqr_detection,
            'moving_average': self._moving_average_detection,
            'isolation_forest': self._isolation_forest_detection
        }
    
    def detect_anomalies(self, metrics_collector: MetricsCollector, 
                        metric_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        
        if metric_names is None:
            metric_names = list(metrics_collector.metrics_history.keys())
        
        for metric_name in metric_names:
            history = metrics_collector.get_metric(metric_name)
            if len(history) < 10:  # 需要足够的数据点
                continue
            
            # 使用多种算法检测
            for algorithm_name, algorithm_func in self.algorithms.items():
                try:
                    anomaly_indices = algorithm_func(history)
                    for idx in anomaly_indices:
                        anomaly = {
                            'metric_name': metric_name,
                            'timestamp': history[idx].timestamp,
                            'value': history[idx].value,
                            'algorithm': algorithm_name,
                            'severity': self._calculate_severity(history, idx),
                            'tags': history[idx].tags
                        }
                        anomalies.append(anomaly)
                except Exception as e:
                    logging.error(f"Error in anomaly detection algorithm {algorithm_name}: {e}")
        
        # 记录异常历史
        with self.lock:
            self.anomaly_history.extend(anomalies)
            # 限制历史记录大小
            if len(self.anomaly_history) > 1000:
                self.anomaly_history = self.anomaly_history[-1000:]
        
        return anomalies
    
    def _z_score_detection(self, history: List[MetricData]) -> List[int]:
        """Z-Score异常检测"""
        if len(history) < 2:
            return []
        
        values = [m.value for m in history]
        mean = np.mean(values) if HAS_NUMPY else sum(values) / len(values)
        std = np.std(values) if HAS_NUMPY else (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        
        if std == 0:
            return []
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean) / std
            if z_score > self.sensitivity:
                anomalies.append(i)
        
        return anomalies
    
    def _iqr_detection(self, history: List[MetricData]) -> List[int]:
        """IQR（四分位距）异常检测"""
        if len(history) < 4:
            return []
        
        values = sorted([m.value for m in history])
        q1 = values[len(values) // 4]
        q3 = values[3 * len(values) // 4]
        iqr = q3 - q1
        
        if iqr == 0:
            return []
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = []
        for i, metric in enumerate(history):
            if metric.value < lower_bound or metric.value > upper_bound:
                anomalies.append(i)
        
        return anomalies
    
    def _moving_average_detection(self, history: List[MetricData]) -> List[int]:
        """移动平均异常检测"""
        if len(history) < self.window_size:
            return []
        
        anomalies = []
        window_size = min(self.window_size, len(history) // 2)
        
        for i in range(window_size, len(history)):
            window_values = [m.value for m in history[i-window_size:i]]
            moving_avg = sum(window_values) / len(window_values)
            current_value = history[i].value
            
            # 计算偏差
            deviation = abs(current_value - moving_avg) / moving_avg if moving_avg != 0 else 0
            
            if deviation > 0.3:  # 30%偏差阈值
                anomalies.append(i)
        
        return anomalies
    
    def _isolation_forest_detection(self, history: List[MetricData]) -> List[int]:
        """孤立森林异常检测（简化版本）"""
        if len(history) < 20:
            return []
        
        # 简化的孤立森林实现
        values = np.array([[m.value] for m in history]) if HAS_NUMPY else [[m.value] for m in history]
        
        # 使用简单的统计方法作为替代
        if HAS_NUMPY:
            mean = np.mean(values)
            std = np.std(values)
            z_scores = np.abs((values - mean) / std)
            
            anomalies = []
            for i, z_score in enumerate(z_scores):
                if z_score > self.sensitivity:
                    anomalies.append(i)
            
            return anomalies
        else:
            return self._z_score_detection(history)
    
    def _calculate_severity(self, history: List[MetricData], index: int) -> str:
        """计算异常严重程度"""
        if index == 0:
            return "low"
        
        # 计算相对于前一个点的变化
        change = abs(history[index].value - history[index-1].value)
        relative_change = change / abs(history[index-1].value) if history[index-1].value != 0 else 0
        
        if relative_change > 1.0:  # 100%变化
            return "high"
        elif relative_change > 0.5:  # 50%变化
            return "medium"
        else:
            return "low"
    
    def update_baseline(self, metrics_collector: MetricsCollector, metric_names: Optional[List[str]] = None):
        """更新基线统计"""
        if metric_names is None:
            metric_names = list(metrics_collector.metrics_history.keys())
        
        for metric_name in metric_names:
            history = metrics_collector.get_metric(metric_name)
            if len(history) < 10:
                continue
            
            values = [m.value for m in history]
            
            with self.lock:
                self.baseline_stats[metric_name] = {
                    'mean': np.mean(values) if HAS_NUMPY else sum(values) / len(values),
                    'std': np.std(values) if HAS_NUMPY else (sum((x - sum(values)/len(values)) ** 2 for x in values) / len(values)) ** 0.5,
                    'min': min(values),
                    'max': max(values),
                    'median': np.median(values) if HAS_NUMPY else sorted(values)[len(values)//2],
                    'count': len(values),
                    'last_updated': time.time()
                }
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """获取异常摘要"""
        with self.lock:
            if not self.anomaly_history:
                return {'total_anomalies': 0, 'recent_anomalies': []}
            
            # 最近24小时的异常
            recent_threshold = time.time() - 24 * 3600
            recent_anomalies = [
                a for a in self.anomaly_history 
                if a['timestamp'] > recent_threshold
            ]
            
            # 按指标分组
            anomalies_by_metric = {}
            for anomaly in recent_anomalies:
                metric_name = anomaly['metric_name']
                if metric_name not in anomalies_by_metric:
                    anomalies_by_metric[metric_name] = []
                anomalies_by_metric[metric_name].append(anomaly)
            
            return {
                'total_anomalies': len(self.anomaly_history),
                'recent_anomalies': len(recent_anomalies),
                'anomalies_by_metric': anomalies_by_metric,
                'baseline_stats': self.baseline_stats
            }


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: Dict[str, callable] = {}
        self.lock = threading.Lock()
        
        # 默认告警规则
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认告警规则"""
        default_rules = {
            'high_cpu_usage': {
                'metric_name': 'system_cpu_usage',
                'condition': 'greater_than',
                'threshold': 80.0,
                'level': AlertLevel.WARNING,
                'duration': 300,  # 5分钟
                'message': 'CPU使用率过高: {value}%'
            },
            'high_memory_usage': {
                'metric_name': 'system_memory_usage',
                'condition': 'greater_than',
                'threshold': 85.0,
                'level': AlertLevel.WARNING,
                'duration': 300,
                'message': '内存使用率过高: {value}%'
            },
            'high_gpu_memory_usage': {
                'metric_name': 'gpu_memory_usage',
                'condition': 'greater_than',
                'threshold': 90.0,
                'level': AlertLevel.WARNING,
                'duration': 60,
                'message': 'GPU内存使用率过高: {value}%'
            },
            'training_loss_spike': {
                'metric_name': 'training_loss',
                'condition': 'spike',
                'threshold': 2.0,  # 2倍变化
                'level': AlertLevel.INFO,
                'duration': 0,
                'message': '训练损失出现异常波动: {value}'
            },
            'gradient_explosion': {
                'metric_name': 'training_gradient_norm',
                'condition': 'greater_than',
                'threshold': 10.0,
                'level': AlertLevel.ERROR,
                'duration': 0,
                'message': '梯度爆炸风险: {value}'
            }
        }
        
        for rule_name, rule in default_rules.items():
            self.add_alert_rule(rule_name, rule)
    
    def add_alert_rule(self, name: str, rule: Dict[str, Any]):
        """添加告警规则"""
        required_fields = ['metric_name', 'condition', 'threshold', 'level']
        for field in required_fields:
            if field not in rule:
                raise ValueError(f"Missing required field: {field}")
        
        self.alert_rules[name] = rule
    
    def remove_alert_rule(self, name: str):
        """移除告警规则"""
        if name in self.alert_rules:
            del self.alert_rules[name]
    
    def check_alerts(self, metrics_collector: MetricsCollector) -> List[Alert]:
        """检查告警"""
        new_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            try:
                alert = self._evaluate_rule(rule_name, rule, metrics_collector)
                if alert:
                    new_alerts.append(alert)
            except Exception as e:
                logging.error(f"Error evaluating alert rule {rule_name}: {e}")
        
        # 处理新告警
        for alert in new_alerts:
            self._handle_alert(alert)
        
        return new_alerts
    
    def _evaluate_rule(self, rule_name: str, rule: Dict[str, Any], 
                      metrics_collector: MetricsCollector) -> Optional[Alert]:
        """评估告警规则"""
        metric_name = rule['metric_name']
        condition = rule['condition']
        threshold = rule['threshold']
        level = rule['level']
        
        # 获取指标数据
        latest_metric = metrics_collector.get_latest_metric(metric_name)
        if not latest_metric:
            return None
        
        current_value = latest_metric.value
        
        # 检查条件
        triggered = False
        
        if condition == 'greater_than':
            triggered = current_value > threshold
        elif condition == 'less_than':
            triggered = current_value < threshold
        elif condition == 'equal':
            triggered = abs(current_value - threshold) < 1e-6
        elif condition == 'spike':
            # 检查是否出现突变
            history = metrics_collector.get_metric(metric_name, 
                                                  time.time() - rule.get('duration', 300))
            if len(history) >= 2:
                prev_value = history[-2].value
                if prev_value != 0:
                    change_ratio = abs(current_value - prev_value) / abs(prev_value)
                    triggered = change_ratio > threshold
        else:
            logging.warning(f"Unknown condition: {condition}")
            return None
        
        if not triggered:
            # 检查是否需要解决现有告警
            self._resolve_alert_if_needed(rule_name, current_value)
            return None
        
        # 检查持续时间
        duration = rule.get('duration', 0)
        if duration > 0:
            # 检查是否持续超过指定时间
            since_time = time.time() - duration
            recent_metrics = metrics_collector.get_metric(metric_name, since_time)
            
            if not recent_metrics:
                return None
            
            # 检查所有最近的指标是否都满足条件
            all_triggered = True
            for metric in recent_metrics:
                if condition == 'greater_than':
                    if metric.value <= threshold:
                        all_triggered = False
                        break
                elif condition == 'less_than':
                    if metric.value >= threshold:
                        all_triggered = False
                        break
            
            if not all_triggered:
                return None
        
        # 生成告警ID
        alert_id = f"{rule_name}_{int(time.time())}"
        
        # 生成告警消息
        message = rule['message'].format(value=current_value)
        
        # 创建告警
        alert = Alert(
            id=alert_id,
            name=rule_name,
            level=level,
            message=message,
            timestamp=time.time(),
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            tags=latest_metric.tags
        )
        
        return alert
    
    def _resolve_alert_if_needed(self, rule_name: str, current_value: float):
        """根据需要解决告警"""
        # 查找该规则的活动告警
        active_alerts = [a for a in self.alerts.values() 
                        if a.name == rule_name and not a.resolved]
        
        for alert in active_alerts:
            rule = self.alert_rules.get(rule_name)
            if not rule:
                continue
            
            condition = rule['condition']
            threshold = rule['threshold']
            
            # 检查是否恢复正常
            resolved = False
            
            if condition == 'greater_than':
                resolved = current_value <= threshold * 0.9  # 10%缓冲
            elif condition == 'less_than':
                resolved = current_value >= threshold * 1.1  # 10%缓冲
            elif condition == 'spike':
                resolved = True  # 突变告警自动解决
            
            if resolved:
                self.resolve_alert(alert.id)
    
    def _handle_alert(self, alert: Alert):
        """处理告警"""
        with self.lock:
            # 检查是否已有相同规则的未解决告警
            existing_alerts = [a for a in self.alerts.values() 
                             if a.name == alert.name and not a.resolved]
            
            if existing_alerts:
                # 更新现有告警
                existing_alert = existing_alerts[0]
                existing_alert.current_value = alert.current_value
                existing_alert.timestamp = alert.timestamp
                alert = existing_alert
            else:
                # 添加新告警
                self.alerts[alert.id] = alert
            
            # 添加到历史记录
            self.alert_history.append(alert)
            
            # 限制历史记录大小
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
        
        # 发送通知
        self._send_notifications(alert)
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                if not alert.resolved:
                    alert.resolved = True
                    alert.resolved_timestamp = time.time()
                    
                    # 发送解决通知
                    self._send_resolved_notification(alert)
    
    def add_notification_handler(self, name: str, handler: callable):
        """添加通知处理器"""
        self.notification_handlers[name] = handler
    
    def _send_notifications(self, alert: Alert):
        """发送通知"""
        for handler_name, handler in self.notification_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Error in notification handler {handler_name}: {e}")
    
    def _send_resolved_notification(self, alert: Alert):
        """发送解决通知"""
        # 创建解决通知
        resolved_alert = Alert(
            id=f"{alert.id}_resolved",
            name=alert.name,
            level=AlertLevel.INFO,
            message=f"告警已解决: {alert.message}",
            timestamp=time.time(),
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold=alert.threshold,
            tags=alert.tags,
            resolved=True,
            resolved_timestamp=time.time()
        )
        
        self._send_notifications(resolved_alert)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活动告警"""
        with self.lock:
            return [a for a in self.alerts.values() if not a.resolved]
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        with self.lock:
            return self.alert_history[-limit:]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        with self.lock:
            active_alerts = self.get_active_alerts()
            
            # 按级别分组
            alerts_by_level = {}
            for alert in active_alerts:
                level = alert.level.value
                if level not in alerts_by_level:
                    alerts_by_level[level] = []
                alerts_by_level[level].append(alert)
            
            # 按规则分组
            alerts_by_rule = {}
            for alert in active_alerts:
                rule = alert.name
                if rule not in alerts_by_rule:
                    alerts_by_rule[rule] = []
                alerts_by_rule[rule].append(alert)
            
            return {
                'total_active_alerts': len(active_alerts),
                'alerts_by_level': alerts_by_level,
                'alerts_by_rule': alerts_by_rule,
                'total_alerts_count': len(self.alert_history)
            }


class BasicVisualization:
    """基础可视化界面"""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 alert_manager: AlertManager, 
                 anomaly_detector: AnomalyDetector):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.anomaly_detector = anomaly_detector
        self.is_running = False
        self.update_interval = 5.0  # 5秒更新一次
        
        # 可视化配置
        self.fig = None
        self.axes = None
        self.animation = None
        
        # 图表数据
        self.chart_data = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'training_loss': [],
            'gpu_usage': []
        }
    
    def start_dashboard(self, update_interval: float = 5.0):
        """启动仪表板"""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, using console dashboard")
            self.start_console_dashboard(update_interval)
            return
        
        self.update_interval = update_interval
        self.is_running = True
        
        # 创建图表
        self._create_charts()
        
        # 启动动画
        self.animation = animation.FuncAnimation(
            self.fig, self._update_charts, 
            interval=int(update_interval * 1000),
            blit=False
        )
        
        plt.show()
    
    def start_console_dashboard(self, update_interval: float = 5.0):
        """启动控制台仪表板"""
        self.update_interval = update_interval
        self.is_running = True
        
        try:
            while self.is_running:
                self._update_console_dashboard()
                time.sleep(update_interval)
        except KeyboardInterrupt:
            self.stop_dashboard()
    
    def _create_charts(self):
        """创建图表"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Picotron 监控仪表板', fontsize=16)
        
        # 设置子图标题
        self.axes[0, 0].set_title('系统资源使用率')
        self.axes[0, 1].set_title('训练损失')
        self.axes[1, 0].set_title('GPU使用率')
        self.axes[1, 1].set_title('告警状态')
        
        # 调整布局
        plt.tight_layout()
    
    def _update_charts(self, frame):
        """更新图表"""
        try:
            # 清除所有子图
            for ax in self.axes.flat:
                ax.clear()
            
            # 更新系统资源图表
            self._update_system_chart()
            
            # 更新训练损失图表
            self._update_training_chart()
            
            # 更新GPU使用率图表
            self._update_gpu_chart()
            
            # 更新告警状态图表
            self._update_alert_chart()
            
            # 重新设置标题
            self.axes[0, 0].set_title('系统资源使用率')
            self.axes[0, 1].set_title('训练损失')
            self.axes[1, 0].set_title('GPU使用率')
            self.axes[1, 1].set_title('告警状态')
            
        except Exception as e:
            logging.error(f"Error updating charts: {e}")
    
    def _update_system_chart(self):
        """更新系统资源图表"""
        ax = self.axes[0, 0]
        
        # 获取最近的CPU和内存使用率
        since_time = time.time() - 300  # 5分钟
        cpu_data = self.metrics_collector.get_metric('system_cpu_usage', since_time)
        memory_data = self.metrics_collector.get_metric('system_memory_usage', since_time)
        
        if cpu_data and memory_data:
            timestamps = [m.timestamp for m in cpu_data]
            cpu_values = [m.value for m in cpu_data]
            memory_values = [m.value for m in memory_data]
            
            ax.plot(timestamps, cpu_values, 'b-', label='CPU使用率')
            ax.plot(timestamps, memory_values, 'r-', label='内存使用率')
            ax.set_ylim(0, 100)
            ax.legend()
            ax.set_ylabel('使用率 (%)')
            ax.grid(True, alpha=0.3)
    
    def _update_training_chart(self):
        """更新训练损失图表"""
        ax = self.axes[0, 1]
        
        # 获取最近的训练损失
        since_time = time.time() - 300  # 5分钟
        loss_data = self.metrics_collector.get_metric('training_loss', since_time)
        
        if loss_data:
            timestamps = [m.timestamp for m in loss_data]
            loss_values = [m.value for m in loss_data]
            
            ax.plot(timestamps, loss_values, 'g-', label='训练损失')
            ax.set_ylabel('损失值')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _update_gpu_chart(self):
        """更新GPU使用率图表"""
        ax = self.axes[1, 0]
        
        # 获取GPU指标
        gpu_metrics = []
        for name in self.metrics_collector.metrics_history.keys():
            if name.startswith('gpu_') and name.endswith('_memory_usage'):
                gpu_metrics.append(name)
        
        if gpu_metrics:
            since_time = time.time() - 300  # 5分钟
            
            for metric_name in gpu_metrics:
                data = self.metrics_collector.get_metric(metric_name, since_time)
                if data:
                    timestamps = [m.timestamp for m in data]
                    values = [m.value for m in data]
                    label = metric_name.replace('_memory_usage', '')
                    ax.plot(timestamps, values, label=label)
            
            ax.set_ylim(0, 100)
            ax.set_ylabel('使用率 (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _update_alert_chart(self):
        """更新告警状态图表"""
        ax = self.axes[1, 1]
        
        # 获取告警摘要
        alert_summary = self.alert_manager.get_alert_summary()
        
        # 显示告警统计
        active_alerts = alert_summary['total_active_alerts']
        alerts_by_level = alert_summary['alerts_by_level']
        
        # 创建条形图
        levels = list(alerts_by_level.keys())
        counts = [len(alerts_by_level[level]) for level in levels]
        
        if levels:
            colors = ['green', 'yellow', 'orange', 'red']
            bars = ax.bar(levels, counts, color=colors[:len(levels)])
            ax.set_ylabel('告警数量')
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
        
        ax.set_title(f'活动告警: {active_alerts}')
    
    def _update_console_dashboard(self):
        """更新控制台仪表板"""
        # 清屏
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 60)
        print("Picotron 监控仪表板")
        print("=" * 60)
        print(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 系统资源
        print("系统资源:")
        cpu_metric = self.metrics_collector.get_latest_metric('system_cpu_usage')
        memory_metric = self.metrics_collector.get_latest_metric('system_memory_usage')
        
        if cpu_metric:
            print(f"  CPU使用率: {cpu_metric.value:.1f}%")
        if memory_metric:
            print(f"  内存使用率: {memory_metric.value:.1f}%")
        
        # 训练指标
        print("\n训练指标:")
        loss_metric = self.metrics_collector.get_latest_metric('training_loss')
        lr_metric = self.metrics_collector.get_latest_metric('training_learning_rate')
        
        if loss_metric:
            print(f"  训练损失: {loss_metric.value:.4f}")
        if lr_metric:
            print(f"  学习率: {lr_metric.value:.6f}")
        
        # GPU使用率
        print("\nGPU使用率:")
        for name in self.metrics_collector.metrics_history.keys():
            if name.startswith('gpu_') and name.endswith('_memory_usage'):
                metric = self.metrics_collector.get_latest_metric(name)
                if metric:
                    print(f"  {name}: {metric.value:.1f}%")
        
        # 告警状态
        print("\n告警状态:")
        alert_summary = self.alert_manager.get_alert_summary()
        active_alerts = alert_summary['total_active_alerts']
        
        if active_alerts > 0:
            print(f"  活动告警: {active_alerts}")
            for alert in self.alert_manager.get_active_alerts()[:5]:  # 显示前5个
                print(f"    [{alert.level.value.upper()}] {alert.message}")
        else:
            print("  无活动告警")
        
        # 异常检测
        print("\n异常检测:")
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        recent_anomalies = anomaly_summary['recent_anomalies']
        
        if recent_anomalies > 0:
            print(f"  最近异常: {recent_anomalies}")
        else:
            print("  无检测到异常")
        
        print("\n按 Ctrl+C 退出...")
    
    def stop_dashboard(self):
        """停止仪表板"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        if self.fig:
            plt.close(self.fig)
    
    def generate_report(self, filename: str, time_range: int = 3600):
        """生成报告"""
        since_time = time.time() - time_range
        
        # 收集数据
        metrics_summary = self.metrics_collector.get_metrics_summary()
        alert_summary = self.alert_manager.get_alert_summary()
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        
        # 生成报告
        report = {
            'report_time': time.time(),
            'time_range': time_range,
            'metrics_summary': metrics_summary,
            'alert_summary': alert_summary,
            'anomaly_summary': anomaly_summary,
            'system_info': self._get_system_info()
        }
        
        # 保存报告
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"报告已生成: {filename}")
    
    def _get_system_info(self):
        """获取系统信息"""
        info = {
            'platform': os.name,
            'python_version': sys.version,
            'timestamp': time.time()
        }
        
        if HAS_PSUTIL:
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total
            })
        
        if HAS_TORCH and torch.cuda.is_available():
            info.update({
                'cuda_available': True,
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda
            })
        
        return info


class BasicMonitoringSystem:
    """基础监控系统主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 初始化组件
        self.metrics_collector = MetricsCollector(
            max_history_size=self.config.get('max_history_size', 10000)
        )
        
        self.anomaly_detector = AnomalyDetector(
            window_size=self.config.get('anomaly_window_size', 100),
            sensitivity=self.config.get('anomaly_sensitivity', 2.0)
        )
        
        self.alert_manager = AlertManager()
        
        self.visualization = BasicVisualization(
            self.metrics_collector, self.alert_manager, self.anomaly_detector
        )
        
        # 监控线程
        self.monitoring_thread = None
        self.is_monitoring = False
        self.monitoring_interval = self.config.get('monitoring_interval', 5.0)
        
        # 设置默认通知处理器
        self._setup_default_notifications()
    
    def _setup_default_notifications(self):
        """设置默认通知处理器"""
        # 控制台通知
        def console_notification(alert):
            level_icon = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨"
            }
            
            icon = level_icon.get(alert.level, "📢")
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))
            
            print(f"{icon} [{alert.level.value.upper()}] {timestamp}")
            print(f"   {alert.message}")
            if alert.resolved:
                print(f"   状态: 已解决")
            print()
        
        self.alert_manager.add_notification_handler('console', console_notification)
    
    def start_monitoring(self):
        """开始监控"""
        self.is_monitoring = True
        
        # 启动指标收集
        self.metrics_collector.start_collection(self.monitoring_interval)
        
        # 启动监控线程
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
        
        print("监控系统已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        
        # 停止指标收集
        self.metrics_collector.stop_collection()
        
        # 等待监控线程结束
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # 停止可视化
        self.visualization.stop_dashboard()
        
        print("监控系统已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 检查告警
                new_alerts = self.alert_manager.check_alerts(self.metrics_collector)
                
                # 检测异常
                anomalies = self.anomaly_detector.detect_anomalies(self.metrics_collector)
                
                # 定期更新基线
                if int(time.time()) % 3600 == 0:  # 每小时更新一次
                    self.anomaly_detector.update_baseline(self.metrics_collector)
                
                time.sleep(self.monitoring_interval)
            
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def start_dashboard(self, console_only: bool = False):
        """启动仪表板"""
        if console_only or not HAS_MATPLOTLIB:
            self.visualization.start_console_dashboard()
        else:
            self.visualization.start_dashboard()
    
    def add_custom_alert_rule(self, name: str, rule: Dict[str, Any]):
        """添加自定义告警规则"""
        self.alert_manager.add_alert_rule(name, rule)
    
    def add_custom_metric_collector(self, name: str, collector_func):
        """添加自定义指标收集器"""
        self.metrics_collector.add_custom_collector(name, collector_func)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'metrics_summary': self.metrics_collector.get_metrics_summary(),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'anomaly_summary': self.anomaly_detector.get_anomaly_summary(),
            'timestamp': time.time()
        }
    
    def export_data(self, output_dir: str):
        """导出监控数据"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 导出指标数据
        metrics_file = os.path.join(output_dir, 'metrics.json')
        self.metrics_collector.export_metrics(metrics_file)
        
        # 生成报告
        report_file = os.path.join(output_dir, 'monitoring_report.json')
        self.visualization.generate_report(report_file)
        
        print(f"监控数据已导出到: {output_dir}")


__all__ = [
    'MetricsCollector', 'AnomalyDetector', 'AlertManager', 
    'BasicVisualization', 'BasicMonitoringSystem',
    'MetricType', 'AlertLevel', 'MetricData', 'Alert'
]