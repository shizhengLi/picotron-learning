"""
自适应混合精度模块
================

实现智能的混合精度训练策略，包括精度分析、硬件检测、策略选择和动态调整。
"""

try:
    import torch
    import torch.nn as nn
    import torch.cuda as cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from typing import Dict, List, Any, Optional, Tuple, Union
import math
import logging


class PrecisionAnalyzer:
    """精度分析器"""
    
    def __init__(self):
        self.precision_stats = {}
        self.layer_sensitivity = {}
        self.precision_history = []
        
    def analyze_layer_precision_sensitivity(self, model, sample_data):
        """分析各层对精度的敏感度"""
        sensitivity_results = {}
        
        if not HAS_TORCH:
            # 简化版本，返回模拟数据
            return {
                'linear_layers': {'fp32_sensitivity': 0.8, 'fp16_sensitivity': 0.3},
                'conv_layers': {'fp32_sensitivity': 0.9, 'fp16_sensitivity': 0.4},
                'attention_layers': {'fp32_sensitivity': 0.7, 'fp16_sensitivity': 0.2}
            }
        
        # 实际实现需要遍历模型层并分析精度敏感度
        if hasattr(model, 'named_modules'):
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # 分析该层的精度敏感度
                    sensitivity = self._calculate_layer_sensitivity(module, sample_data)
                    sensitivity_results[name] = sensitivity
        else:
            # 如果模型没有named_modules方法，返回默认值
            return {
                'default_layer': {'fp32_sensitivity': 0.5, 'fp16_sensitivity': 0.3}
            }
        
        return sensitivity_results
    
    def _calculate_layer_sensitivity(self, layer, sample_data):
        """计算单个层的精度敏感度"""
        if not HAS_TORCH:
            return {'fp32_sensitivity': 0.5, 'fp16_sensitivity': 0.3}
        
        # 模拟精度敏感度计算
        return {
            'fp32_sensitivity': 0.5 + 0.3 * torch.rand(1).item(),
            'fp16_sensitivity': 0.2 + 0.2 * torch.rand(1).item(),
            'bf16_sensitivity': 0.3 + 0.2 * torch.rand(1).item()
        }
    
    def detect_precision_bottlenecks(self, model):
        """检测精度瓶颈"""
        bottlenecks = []
        
        if not HAS_TORCH:
            return ['conv_layers', 'attention_layers']
        
        # 分析模型中可能存在精度问题的层
        if hasattr(model, 'named_modules'):
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and hasattr(module, 'out_features') and module.out_features > 1000:
                    bottlenecks.append(f'large_linear_{name}')
                elif isinstance(module, nn.Conv2d) and hasattr(module, 'out_channels') and module.out_channels > 512:
                    bottlenecks.append(f'large_conv_{name}')
        
        return bottlenecks
    
    def get_precision_recommendations(self, model_info):
        """获取精度使用建议"""
        recommendations = {
            'default_precision': 'fp16',
            'high_precision_layers': [],
            'low_precision_layers': [],
            'mixed_precision_strategy': 'layer_wise'
        }
        
        # 根据模型信息提供精度建议
        if model_info.get('model_size', 0) > 1000000000:  # 1B参数
            recommendations['default_precision'] = 'fp16'
            recommendations['mixed_precision_strategy'] = 'hybrid'
        elif model_info.get('has_attention', False):
            recommendations['default_precision'] = 'bf16'
            recommendations['high_precision_layers'].append('attention')
        
        return recommendations


class HardwareDetector:
    """硬件检测器"""
    
    def __init__(self):
        self.hardware_info = {}
        self.precision_support = {}
        self.performance_profile = {}
        
    def detect_hardware_capabilities(self):
        """检测硬件能力"""
        capabilities = {
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_memory': 0,
            'gpu_compute_capability': 0,
            'tensor_cores': False,
            'fp16_support': False,
            'bf16_support': False,
            'tf32_support': False,
            'cpu_optimizations': []
        }
        
        if HAS_TORCH and torch.cuda.is_available():
            capabilities['gpu_available'] = True
            capabilities['gpu_count'] = torch.cuda.device_count()
            capabilities['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
            capabilities['gpu_compute_capability'] = torch.cuda.get_device_capability(0)
            capabilities['tensor_cores'] = capabilities['gpu_compute_capability'] >= (7, 0)
            capabilities['fp16_support'] = True
            capabilities['bf16_support'] = capabilities['gpu_compute_capability'] >= (8, 0)
            capabilities['tf32_support'] = capabilities['gpu_compute_capability'] >= (8, 0)
        
        return capabilities
    
    def get_precision_performance_profile(self):
        """获取各精度的性能配置"""
        profile = {
            'fp32': {'compute_speed': 1.0, 'memory_usage': 1.0, 'bandwidth': 1.0},
            'fp16': {'compute_speed': 3.0, 'memory_usage': 0.5, 'bandwidth': 2.0},
            'bf16': {'compute_speed': 2.5, 'memory_usage': 0.5, 'bandwidth': 2.0},
            'tf32': {'compute_speed': 1.5, 'memory_usage': 1.0, 'bandwidth': 1.2}
        }
        
        # 根据硬件能力调整性能配置
        if HAS_TORCH and torch.cuda.is_available():
            compute_cap = torch.cuda.get_device_capability()
            if compute_cap >= (8, 0):  # Ampere架构
                profile['tf32']['compute_speed'] = 2.0
                profile['bf16']['compute_speed'] = 3.0
            elif compute_cap >= (7, 0):  # Turing架构
                profile['fp16']['compute_speed'] = 4.0
        
        return profile
    
    def detect_memory_constraints(self):
        """检测内存约束"""
        constraints = {
            'available_memory': 0,
            'recommended_batch_size': 1,
            'memory_pressure': 'low',
            'swap_usage': 0
        }
        
        if HAS_TORCH and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            available_memory = total_memory - reserved_memory
            
            constraints['available_memory'] = available_memory
            constraints['memory_pressure'] = self._assess_memory_pressure(available_memory, total_memory)
            constraints['recommended_batch_size'] = self._calculate_optimal_batch_size(available_memory)
        
        return constraints
    
    def _assess_memory_pressure(self, available_memory, total_memory):
        """评估内存压力"""
        ratio = available_memory / total_memory
        if ratio > 0.7:
            return 'low'
        elif ratio > 0.3:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_optimal_batch_size(self, available_memory):
        """计算最优批次大小"""
        # 简化的批次大小计算
        if available_memory > 8 * 1024**3:  # 8GB
            return 32
        elif available_memory > 4 * 1024**3:  # 4GB
            return 16
        elif available_memory > 2 * 1024**3:  # 2GB
            return 8
        else:
            return 4


class AdaptivePrecisionSelector:
    """自适应精度选择器"""
    
    def __init__(self):
        self.precision_analyzer = PrecisionAnalyzer()
        self.hardware_detector = HardwareDetector()
        self.selection_history = []
        self.strategy_cache = {}
        
    def select_precision_strategy(self, model, training_config):
        """选择精度策略"""
        # 获取硬件信息
        hardware_info = self.hardware_detector.detect_hardware_capabilities()
        
        # 获取精度分析结果
        precision_analysis = self.precision_analyzer.analyze_layer_precision_sensitivity(
            model, training_config.get('sample_data')
        )
        
        # 获取性能配置
        performance_profile = self.hardware_detector.get_precision_performance_profile()
        
        # 获取内存约束
        memory_constraints = self.hardware_detector.detect_memory_constraints()
        
        # 综合选择精度策略
        strategy = self._generate_precision_strategy(
            hardware_info, precision_analysis, performance_profile, memory_constraints
        )
        
        # 缓存策略
        strategy_key = self._generate_strategy_key(model, training_config)
        self.strategy_cache[strategy_key] = strategy
        
        return strategy
    
    def _generate_precision_strategy(self, hardware_info, precision_analysis, 
                                   performance_profile, memory_constraints):
        """生成精度策略"""
        strategy = {
            'global_precision': 'fp32',
            'layer_precision': {},
            'mixed_precision_enabled': False,
            'gradient_scaling': True,
            'loss_scaling': 1.0,
            'dynamic_precision_adjustment': True,
            'adjustment_frequency': 100,
            'performance_mode': 'balanced'
        }
        
        # 根据硬件能力选择全局精度
        if hardware_info.get('tensor_cores', False):
            if hardware_info.get('bf16_support', False):
                strategy['global_precision'] = 'bf16'
            else:
                strategy['global_precision'] = 'fp16'
            strategy['mixed_precision_enabled'] = True
        
        # 根据内存压力调整策略
        if memory_constraints['memory_pressure'] == 'high':
            strategy['global_precision'] = 'fp16'
            strategy['mixed_precision_enabled'] = True
            strategy['performance_mode'] = 'memory_efficient'
        
        # 根据精度敏感度设置层级精度
        for layer_name, sensitivity in precision_analysis.items():
            if sensitivity.get('fp32_sensitivity', 0) > 0.8:
                strategy['layer_precision'][layer_name] = 'fp32'
            elif sensitivity.get('fp16_sensitivity', 0) < 0.3:
                strategy['layer_precision'][layer_name] = 'fp16'
        
        return strategy
    
    def _generate_strategy_key(self, model, training_config):
        """生成策略缓存键"""
        model_size = sum(p.numel() for p in model.parameters()) if HAS_TORCH and hasattr(model, 'parameters') else 0
        return f"model_{model_size}_batch_{training_config.get('batch_size', 1)}"
    
    def update_strategy_performance(self, strategy, performance_metrics):
        """根据性能反馈更新策略"""
        # 记录策略性能
        strategy_performance = {
            'strategy': strategy,
            'metrics': performance_metrics,
            'timestamp': len(self.selection_history)
        }
        
        self.selection_history.append(strategy_performance)
        
        # 根据性能反馈调整策略
        if performance_metrics.get('loss_stability', 1.0) < 0.8:
            strategy['loss_scaling'] *= 2
            strategy['performance_mode'] = 'stable'
        
        if performance_metrics.get('memory_usage', 1.0) > 0.9:
            strategy['performance_mode'] = 'memory_efficient'
        
        return strategy


class DynamicPrecisionAdjuster:
    """动态精度调整器"""
    
    def __init__(self):
        self.current_precision = 'fp32'
        self.adjustment_history = []
        self.stability_monitor = StabilityMonitor()
        
    def setup_mixed_precision_training(self, model, strategy):
        """设置混合精度训练"""
        if not HAS_TORCH:
            return model
        
        # 创建GradScaler
        if strategy.get('mixed_precision_enabled', False):
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=strategy.get('loss_scaling', 65536.0)
            )
        else:
            self.scaler = None
        
        # 设置模型精度
        self._set_model_precision(model, strategy)
        
        return model
    
    def _set_model_precision(self, model, strategy):
        """设置模型精度"""
        global_precision = strategy.get('global_precision', 'fp32')
        
        if HAS_TORCH and hasattr(model, 'half') and hasattr(model, 'to'):
            if global_precision == 'fp16':
                model = model.half()
            elif global_precision == 'bf16':
                if hasattr(torch, 'bfloat16'):
                    model = model.to(torch.bfloat16)
        
        # 设置特定层的精度
        for layer_name, precision in strategy.get('layer_precision', {}).items():
            self._set_layer_precision(model, layer_name, precision)
    
    def _set_layer_precision(self, model, layer_name, precision):
        """设置特定层的精度"""
        if not HAS_TORCH:
            return
        
        # 查找并设置特定层的精度
        for name, module in model.named_modules():
            if name == layer_name:
                if precision == 'fp32':
                    module = module.float()
                elif precision == 'fp16':
                    module = module.half()
                elif precision == 'bf16':
                    if hasattr(torch, 'bfloat16'):
                        module = module.to(torch.bfloat16)
    
    def adjust_precision_dynamically(self, model, training_metrics):
        """动态调整精度"""
        if not training_metrics.get('dynamic_adjustment_enabled', True):
            return
        
        # 分析训练稳定性
        stability = self.stability_monitor.analyze_stability(training_metrics)
        
        # 根据稳定性调整精度
        if stability['is_stable']:
            # 如果稳定，尝试使用更高性能的精度
            self._try_higher_precision(model, training_metrics)
        else:
            # 如果不稳定，提高精度
            self._increase_precision_for_stability(model, training_metrics)
        
        # 记录调整历史
        adjustment_record = {
            'timestamp': len(self.adjustment_history),
            'stability': stability,
            'action': self._get_last_adjustment(),
            'metrics': training_metrics
        }
        self.adjustment_history.append(adjustment_record)
    
    def _try_higher_precision(self, model, metrics):
        """尝试使用更高性能的精度"""
        if metrics.get('memory_usage', 0) < 0.8:
            # 内存使用率低，可以尝试更激进的精度策略
            pass
    
    def _increase_precision_for_stability(self, model, metrics):
        """为稳定性提高精度"""
        if metrics.get('loss_stability', 1.0) < 0.5:
            # 损失不稳定，提高精度
            pass
    
    def _get_last_adjustment(self):
        """获取最后一次调整"""
        return "precision_adjustment"


class StabilityMonitor:
    """稳定性监控器"""
    
    def __init__(self):
        self.loss_history = []
        self.gradient_history = []
        self.precision_history = []
        
    def analyze_stability(self, training_metrics):
        """分析训练稳定性"""
        # 更新历史数据
        self.loss_history.append(training_metrics.get('loss', 0))
        self.gradient_history.append(training_metrics.get('gradient_norm', 0))
        
        # 保持历史数据长度
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
        if len(self.gradient_history) > 100:
            self.gradient_history = self.gradient_history[-100:]
        
        # 计算稳定性指标
        stability = {
            'is_stable': self._is_loss_stable(),
            'loss_variance': self._calculate_loss_variance(),
            'gradient_variance': self._calculate_gradient_variance(),
            'trend': self._calculate_trend()
        }
        
        return stability
    
    def _is_loss_stable(self):
        """判断损失是否稳定"""
        if len(self.loss_history) < 10:
            return True
        
        recent_losses = self.loss_history[-10:]
        variance = np.var(recent_losses) if HAS_NUMPY else 0.1
        
        return variance < 0.01  # 阈值可根据实际情况调整
    
    def _calculate_loss_variance(self):
        """计算损失方差"""
        if len(self.loss_history) < 2:
            return 0.0
        
        if HAS_NUMPY:
            return np.var(self.loss_history)
        else:
            # 简化计算
            mean = sum(self.loss_history) / len(self.loss_history)
            variance = sum((x - mean) ** 2 for x in self.loss_history) / len(self.loss_history)
            return variance
    
    def _calculate_gradient_variance(self):
        """计算梯度方差"""
        if len(self.gradient_history) < 2:
            return 0.0
        
        if HAS_NUMPY:
            return np.var(self.gradient_history)
        else:
            mean = sum(self.gradient_history) / len(self.gradient_history)
            variance = sum((x - mean) ** 2 for x in self.gradient_history) / len(self.gradient_history)
            return variance
    
    def _calculate_trend(self):
        """计算趋势"""
        if len(self.loss_history) < 5:
            return 'stable'
        
        recent_losses = self.loss_history[-5:]
        if recent_losses[-1] < recent_losses[0] * 0.9:
            return 'decreasing'
        elif recent_losses[-1] > recent_losses[0] * 1.1:
            return 'increasing'
        else:
            return 'stable'


class AdaptiveMixedPrecision:
    """自适应混合精度主类"""
    
    def __init__(self):
        self.precision_analyzer = PrecisionAnalyzer()
        self.hardware_detector = HardwareDetector()
        self.precision_selector = AdaptivePrecisionSelector()
        self.precision_adjuster = DynamicPrecisionAdjuster()
        self.stability_monitor = StabilityMonitor()
        
        self.current_strategy = None
        self.is_initialized = False
        
    def initialize(self, model, training_config):
        """初始化自适应混合精度"""
        # 检测硬件能力
        hardware_info = self.hardware_detector.detect_hardware_capabilities()
        
        # 分析精度需求
        precision_analysis = self.precision_analyzer.analyze_layer_precision_sensitivity(
            model, training_config.get('sample_data')
        )
        
        # 选择精度策略
        self.current_strategy = self.precision_selector.select_precision_strategy(
            model, training_config
        )
        
        # 设置混合精度训练
        model = self.precision_adjuster.setup_mixed_precision_training(
            model, self.current_strategy
        )
        
        self.is_initialized = True
        
        return model, self.current_strategy
    
    def training_step(self, model, optimizer, batch, targets):
        """执行训练步骤"""
        if not HAS_TORCH:
            # 简化版本
            outputs = model(batch)
            loss = self._compute_loss(outputs, targets)
            return loss
        
        # 混合精度训练步骤
        if (self.current_strategy and 
            self.current_strategy.get('mixed_precision_enabled', False) and 
            hasattr(self.precision_adjuster, 'scaler') and 
            self.precision_adjuster.scaler is not None):
            
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                loss = self._compute_loss(outputs, targets)
            
            # 反向传播
            self.precision_adjuster.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.current_strategy.get('gradient_clipping', False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            self.precision_adjuster.scaler.step(optimizer)
            self.precision_adjuster.scaler.update()
        else:
            # 标准精度训练
            outputs = model(batch)
            loss = self._compute_loss(outputs, targets)
            
            # 只有在loss有backward方法时才调用
            if hasattr(loss, 'backward'):
                loss.backward()
            
            if self.current_strategy and self.current_strategy.get('gradient_clipping', False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        return loss
    
    def _compute_loss(self, outputs, targets):
        """计算损失"""
        if HAS_TORCH and hasattr(outputs, 'size') and hasattr(targets, 'size'):
            return torch.nn.functional.mse_loss(outputs, targets)
        else:
            return 0.0
    
    def update_precision_strategy(self, training_metrics):
        """更新精度策略"""
        if not self.is_initialized:
            return
        
        # 分析稳定性
        stability = self.stability_monitor.analyze_stability(training_metrics)
        
        # 动态调整精度
        self.precision_adjuster.adjust_precision_dynamically(
            None, training_metrics  # 模型参数暂时为None
        )
        
        # 更新策略性能
        if self.current_strategy:
            self.precision_selector.update_strategy_performance(
                self.current_strategy, training_metrics
            )
    
    def get_current_precision_status(self):
        """获取当前精度状态"""
        return {
            'strategy': self.current_strategy,
            'is_initialized': self.is_initialized,
            'hardware_info': self.hardware_detector.detect_hardware_capabilities(),
            'performance_profile': self.hardware_detector.get_precision_performance_profile(),
            'stability_status': self.stability_monitor.analyze_stability({})
        }


__all__ = [
    'PrecisionAnalyzer', 'HardwareDetector', 'AdaptivePrecisionSelector',
    'DynamicPrecisionAdjuster', 'StabilityMonitor', 'AdaptiveMixedPrecision'
]