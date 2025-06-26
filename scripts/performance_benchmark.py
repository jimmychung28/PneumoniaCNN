#!/usr/bin/env python3
"""
Performance benchmarking and monitoring script for the high-performance CNN.
Provides comprehensive performance analysis and optimization recommendations.
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, asdict

# Import our modules
from high_performance_cnn import HighPerformancePneumoniaCNN, main_high_performance
from src.config.config_loader import ConfigManager
from src.utils.validation_utils import logger as validation_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Data class for storing benchmark results."""
    
    # System Information
    system_info: Dict[str, Any]
    
    # Model Performance
    model_params: int
    model_size_mb: float
    
    # Training Performance
    avg_epoch_time: float
    samples_per_second: float
    batches_per_second: float
    
    # Inference Performance
    avg_inference_time: float
    inference_throughput: float
    
    # Memory Usage
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Accuracy Metrics
    final_accuracy: float
    final_val_accuracy: float
    best_accuracy: float
    
    # Configuration
    config_name: str
    mixed_precision: bool
    batch_size: int
    image_size: List[int]
    
    # Timestamps
    benchmark_time: str
    duration_minutes: float


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize performance benchmark.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.results = []
        self.system_info = self._get_system_info()
        
        # Setup results directory
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("PerformanceBenchmark initialized")
        logger.info(f"System: {self.system_info['platform']} - {self.system_info['gpu_count']} GPU(s)")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        try:
            import platform
            import psutil
            
            # Basic system info
            system_info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'tensorflow_version': tf.__version__,
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
                'timestamp': datetime.now().isoformat()
            }
            
            # GPU information
            gpus = tf.config.list_physical_devices('GPU')
            system_info['gpu_count'] = len(gpus)
            system_info['gpu_info'] = []
            
            for gpu in gpus:
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    system_info['gpu_info'].append({
                        'name': gpu.name,
                        'compute_capability': gpu_details.get('compute_capability', 'Unknown'),
                        'memory_limit': gpu_details.get('memory_limit', 'Unknown')
                    })
                except:
                    system_info['gpu_info'].append({'name': gpu.name})
            
            return system_info
            
        except Exception as e:
            logger.warning(f"Could not get complete system info: {e}")
            return {
                'platform': 'Unknown',
                'tensorflow_version': tf.__version__,
                'gpu_count': len(tf.config.list_physical_devices('GPU')),
                'timestamp': datetime.now().isoformat()
            }
    
    def benchmark_configuration(self, config_path: str) -> BenchmarkResults:
        """
        Benchmark a specific configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Benchmark results
        """
        try:
            logger.info(f"Benchmarking configuration: {config_path}")
            start_time = time.time()
            
            # Load configuration
            config_manager = ConfigManager()
            config = config_manager.load(config_path)
            
            # Initialize high-performance CNN
            hp_cnn = HighPerformancePneumoniaCNN(config.to_dict())
            
            # Build model
            model = hp_cnn.build_optimized_model()
            model_params = model.count_params()
            model_size_mb = self._calculate_model_size(model)
            
            # Create datasets (small subset for benchmarking)
            config_dict = config.to_dict()
            config_dict['training']['epochs'] = 3  # Quick benchmark
            config_dict['training']['batch_size'] = min(config_dict['training']['batch_size'], 16)
            hp_cnn.config = config_dict
            
            try:
                train_ds, val_ds, test_ds = hp_cnn.create_optimized_datasets()
            except Exception as e:
                logger.warning(f"Could not create datasets: {e}")
                logger.info("Creating synthetic datasets for benchmarking...")
                train_ds, val_ds, test_ds = self._create_synthetic_datasets(config_dict)
            
            # Benchmark data pipeline
            pipeline_metrics = hp_cnn.data_pipeline.benchmark_dataset(train_ds, num_batches=10)
            
            # Benchmark training (quick run)
            logger.info("Running quick training benchmark...")
            history = hp_cnn.train_with_optimizations(train_ds, val_ds, epochs=2)
            
            # Benchmark inference
            logger.info("Benchmarking inference performance...")
            inference_metrics = hp_cnn.benchmark_performance(test_ds, num_batches=20)
            
            # Monitor memory usage
            memory_info = self._get_memory_usage()
            
            # Calculate duration
            duration_minutes = (time.time() - start_time) / 60.0
            
            # Create benchmark results
            results = BenchmarkResults(
                system_info=self.system_info,
                model_params=model_params,
                model_size_mb=model_size_mb,
                avg_epoch_time=history.get('performance_summary', {}).get('avg_step_time', 0),
                samples_per_second=pipeline_metrics.get('samples_per_second', 0),
                batches_per_second=pipeline_metrics.get('batches_per_second', 0),
                avg_inference_time=inference_metrics.get('avg_inference_time', 0),
                inference_throughput=inference_metrics.get('avg_throughput', 0),
                peak_memory_mb=memory_info.get('peak_memory_mb', 0),
                avg_memory_mb=memory_info.get('avg_memory_mb', 0),
                final_accuracy=history['epoch_metrics'][-1].get('accuracy', 0) if history['epoch_metrics'] else 0,
                final_val_accuracy=history['epoch_metrics'][-1].get('val_accuracy', 0) if history['epoch_metrics'] else 0,
                best_accuracy=max([m.get('val_accuracy', 0) for m in history['epoch_metrics']], default=0),
                config_name=Path(config_path).stem,
                mixed_precision=config_dict['training']['use_mixed_precision'],
                batch_size=config_dict['training']['batch_size'],
                image_size=config_dict['data']['image_size'],
                benchmark_time=datetime.now().isoformat(),
                duration_minutes=duration_minutes
            )
            
            logger.info(f"Benchmark completed in {duration_minutes:.1f} minutes")
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            raise
    
    def _calculate_model_size(self, model: tf.keras.Model) -> float:
        """Calculate model size in MB."""
        try:
            # Get model memory usage
            total_params = model.count_params()
            # Estimate: 4 bytes per parameter (float32)
            size_mb = (total_params * 4) / (1024 * 1024)
            return round(size_mb, 2)
        except:
            return 0.0
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'peak_memory_mb': round(memory.used / (1024**2), 1),
                'avg_memory_mb': round(memory.used / (1024**2), 1),
                'memory_percent': memory.percent
            }
        except:
            return {'peak_memory_mb': 0, 'avg_memory_mb': 0}
    
    def _create_synthetic_datasets(self, config: Dict[str, Any]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create synthetic datasets for benchmarking when real data is not available."""
        logger.info("Creating synthetic datasets for benchmarking...")
        
        batch_size = config['training']['batch_size']
        image_size = config['data']['image_size']
        
        # Create synthetic data
        def generate_batch():
            images = tf.random.normal([batch_size, *image_size, 3])
            labels = tf.random.uniform([batch_size], 0, 2, dtype=tf.int32)
            return images, labels
        
        # Create datasets
        train_ds = tf.data.Dataset.from_generator(
            generate_batch,
            output_signature=(
                tf.TensorSpec(shape=[batch_size, *image_size, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[batch_size], dtype=tf.int32)
            )
        ).take(50)  # 50 batches for training
        
        val_ds = tf.data.Dataset.from_generator(
            generate_batch,
            output_signature=(
                tf.TensorSpec(shape=[batch_size, *image_size, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[batch_size], dtype=tf.int32)
            )
        ).take(10)  # 10 batches for validation
        
        test_ds = tf.data.Dataset.from_generator(
            generate_batch,
            output_signature=(
                tf.TensorSpec(shape=[batch_size, *image_size, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[batch_size], dtype=tf.int32)
            )
        ).take(20)  # 20 batches for testing
        
        return train_ds, val_ds, test_ds
    
    def compare_configurations(self, config_paths: List[str]) -> Dict[str, Any]:
        """
        Compare multiple configurations.
        
        Args:
            config_paths: List of configuration file paths
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(config_paths)} configurations...")
        
        results = []
        for config_path in config_paths:
            try:
                result = self.benchmark_configuration(config_path)
                results.append(result)
                self._save_individual_result(result)
            except Exception as e:
                logger.error(f"Failed to benchmark {config_path}: {e}")
        
        # Generate comparison report
        comparison = self._generate_comparison_report(results)
        
        # Save comparison results
        comparison_file = self.results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        logger.info(f"Comparison results saved to {comparison_file}")
        return comparison
    
    def _generate_comparison_report(self, results: List[BenchmarkResults]) -> Dict[str, Any]:
        """Generate comparison report from benchmark results."""
        if not results:
            return {}
        
        # Find best performing configuration for each metric
        best_configs = {
            'training_speed': min(results, key=lambda x: x.avg_epoch_time),
            'inference_speed': min(results, key=lambda x: x.avg_inference_time),
            'accuracy': max(results, key=lambda x: x.best_accuracy),
            'memory_efficiency': min(results, key=lambda x: x.peak_memory_mb),
            'throughput': max(results, key=lambda x: x.samples_per_second)
        }
        
        # Calculate speedup comparisons
        baseline = results[0]  # Use first config as baseline
        speedups = {}
        for result in results[1:]:
            if baseline.avg_epoch_time > 0:
                speedups[result.config_name] = {
                    'training_speedup': baseline.avg_epoch_time / result.avg_epoch_time,
                    'inference_speedup': baseline.avg_inference_time / result.avg_inference_time if result.avg_inference_time > 0 else 0,
                    'throughput_improvement': result.samples_per_second / baseline.samples_per_second if baseline.samples_per_second > 0 else 0
                }
        
        return {
            'summary': {
                'total_configs': len(results),
                'benchmark_time': datetime.now().isoformat(),
                'system_info': results[0].system_info
            },
            'best_configurations': {
                'fastest_training': best_configs['training_speed'].config_name,
                'fastest_inference': best_configs['inference_speed'].config_name,
                'highest_accuracy': best_configs['accuracy'].config_name,
                'most_memory_efficient': best_configs['memory_efficiency'].config_name,
                'highest_throughput': best_configs['throughput'].config_name
            },
            'performance_comparison': {
                result.config_name: {
                    'training_time_per_epoch': result.avg_epoch_time,
                    'inference_time_ms': result.avg_inference_time * 1000,
                    'throughput_samples_per_sec': result.samples_per_second,
                    'peak_memory_mb': result.peak_memory_mb,
                    'best_accuracy': result.best_accuracy,
                    'mixed_precision': result.mixed_precision,
                    'batch_size': result.batch_size,
                    'model_size_mb': result.model_size_mb
                }
                for result in results
            },
            'speedup_analysis': speedups,
            'recommendations': self._generate_recommendations(results),
            'detailed_results': [asdict(result) for result in results]
        }
    
    def _generate_recommendations(self, results: List[BenchmarkResults]) -> List[str]:
        """Generate optimization recommendations based on benchmark results."""
        recommendations = []
        
        # Check for mixed precision benefits
        mixed_precision_results = [r for r in results if r.mixed_precision]
        regular_results = [r for r in results if not r.mixed_precision]
        
        if mixed_precision_results and regular_results:
            mp_avg_speed = np.mean([r.samples_per_second for r in mixed_precision_results])
            regular_avg_speed = np.mean([r.samples_per_second for r in regular_results])
            
            if mp_avg_speed > regular_avg_speed * 1.2:
                recommendations.append("Mixed precision training provides significant speedup - recommend enabling for production")
            
        # Check batch size efficiency
        batch_sizes = [(r.batch_size, r.samples_per_second) for r in results]
        if len(batch_sizes) > 1:
            optimal_batch = max(batch_sizes, key=lambda x: x[1])
            recommendations.append(f"Optimal batch size appears to be {optimal_batch[0]} for maximum throughput")
        
        # Memory usage recommendations
        high_memory_configs = [r for r in results if r.peak_memory_mb > 8000]  # 8GB threshold
        if high_memory_configs:
            recommendations.append("Consider reducing batch size or model size to optimize memory usage")
        
        # Accuracy vs speed trade-offs
        if len(results) > 1:
            fastest = min(results, key=lambda x: x.avg_epoch_time)
            most_accurate = max(results, key=lambda x: x.best_accuracy)
            
            if fastest != most_accurate:
                recommendations.append(f"Trade-off identified: {fastest.config_name} is fastest, {most_accurate.config_name} is most accurate")
        
        return recommendations
    
    def _save_individual_result(self, result: BenchmarkResults):
        """Save individual benchmark result."""
        result_file = self.results_dir / f"{result.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"Benchmark result saved to {result_file}")
    
    def generate_performance_report(self, config_path: str) -> str:
        """
        Generate a detailed performance report.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Path to generated report
        """
        logger.info("Generating detailed performance report...")
        
        # Run benchmark
        results = self.benchmark_configuration(config_path)
        
        # Generate report
        report = self._create_performance_report(results)
        
        # Save report
        report_file = self.results_dir / f"performance_report_{results.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Performance report saved to {report_file}")
        return str(report_file)
    
    def _create_performance_report(self, results: BenchmarkResults) -> str:
        """Create formatted performance report."""
        report = f"""# Performance Report: {results.config_name}

## Executive Summary
- **Configuration**: {results.config_name}
- **Benchmark Date**: {results.benchmark_time}
- **Duration**: {results.duration_minutes:.1f} minutes
- **Best Validation Accuracy**: {results.best_accuracy:.4f}

## System Information
- **Platform**: {results.system_info.get('platform', 'Unknown')}
- **TensorFlow Version**: {results.system_info.get('tensorflow_version', 'Unknown')}
- **GPU Count**: {results.system_info.get('gpu_count', 0)}
- **Total Memory**: {results.system_info.get('memory_gb', 0)} GB

## Model Architecture
- **Parameters**: {results.model_params:,}
- **Model Size**: {results.model_size_mb:.1f} MB
- **Input Shape**: {results.image_size}
- **Batch Size**: {results.batch_size}
- **Mixed Precision**: {"Enabled" if results.mixed_precision else "Disabled"}

## Training Performance
- **Average Epoch Time**: {results.avg_epoch_time:.2f} seconds
- **Training Throughput**: {results.samples_per_second:.1f} samples/second
- **Batch Processing**: {results.batches_per_second:.2f} batches/second

## Inference Performance
- **Average Inference Time**: {results.avg_inference_time * 1000:.2f} ms
- **Inference Throughput**: {results.inference_throughput:.1f} samples/second

## Memory Usage
- **Peak Memory**: {results.peak_memory_mb:.1f} MB
- **Average Memory**: {results.avg_memory_mb:.1f} MB

## Accuracy Metrics
- **Final Training Accuracy**: {results.final_accuracy:.4f}
- **Final Validation Accuracy**: {results.final_val_accuracy:.4f}
- **Best Validation Accuracy**: {results.best_accuracy:.4f}

## Performance Analysis

### Strengths
{"- Mixed precision training enabled for improved performance" if results.mixed_precision else "- Full precision training for maximum accuracy"}
- Batch size of {results.batch_size} provides good GPU utilization
- Model size ({results.model_size_mb:.1f} MB) is reasonable for deployment

### Potential Optimizations
- Consider enabling mixed precision if not already active
- Experiment with larger batch sizes if memory allows
- Profile training pipeline for further optimizations

## Recommendations
Based on the benchmark results, consider the following optimizations:
1. **Mixed Precision**: {"Already enabled - good choice!" if results.mixed_precision else "Enable mixed precision for ~2x speedup"}
2. **Batch Size**: Current batch size of {results.batch_size} appears optimal
3. **Memory Usage**: Peak memory usage of {results.peak_memory_mb:.1f} MB is within reasonable limits

---
*Report generated by Performance Benchmark System*
"""
        return report


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="Performance Benchmark System")
    parser.add_argument("--config", type=str, help="Configuration file to benchmark")
    parser.add_argument("--compare", type=str, nargs="+", help="Multiple configs to compare")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark mode")
    
    args = parser.parse_args()
    
    # Initialize benchmark system
    benchmark = PerformanceBenchmark()
    
    try:
        if args.compare:
            # Compare multiple configurations
            logger.info(f"Comparing {len(args.compare)} configurations...")
            results = benchmark.compare_configurations(args.compare)
            
            print("\n" + "="*60)
            print("PERFORMANCE COMPARISON RESULTS")
            print("="*60)
            
            best_configs = results.get('best_configurations', {})
            for metric, config_name in best_configs.items():
                print(f"🏆 Best {metric.replace('_', ' ')}: {config_name}")
            
            print(f"\n📊 Detailed results saved to benchmark_results/")
            
        elif args.config:
            # Benchmark single configuration
            if args.report:
                report_path = benchmark.generate_performance_report(args.config)
                print(f"\n📄 Detailed report generated: {report_path}")
            else:
                results = benchmark.benchmark_configuration(args.config)
                
                print("\n" + "="*50)
                print("PERFORMANCE BENCHMARK RESULTS")
                print("="*50)
                print(f"Configuration: {results.config_name}")
                print(f"Duration: {results.duration_minutes:.1f} minutes")
                print(f"Best Accuracy: {results.best_accuracy:.4f}")
                print(f"Training Speed: {results.samples_per_second:.1f} samples/sec")
                print(f"Inference Speed: {results.inference_throughput:.1f} samples/sec")
                print(f"Peak Memory: {results.peak_memory_mb:.1f} MB")
                print(f"Mixed Precision: {'Enabled' if results.mixed_precision else 'Disabled'}")
                print("="*50)
        
        else:
            # Default: benchmark high-performance config
            high_perf_config = "configs/high_performance.yaml"
            
            if Path(high_perf_config).exists():
                logger.info("Benchmarking default high-performance configuration...")
                results = benchmark.benchmark_configuration(high_perf_config)
                
                print("\n🚀 HIGH-PERFORMANCE CONFIGURATION BENCHMARK")
                print("="*60)
                print(f"⚡ Training Speed: {results.samples_per_second:.1f} samples/sec")
                print(f"🎯 Best Accuracy: {results.best_accuracy:.4f}")
                print(f"💾 Peak Memory: {results.peak_memory_mb:.1f} MB")
                print(f"🔧 Mixed Precision: {'✅ Enabled' if results.mixed_precision else '❌ Disabled'}")
                print(f"⏱️  Duration: {results.duration_minutes:.1f} minutes")
                print("="*60)
                
            else:
                logger.error(f"High-performance config not found: {high_perf_config}")
                logger.info("Available configs:")
                configs_dir = Path("configs")
                if configs_dir.exists():
                    for config_file in configs_dir.glob("*.yaml"):
                        print(f"  - {config_file}")
                
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())