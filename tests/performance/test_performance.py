"""
Performance tests for critical components.
"""
import pytest
import time
import psutil
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
from contextlib import contextmanager
from typing import Dict, Any, Generator
import gc

from src.models.cnn import PneumoniaCNN
from src.training.data_pipeline import create_data_pipeline
from src.config.config_loader import ConfigLoader


@contextmanager
def measure_performance() -> Generator[Dict[str, Any], None, None]:
    """Context manager to measure performance metrics."""
    # Force garbage collection before measurement
    gc.collect()
    
    # Get initial metrics
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    metrics = {}
    yield metrics
    
    # Get final metrics
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    metrics['execution_time'] = end_time - start_time
    metrics['memory_used'] = end_memory - start_memory
    metrics['peak_memory'] = max(end_memory, start_memory)
    
    # Force garbage collection after measurement
    gc.collect()


class TestDataPipelinePerformance:
    """Test data pipeline performance."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_data_loading_speed(self, test_data_dir):
        """Test data loading performance."""
        batch_sizes = [16, 32, 64]
        results = []
        
        for batch_size in batch_sizes:
            with measure_performance() as metrics:
                dataset = create_data_pipeline(
                    data_dir=str(test_data_dir / "train"),
                    batch_size=batch_size,
                    image_size=(128, 128),
                    augment=False,
                    cache=True,
                    prefetch=True
                )
                
                # Process 10 batches
                count = 0
                for images, labels in dataset.take(10):
                    count += 1
                    assert images.shape[0] == batch_size
                
                metrics['batch_size'] = batch_size
                metrics['batches_processed'] = count
                metrics['images_per_second'] = (count * batch_size) / metrics['execution_time']
            
            results.append(metrics)
        
        # Verify performance improves with optimization
        for result in results:
            print(f"Batch size: {result['batch_size']}")
            print(f"  Time: {result['execution_time']:.2f}s")
            print(f"  Memory: {result['memory_used']:.2f}MB")
            print(f"  Images/sec: {result['images_per_second']:.2f}")
            
            # Basic performance assertions
            assert result['execution_time'] < 10  # Should process 10 batches in < 10s
            assert result['images_per_second'] > 10  # Should process > 10 images/sec
    
    @pytest.mark.performance
    def test_augmentation_performance(self, test_data_dir):
        """Test performance impact of augmentation."""
        results = {}
        
        # Test without augmentation
        with measure_performance() as metrics:
            dataset = create_data_pipeline(
                data_dir=str(test_data_dir / "train"),
                batch_size=32,
                image_size=(128, 128),
                augment=False
            )
            
            for _ in dataset.take(20):
                pass
            
            results['no_augmentation'] = metrics.copy()
        
        # Test with augmentation
        with measure_performance() as metrics:
            dataset = create_data_pipeline(
                data_dir=str(test_data_dir / "train"),
                batch_size=32,
                image_size=(128, 128),
                augment=True
            )
            
            for _ in dataset.take(20):
                pass
            
            results['with_augmentation'] = metrics.copy()
        
        # Augmentation should add some overhead but not too much
        time_increase = (results['with_augmentation']['execution_time'] / 
                        results['no_augmentation']['execution_time'])
        
        print(f"Augmentation time increase: {time_increase:.2f}x")
        assert time_increase < 3.0  # Should be less than 3x slower
    
    @pytest.mark.performance
    def test_caching_performance(self, test_data_dir):
        """Test performance improvement from caching."""
        # First pass without cache
        with measure_performance() as metrics_no_cache:
            dataset = create_data_pipeline(
                data_dir=str(test_data_dir / "train"),
                batch_size=32,
                image_size=(128, 128),
                cache=False
            )
            
            # Process dataset twice
            for _ in range(2):
                for _ in dataset.take(10):
                    pass
        
        # Second pass with cache
        with measure_performance() as metrics_with_cache:
            dataset = create_data_pipeline(
                data_dir=str(test_data_dir / "train"),
                batch_size=32,
                image_size=(128, 128),
                cache=True
            )
            
            # Process dataset twice
            for _ in range(2):
                for _ in dataset.take(10):
                    pass
        
        # Caching should improve performance on second iteration
        print(f"Without cache: {metrics_no_cache['execution_time']:.2f}s")
        print(f"With cache: {metrics_with_cache['execution_time']:.2f}s")
        
        # With small test data, difference might be minimal
        assert metrics_with_cache['execution_time'] <= metrics_no_cache['execution_time'] * 1.1


class TestModelPerformance:
    """Test model training and inference performance."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_model_training_speed(self, test_data_dir, sample_config_dict):
        """Test model training performance."""
        # Configure for performance test
        sample_config_dict['training']['epochs'] = 1
        sample_config_dict['training']['batch_size'] = 32
        sample_config_dict['data']['train_dir'] = str(test_data_dir / "train")
        sample_config_dict['data']['image_size'] = [128, 128]
        
        results = {}
        
        # Test standard mode
        with measure_performance() as metrics:
            loader = ConfigLoader()
            config = loader.load_from_dict(sample_config_dict)
            
            model = PneumoniaCNN(config=config, mode='standard')
            history = model.train()
            
            metrics['mode'] = 'standard'
            metrics['steps'] = len(history.history['loss'])
            results['standard'] = metrics.copy()
        
        # Clear session
        tf.keras.backend.clear_session()
        
        # Test with mixed precision if available
        if tf.config.list_physical_devices('GPU'):
            sample_config_dict['hardware']['mixed_precision'] = True
            
            with measure_performance() as metrics:
                config = loader.load_from_dict(sample_config_dict)
                
                model = PneumoniaCNN(config=config, mode='high_performance')
                history = model.train()
                
                metrics['mode'] = 'high_performance'
                metrics['steps'] = len(history.history['loss'])
                results['high_performance'] = metrics.copy()
        
        # Report results
        for mode, metrics in results.items():
            print(f"\n{mode} mode:")
            print(f"  Training time: {metrics['execution_time']:.2f}s")
            print(f"  Memory used: {metrics['memory_used']:.2f}MB")
            print(f"  Time per step: {metrics['execution_time']/metrics['steps']:.2f}s")
    
    @pytest.mark.performance
    def test_model_inference_speed(self, mock_model):
        """Test model inference performance."""
        batch_sizes = [1, 16, 32]
        results = []
        
        for batch_size in batch_sizes:
            # Create test data
            test_data = np.random.rand(batch_size, 128, 128, 3).astype(np.float32)
            
            # Warm up
            _ = mock_model.predict(test_data, verbose=0)
            
            # Measure inference time
            with measure_performance() as metrics:
                predictions = mock_model.predict(test_data, verbose=0)
                
                metrics['batch_size'] = batch_size
                metrics['inference_time_per_image'] = metrics['execution_time'] / batch_size
                
            results.append(metrics)
            
            assert predictions.shape == (batch_size, 1)
        
        # Report results
        for result in results:
            print(f"\nBatch size: {result['batch_size']}")
            print(f"  Total time: {result['execution_time']*1000:.2f}ms")
            print(f"  Time per image: {result['inference_time_per_image']*1000:.2f}ms")
            
            # Basic performance assertion
            assert result['inference_time_per_image'] < 1.0  # < 1s per image
    
    @pytest.mark.performance
    def test_model_memory_usage(self, sample_config_dict):
        """Test model memory consumption."""
        sample_config_dict['model']['input_shape'] = [128, 128, 3]
        
        loader = ConfigLoader()
        config = loader.load_from_dict(sample_config_dict)
        
        # Measure memory before model creation
        gc.collect()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model
        model = PneumoniaCNN(config=config, mode='standard')
        model.build_model()
        
        # Measure memory after model creation
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        model_memory = memory_after - memory_before
        
        print(f"\nModel memory usage: {model_memory:.2f}MB")
        print(f"Total parameters: {model.model.count_params():,}")
        
        # Basic assertion - model shouldn't use excessive memory
        assert model_memory < 500  # Should be less than 500MB for standard CNN


class TestConfigurationPerformance:
    """Test configuration system performance."""
    
    @pytest.mark.performance
    def test_config_loading_speed(self, sample_config_dict):
        """Test configuration loading performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple config files
            config_files = []
            for i in range(10):
                config_path = Path(tmpdir) / f"config_{i}.yaml"
                with open(config_path, 'w') as f:
                    import yaml
                    yaml.dump(sample_config_dict, f)
                config_files.append(config_path)
            
            # Measure loading time
            with measure_performance() as metrics:
                loader = ConfigLoader()
                configs = []
                
                for config_file in config_files:
                    config = loader.load_config(str(config_file))
                    configs.append(config)
                
                metrics['configs_loaded'] = len(configs)
                metrics['time_per_config'] = metrics['execution_time'] / len(configs)
            
            print(f"\nConfig loading performance:")
            print(f"  Total time: {metrics['execution_time']*1000:.2f}ms")
            print(f"  Time per config: {metrics['time_per_config']*1000:.2f}ms")
            
            # Config loading should be fast
            assert metrics['time_per_config'] < 0.1  # < 100ms per config
    
    @pytest.mark.performance
    def test_config_validation_speed(self, sample_config_dict):
        """Test configuration validation performance."""
        loader = ConfigLoader()
        
        # Create variations of config
        configs = []
        for i in range(100):
            config = sample_config_dict.copy()
            config['experiment_name'] = f"test_{i}"
            config['model']['learning_rate'] = 0.001 * (i + 1)
            configs.append(config)
        
        # Measure validation time
        with measure_performance() as metrics:
            for config in configs:
                loader.validate_config(config)
            
            metrics['configs_validated'] = len(configs)
            metrics['time_per_validation'] = metrics['execution_time'] / len(configs)
        
        print(f"\nConfig validation performance:")
        print(f"  Total time: {metrics['execution_time']*1000:.2f}ms")
        print(f"  Time per validation: {metrics['time_per_validation']*1000:.2f}ms")
        
        # Validation should be very fast
        assert metrics['time_per_validation'] < 0.01  # < 10ms per validation


@pytest.mark.performance
def test_memory_leak_detection(test_data_dir):
    """Test for memory leaks in data pipeline."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run multiple iterations
    for i in range(5):
        dataset = create_data_pipeline(
            data_dir=str(test_data_dir / "train"),
            batch_size=32,
            image_size=(128, 128)
        )
        
        # Process some batches
        for _ in dataset.take(10):
            pass
        
        # Clear dataset
        del dataset
        gc.collect()
        
        # Check memory
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        print(f"Iteration {i+1}: Memory increase: {memory_increase:.2f}MB")
    
    # Memory increase should be minimal after garbage collection
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    total_increase = final_memory - initial_memory
    
    print(f"\nTotal memory increase: {total_increase:.2f}MB")
    assert total_increase < 100  # Should not leak more than 100MB