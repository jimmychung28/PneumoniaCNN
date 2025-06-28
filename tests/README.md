# PneumoniaCNN Test Suite

Comprehensive test suite for the PneumoniaCNN project, including unit tests, integration tests, and performance benchmarks.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_config.py      # Configuration system tests
│   ├── test_validation.py  # Validation utilities tests
│   └── test_data_pipeline.py # Data pipeline tests
├── integration/            # Integration tests for workflows
│   └── test_training_pipeline.py # End-to-end training tests
├── performance/            # Performance and benchmark tests
│   └── test_performance.py # Performance measurements
├── conftest.py            # Pytest configuration and fixtures
├── utils.py               # Test utilities and helpers
└── README.md              # This file
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r requirements.txt  # or requirements_apple_silicon.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Using the Test Runner

```bash
# Run unit tests
python run_tests.py unit

# Run integration tests
python run_tests.py integration

# Run performance tests
python run_tests.py performance

# Run all tests with coverage
python run_tests.py coverage

# Run quick tests (exclude slow/GPU tests)
python run_tests.py quick

# List all available tests
python run_tests.py list

# Run specific test file
python run_tests.py unit -t test_config.py

# Run tests with specific marker
python run_tests.py marker gpu
```

### Direct pytest Usage

```bash
# Run specific test file
pytest tests/unit/test_config.py

# Run tests matching pattern
pytest -k "config"

# Run with specific markers
pytest -m "unit and not slow"

# Run in parallel (faster)
pytest -n auto

# Run with verbose output
pytest -v

# Re-run failed tests
pytest --lf
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, test workflows)
- `@pytest.mark.performance` - Performance tests (measure execution time/memory)
- `@pytest.mark.slow` - Tests that take more than a few seconds
- `@pytest.mark.gpu` - Tests that require GPU
- `@pytest.mark.requires_data` - Tests that need the chest X-ray dataset

## Writing Tests

### Unit Test Example

```python
import pytest
from src.config.config_schema import ModelConfig

class TestModelConfig:
    @pytest.mark.unit
    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = ModelConfig()
        assert config.architecture == "standard"
        assert config.learning_rate == 0.001
```

### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.slow
def test_training_pipeline(test_data_dir, sample_config_dict):
    """Test complete training pipeline."""
    config = load_config(sample_config_dict)
    model = PneumoniaCNN(config=config)
    history = model.train()
    assert history is not None
```

### Performance Test Example

```python
@pytest.mark.performance
def test_data_loading_speed(test_data_dir):
    """Test data loading performance."""
    with measure_performance() as metrics:
        dataset = create_data_pipeline(data_dir=test_data_dir)
        for _ in dataset.take(100):
            pass
    assert metrics['execution_time'] < 10.0
```

## Fixtures

Common fixtures available in `conftest.py`:

- `test_data_dir` - Temporary directory with test dataset structure
- `sample_config_dict` - Sample configuration dictionary
- `temp_config_file` - Temporary YAML config file
- `sample_images` - Synthetic normal/pneumonia images
- `mock_model` - Simple mock model for testing
- `disable_gpu` - Temporarily disable GPU

## Coverage Reports

After running tests with coverage:

```bash
# View coverage in terminal
pytest --cov=src --cov-report=term-missing

# Generate HTML report
pytest --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=src --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Performance Testing

Performance tests measure:
- Data loading speed
- Model training time
- Inference speed
- Memory usage
- Configuration loading time

Results are printed and can be saved for tracking:

```bash
# Run performance tests with detailed output
pytest tests/performance -v -s

# Save results to file
pytest tests/performance --json-report --json-report-file=perf_results.json
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the project root and have installed dependencies
2. **GPU tests failing**: Use `--no-gpu` or mark tests with `@pytest.mark.gpu`
3. **Slow tests**: Use `-m "not slow"` to skip slow tests during development
4. **Missing test data**: Tests create synthetic data automatically

### Debug Mode

```bash
# Run with debugging output
pytest -vv -s --log-cli-level=DEBUG

# Run with pdb on failure
pytest --pdb

# Run specific test with debugging
pytest tests/unit/test_config.py::TestConfigSchema::test_model_config_defaults -vv
```

## Best Practices

1. **Keep tests fast**: Unit tests should run in < 1 second
2. **Use fixtures**: Share test data and setup code
3. **Test edge cases**: Include tests for error conditions
4. **Mock external dependencies**: Don't rely on external services
5. **Use meaningful names**: Test names should describe what they test
6. **Assert specific values**: Avoid vague assertions
7. **Clean up resources**: Use context managers and cleanup fixtures

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Add appropriate markers
5. Update this README if needed