# Evaluation Pipeline Testing Guide

This guide explains how to test your evaluation pipeline to ensure it works correctly before running it on your full dataset.

## 🚀 Quick Start

### 1. Run Smoke Tests (Recommended First Step)

The smoke tests quickly verify that all components can be loaded and initialized:

```bash
python scripts/test_pipeline_smoke.py
```

This will test:
- ✅ Module imports
- ✅ Environment setup
- ✅ Configuration loading
- ✅ Model initialization (mocked)
- ✅ EvaluationRunner setup

### 2. Run Unit Tests

Test individual components in isolation:

```bash
# Run all unit tests
python tests/run_evaluation_tests.py unit

# Or run specific test files
python -m pytest tests/core/evaluation/test_evaluation_runner.py -v
```

### 3. Run Integration Tests

Test the complete pipeline with real configurations:

```bash
# Run all integration tests
python tests/run_evaluation_tests.py integration

# Or run specific integration tests
python -m pytest tests/core/evaluation/test_integration.py -v
```

### 4. Run Script Tests

Test the main evaluation script:

```bash
# Run script tests
python tests/run_evaluation_tests.py script
```

### 5. Run All Tests

Run the complete test suite:

```bash
# Run all evaluation tests
python tests/run_evaluation_tests.py all
```

## 📋 Test Categories

### Unit Tests (`test_evaluation_runner.py`)

These tests verify individual components work correctly:

- **Initialization**: Tests that `EvaluationRunner` can be created
- **Model Integration**: Tests that LLM and LMv3 models are called correctly
- **Data Processing**: Tests dataset filtering and processing
- **Error Handling**: Tests graceful handling of errors
- **Metrics Calculation**: Tests that evaluation metrics are computed

### Integration Tests (`test_integration.py`)

These tests verify the complete pipeline works with real configurations:

- **Configuration Loading**: Tests that real config files can be loaded
- **Model Initialization**: Tests that models can be initialized with real configs
- **Complete Flow**: Tests the entire evaluation process
- **Environment Setup**: Tests that the environment is properly configured

### Script Tests (`test_evaluate_script.py`)

These tests verify the main evaluation script works:

- **Main Function**: Tests the main function flow
- **Dataset Filtering**: Tests schematism filtering
- **Error Handling**: Tests script-level error handling
- **Configuration Usage**: Tests that configs are used correctly

## 🔧 Test Configuration

### Test Configuration Files

The tests use dedicated test configuration files:

- **Dataset**: `configs/dataset/tests_dataset_config.yaml`
- **LLM Model**: `configs/models/llm/tests_llm_config.yaml`
- **LMv3 Model**: `configs/models/lmv3/tests_lmv3_config.yaml`

These configs are designed for testing and use:
- Smaller models for faster testing
- CPU-only execution
- Minimal dataset sizes
- Mock API endpoints where possible

### Environment Setup

Tests require:

1. **Environment Variables**: Loaded from `.env` file
2. **Python Path**: `src/` directory in Python path
3. **Dependencies**: All required packages installed
4. **Configs Directory**: Configuration files in place

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when running tests

**Solution**:
```bash
# Make sure you're in the project root
cd /path/to/your/project

# Install the package in development mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. Configuration Errors

**Problem**: Tests fail to load configurations

**Solution**:
```bash
# Check that config files exist
ls configs/dataset/
ls configs/models/llm/
ls configs/models/lmv3/

# Verify config file syntax
python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/dataset/tests_dataset_config.yaml')"
```

#### 3. Model Loading Errors

**Problem**: Tests fail when trying to load models

**Solution**:
- Tests use mocking to avoid loading actual models
- If you see model loading errors, check that the mocking is working correctly
- Ensure test configs point to valid model checkpoints

#### 4. Dataset Loading Errors

**Problem**: Tests fail to load datasets

**Solution**:
```bash
# Check HuggingFace token
echo $HF_TOKEN

# Test dataset loading manually
python -c "
from core.data.utils import get_dataset
from core.config.manager import ConfigManager
from core.config.constants import ConfigType, DatasetConfigSubtype
from core.utils.shared import CONFIGS_DIR

config_manager = ConfigManager(CONFIGS_DIR)
dataset_config = config_manager.load_config(
    ConfigType.DATASET,
    DatasetConfigSubtype.EVALUATION,
    'schematism_dataset_config'
)
dataset = get_dataset(dataset_config)
print(f'Dataset loaded: {len(dataset)} samples')
"
```

## 🧪 Manual Testing

### Test Individual Components

#### 1. Test Configuration Loading

```python
from core.config.manager import ConfigManager
from core.config.constants import ConfigType, DatasetConfigSubtype, ModelsConfigSubtype
from core.utils.shared import CONFIGS_DIR

config_manager = ConfigManager(CONFIGS_DIR)

# Test dataset config
dataset_config = config_manager.load_config(
    ConfigType.DATASET,
    DatasetConfigSubtype.EVALUATION,
    "schematism_dataset_config"
)
print("Dataset config loaded successfully")

# Test model configs
llm_config = config_manager.load_config(
    ConfigType.MODELS,
    ModelsConfigSubtype.LLM,
    "tests_llm_config"
)
print("LLM config loaded successfully")
```

#### 2. Test Model Initialization

```python
from core.models.llm.model import LLMModel
from core.models.lmv3.model import LMv3Model

# Initialize models (this will use test configs)
llm_model = LLMModel(llm_config)
lmv3_model = LMv3Model(lmv3_config)
print("Models initialized successfully")
```

#### 3. Test Dataset Loading

```python
from core.data.utils import get_dataset

dataset = get_dataset(dataset_config, wrapper=False)
print(f"Dataset loaded: {len(dataset)} samples")

# Test filtering
from core.data.filters import filter_schematisms
filter_func = filter_schematisms(to_filter="wloclawek_1872")
filtered_dataset = dataset.filter(filter_func, input_columns=["schematism_name"])
print(f"Filtered dataset: {len(filtered_dataset)} samples")
```

#### 4. Test Evaluation Runner

```python
from core.evaluation.runner import EvaluationRunner

evaluation_runner = EvaluationRunner(
    llm_model=llm_model,
    lmv3_model=lmv3_model,
    dataset_config=dataset_config
)

# Test with a small subset
test_subset = filtered_dataset.select(range(min(5, len(filtered_dataset))))
evaluation_runner.run(test_subset)
print("Evaluation runner test completed")
```

## 📊 Test Coverage

The test suite covers:

- ✅ **Configuration Management**: Loading and validation of configs
- ✅ **Model Initialization**: LLM and LMv3 model setup
- ✅ **Dataset Processing**: Loading, filtering, and preprocessing
- ✅ **Evaluation Logic**: Metrics calculation and result processing
- ✅ **Error Handling**: Graceful failure handling
- ✅ **Integration**: End-to-end pipeline testing
- ✅ **Script Execution**: Main evaluation script testing

## 🚀 Production Testing

Before running on your full dataset:

1. **Run Smoke Tests**: `python scripts/test_pipeline_smoke.py`
2. **Run Unit Tests**: `python tests/run_evaluation_tests.py unit`
3. **Run Integration Tests**: `python tests/run_evaluation_tests.py integration`
4. **Test with Small Dataset**: Run evaluation on a small subset first
5. **Monitor Resources**: Check memory and compute usage
6. **Verify Results**: Check that metrics make sense

## 📝 Adding New Tests

### Adding Unit Tests

1. Create test file: `tests/core/evaluation/test_new_component.py`
2. Follow existing patterns from `test_evaluation_runner.py`
3. Use fixtures for common setup
4. Mock external dependencies

### Adding Integration Tests

1. Add to `test_integration.py`
2. Use real configurations
3. Test complete workflows
4. Verify end-to-end functionality

### Running Specific Tests

```bash
# Run specific test class
python -m pytest tests/core/evaluation/test_evaluation_runner.py::TestEvaluationRunner -v

# Run specific test method
python -m pytest tests/core/evaluation/test_evaluation_runner.py::TestEvaluationRunner::test_initialization -v

# Run tests matching pattern
python -m pytest -k "test_initialization" -v
```

## 🔍 Debugging Tests

### Verbose Output

```bash
python -m pytest -v -s --tb=long
```

### Debug Specific Test

```bash
# Add breakpoint in test
import pdb; pdb.set_trace()

# Run with debugger
python -m pytest --pdb
```

### Check Test Discovery

```bash
# List all tests
python -m pytest --collect-only

# Show test structure
python -m pytest --collect-only -q
```

## 📈 Performance Testing

For performance testing:

1. **Memory Usage**: Monitor memory consumption during evaluation
2. **Processing Speed**: Measure time per sample
3. **Scalability**: Test with different dataset sizes
4. **Resource Limits**: Test with limited CPU/memory

```python
import time
import psutil
import os

# Monitor memory usage
process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss / 1024 / 1024  # MB

# Run evaluation
start_time = time.time()
evaluation_runner.run(dataset_subset)
end_time = time.time()

end_memory = process.memory_info().rss / 1024 / 1024  # MB

print(f"Time: {end_time - start_time:.2f}s")
print(f"Memory: {end_memory - start_memory:.2f}MB")
```

This comprehensive testing approach ensures your evaluation pipeline is robust and ready for production use. 