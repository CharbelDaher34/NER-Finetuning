# Modular Training Architecture

This document explains the refactored, modular training architecture that makes it easy to adapt the training pipeline for different tasks.

## Overview

The refactored architecture separates concerns into focused, reusable components:

```
┌─────────────────────────────────────────────────────────────┐
│                    train_refactored.py                      │
│                    (Entry Point)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              TrainerOrchestrator                            │
│  (Coordinates the entire training pipeline)                 │
└──┬────────────────┬─────────────────┬───────────────────┬──┘
   │                │                 │                   │
   ▼                ▼                 ▼                   ▼
┌────────┐   ┌──────────────┐  ┌──────────┐   ┌──────────────┐
│ Config │   │DataProcessor │  │Evaluator │   │   Metrics    │
│(Pydantic)  │  (Abstract)  │  │(Abstract)│   │  (Utilities) │
└────────┘   └──────┬───────┘  └─────┬────┘   └──────────────┘
                    │                 │
                    ▼                 ▼
             ┌──────────────┐  ┌──────────────┐
             │ NERProcessor │  │NERTaskEvaluator│
             │(Concrete Impl)  │(Concrete Impl)│
             └──────────────┘  └──────────────┘
```

## Core Components

### 1. Configuration (`config.py`)

Uses **Pydantic** for type-safe, validated configuration:

- `ExperimentConfig`: Top-level config containing all sub-configs
- `ModelConfig`: Model settings (base model, dtype, attention)
- `DatasetConfig`: Dataset paths and preprocessing settings
- `LoRAConfig`: LoRA adapter parameters
- `TrainingConfig`: Training hyperparameters
- `MonitoringConfig`: Logging, checkpointing, early stopping
- `EvaluationConfig`: Evaluation settings (fuzzy matching, etc.)
- `GenerationConfig`: Text generation parameters

**Benefits:**
- Type safety and validation
- Easy to serialize/deserialize
- Clear documentation of all parameters
- IDE autocomplete support

### 2. Data Processing (`data_processor.py`)

Abstract base class `DataProcessor` defines the interface:

```python
class DataProcessor(ABC):
    @abstractmethod
    def parse_example(self, example) -> List[Dict]:
        """Parse raw example into processable format"""
        
    @abstractmethod
    def format_for_training(self, parsed_data) -> str:
        """Format for training"""
        
    @abstractmethod
    def format_for_inference(self, input_data) -> str:
        """Format for inference"""
```

**To add a new task:** Subclass `DataProcessor` and implement these methods.

### 3. Evaluation (`evaluator.py`)

Abstract base class `Evaluator` defines the interface:

```python
class Evaluator(ABC):
    @abstractmethod
    def parse_dataset_example(self, example) -> Tuple:
        """Parse test example"""
        
    @abstractmethod
    def generate_prediction(self, model, context, question) -> Tuple:
        """Generate model prediction"""
        
    @abstractmethod
    def calculate_metrics(self, predicted, ground_truth) -> Dict:
        """Calculate task-specific metrics"""
        
    @abstractmethod
    def is_valid_prediction(self, prediction) -> bool:
        """Validate prediction format"""
        
    @abstractmethod
    def is_schema_valid(self, prediction) -> bool:
        """Validate prediction schema"""
```

The base `Evaluator` handles:
- Evaluation loop
- Progress logging
- Memory management
- Result aggregation

**To add a new task:** Subclass `Evaluator` and implement the abstract methods.

### 4. Metrics (`metrics.py`)

Reusable utility functions:
- `extract_json_from_text()`: Robust JSON extraction
- `calculate_fuzzy_metrics()`: Fuzzy matching with rapidfuzz
- `normalize_dict_values_to_lists()`: Data normalization

### 5. Orchestrator (`trainer_orchestrator.py`)

The `TrainerOrchestrator` class coordinates the entire pipeline:

1. **Setup**: Environment, logging, authentication, random seeds
2. **Loading**: Model, tokenizer, datasets
3. **Preparation**: LoRA configuration, trainer setup
4. **Evaluation**: Pre-training and post-training
5. **Training**: Main training loop
6. **Saving**: Results, metrics, model adapters

**Key features:**
- Handles all boilerplate
- Memory management
- Error handling and logging
- W&B integration
- Automatic result organization

## Task-Specific Implementations

### NER Task (`tasks/ner_task.py`)

Contains concrete implementations for the NER task:

- `NERDataProcessor`: Parses multi-turn conversations, formats for training/inference
- `NERTaskEvaluator`: Evaluates JSON entity extraction with fuzzy matching

## How to Add a New Task

### Step 1: Create Task-Specific Processor

```python
# tasks/my_task.py
from data_processor import DataProcessor

class MyTaskProcessor(DataProcessor):
    def parse_example(self, example):
        # Parse your data format
        return [{"input": ..., "output": ...}]
    
    def format_for_training(self, parsed_data):
        # Format for training
        messages = [...]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)
    
    def format_for_inference(self, input_data):
        # Format for inference
        return "Your prompt here"
```

### Step 2: Create Task-Specific Evaluator

```python
from evaluator import Evaluator

class MyTaskEvaluator(Evaluator):
    def parse_dataset_example(self, example):
        # Parse test example
        return context, qa_pairs
    
    def generate_prediction(self, model, context, question):
        # Generate prediction
        return response_text, parsed_output
    
    def calculate_metrics(self, predicted, ground_truth):
        # Calculate your metrics (must return tp, fp, fn)
        return {"tp": ..., "fp": ..., "fn": ...}
    
    def is_valid_prediction(self, prediction):
        # Validate format
        return True/False
    
    def is_schema_valid(self, prediction):
        # Validate schema
        return True/False
    
    def get_empty_prediction(self):
        # Return fallback prediction
        return {}
```

### Step 3: Create Entry Point

```python
# train_my_task.py
from config import ExperimentConfig
from tasks.my_task import MyTaskProcessor, MyTaskEvaluator
from trainer_orchestrator import TrainerOrchestrator

def main():
    config = ExperimentConfig()
    config.model.base_model = "my-model"
    # ... configure other settings
    
    processor = MyTaskProcessor(tokenizer=None)
    evaluator = MyTaskEvaluator(config, None, None, processor)
    
    orchestrator = TrainerOrchestrator(config, processor, evaluator)
    orchestrator.run()

if __name__ == "__main__":
    main()
```

## Configuration Examples

### Customize Training

```python
config = ExperimentConfig()

# Model settings
config.model.base_model = "meta-llama/Llama-3.2-1B"
config.model.torch_dtype = "bfloat16"

# Training settings
config.training.num_train_epochs = 10
config.training.learning_rate = 1e-4
config.training.per_device_train_batch_size = 4

# LoRA settings
config.lora.r = 32
config.lora.lora_alpha = 64
```

### Load from JSON

```python
import json
from config import ExperimentConfig

with open("my_config.json") as f:
    config_dict = json.load(f)

config = ExperimentConfig(**config_dict)
```

### Save Configuration

```python
# Save as JSON
with open("config.json", "w") as f:
    json.dump(config.model_dump(), f, indent=2)

# Or use Pydantic's built-in serialization
json_str = config.model_dump_json(indent=2)
```

## Benefits of This Architecture

### 1. **Separation of Concerns**
Each component has a single, well-defined responsibility.

### 2. **Reusability**
Abstract base classes can be reused across different tasks.

### 3. **Extensibility**
Easy to add new tasks without modifying core infrastructure.

### 4. **Type Safety**
Pydantic provides runtime validation and type checking.

### 5. **Testability**
Each component can be unit tested independently.

### 6. **Maintainability**
Clear structure makes code easier to understand and modify.

### 7. **Configurability**
All parameters centralized in Pydantic models.

## File Structure

```
NER-Finetuning/
├── config.py                    # Pydantic configuration models
├── data_processor.py            # Abstract data processor
├── evaluator.py                 # Abstract evaluator
├── metrics.py                   # Reusable utility functions
├── trainer_orchestrator.py      # Training orchestrator
├── tasks/
│   ├── __init__.py
│   ├── ner_task.py              # NER-specific implementations
│   └── your_task.py             # Your custom task
├── train_refactored.py          # Entry point for NER task
├── train.py                     # Original monolithic script
├── ARCHITECTURE.md              # This file
└── TASK_EXAMPLE.md              # Step-by-step task creation example
```

## Migration from Original Script

The original `train.py` has been refactored into:

| Original Code | New Location |
|--------------|--------------|
| Configuration constants | `config.py` (Pydantic models) |
| Parsing functions | `tasks/ner_task.py` (NERDataProcessor) |
| Inference function | `tasks/ner_task.py` (NERTaskEvaluator) |
| Metrics calculation | `metrics.py` + `tasks/ner_task.py` |
| Evaluation loop | `evaluator.py` (base class) |
| Training setup | `trainer_orchestrator.py` |
| Main execution | `train_refactored.py` |

The original script is preserved for reference.

## Running the Refactored Code

```bash
# Using the refactored architecture
uv run train_refactored.py

# Original script still works
uv run train.py
```

Both scripts produce equivalent results, but the refactored version is much more maintainable and extensible.

