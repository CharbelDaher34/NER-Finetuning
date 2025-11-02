# Modular Training Framework

A clean, extensible training framework for fine-tuning language models on any task.

## Quick Start

### Run the NER Task (Original Task)

```bash
# Install dependencies (if not already installed)
uv sync

# Run refactored version
uv run train_refactored.py

# Or run original version
uv run train.py
```

Both produce equivalent results, but the refactored version is modular and extensible.

## Architecture

The framework is built on three key abstractions:

1. **DataProcessor**: Handles task-specific data parsing and formatting
2. **Evaluator**: Handles task-specific evaluation and metrics
3. **TrainerOrchestrator**: Coordinates the entire pipeline

```python
from config import ExperimentConfig
from tasks.your_task import YourDataProcessor, YourEvaluator
from trainer_orchestrator import TrainerOrchestrator

# Configure
config = ExperimentConfig()
config.model.base_model = "your-model"

# Setup task
processor = YourDataProcessor(tokenizer=None)
evaluator = YourEvaluator(config, None, None, processor)

# Run
orchestrator = TrainerOrchestrator(config, processor, evaluator)
orchestrator.run()
```

## Features

✅ **Pydantic Configuration** - Type-safe, validated configuration  
✅ **Abstract Base Classes** - Easy to extend for new tasks  
✅ **Automatic Evaluation** - Pre and post-training metrics  
✅ **W&B Integration** - Automatic logging and tracking  
✅ **Memory Efficient** - Gradient checkpointing, 8-bit optimizer  
✅ **LoRA Fine-Tuning** - Parameter-efficient training  
✅ **Early Stopping** - Prevent overfitting  
✅ **Fuzzy Matching** - Robust entity evaluation (NER task)  

## Project Structure

```
NER-Finetuning/
├── config.py                   # Pydantic configuration models
├── data_processor.py           # Abstract data processor
├── evaluator.py                # Abstract evaluator
├── metrics.py                  # Reusable metrics utilities
├── trainer_orchestrator.py     # Training coordinator
├── tasks/
│   ├── ner_task.py            # NER task implementation
│   └── your_task.py           # Your custom task
├── train_refactored.py        # Entry point (refactored)
├── train.py                   # Entry point (original)
├── ARCHITECTURE.md            # Detailed architecture guide
└── TASK_EXAMPLE.md            # Step-by-step task creation example
```

## Creating a New Task

See [TASK_EXAMPLE.md](TASK_EXAMPLE.md) for a complete walkthrough.

**TL;DR:**

```python
# 1. Implement DataProcessor
class MyDataProcessor(DataProcessor):
    def parse_example(self, example): ...
    def format_for_training(self, data): ...
    def format_for_inference(self, data): ...

# 2. Implement Evaluator
class MyEvaluator(Evaluator):
    def parse_dataset_example(self, example): ...
    def generate_prediction(self, model, context, question): ...
    def calculate_metrics(self, pred, gt): ...
    def is_valid_prediction(self, pred): ...
    def is_schema_valid(self, pred): ...
    def get_empty_prediction(self): ...

# 3. Create entry point
config = ExperimentConfig()
processor = MyDataProcessor(tokenizer=None)
evaluator = MyEvaluator(config, None, None, processor)
orchestrator = TrainerOrchestrator(config, processor, evaluator)
orchestrator.run()
```

## Configuration

All parameters are configured via Pydantic models:

```python
from config import ExperimentConfig

config = ExperimentConfig()

# Model
config.model.base_model = "Qwen/Qwen3-0.6B"
config.model.torch_dtype = "bfloat16"

# Training
config.training.num_train_epochs = 4
config.training.learning_rate = 5e-5
config.training.per_device_train_batch_size = 2

# LoRA
config.lora.r = 16
config.lora.lora_alpha = 32
config.lora.lora_dropout = 0.1

# Monitoring
config.monitoring.wandb_project_name = "My Project"
config.monitoring.early_stopping_patience = 3

# Evaluation
config.evaluation.fuzzy_match_threshold = 85.0
```

### Load from JSON

```python
import json
from config import ExperimentConfig

with open("my_config.json") as f:
    config = ExperimentConfig(**json.load(f))
```

### Save to JSON

```python
with open("config.json", "w") as f:
    json.dump(config.model_dump(), f, indent=2)
```

## Configuration Options

<details>
<summary><b>Model Configuration</b></summary>

```python
config.model.base_model = "Qwen/Qwen3-0.6B"
config.model.new_model = "Qwen/Qwen3-0.6B-finetuned"
config.model.torch_dtype = "bfloat16"  # or "float16", "float32"
config.model.attn_implementation = "sdpa"  # or "eager", "flash_attention_2"
```
</details>

<details>
<summary><b>Dataset Configuration</b></summary>

```python
config.dataset.train_dataset_path = "./dataset.jsonl"
config.dataset.test_dataset_path = "./test_dataset.jsonl"
config.dataset.max_seq_length = 1536
config.dataset.train_test_split = 0.1  # 10% for eval
```
</details>

<details>
<summary><b>LoRA Configuration</b></summary>

```python
config.lora.r = 16
config.lora.lora_alpha = 32
config.lora.lora_dropout = 0.1
config.lora.target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```
</details>

<details>
<summary><b>Training Configuration</b></summary>

```python
config.training.num_train_epochs = 4
config.training.learning_rate = 5e-5
config.training.per_device_train_batch_size = 2
config.training.gradient_accumulation_steps = 8
config.training.optimizer = "adamw_8bit"
config.training.bf16 = True
```
</details>

<details>
<summary><b>Generation Configuration</b></summary>

```python
config.generation.max_new_tokens = 256
config.generation.temperature = 0.1
config.generation.top_p = 0.95
config.generation.repetition_penalty = 1.1
```
</details>

## Advanced Usage

### Custom Metrics

Override the `calculate_metrics` method in your evaluator:

```python
class MyEvaluator(Evaluator):
    def calculate_metrics(self, predicted, ground_truth):
        # Your custom metric logic
        return {
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "custom_score": my_score,
        }
```

### Multi-Stage Training

```python
# Stage 1: Quick training
config.training.num_train_epochs = 2
config.training.learning_rate = 1e-4
orchestrator.run()

# Stage 2: Fine-tune with lower LR
config.training.num_train_epochs = 5
config.training.learning_rate = 1e-5
orchestrator.run()
```

### Skip Evaluations

```python
# Skip pre-training eval (faster iteration)
config.evaluation.run_pre_training_eval = False

# Skip post-training eval (if you only want to train)
config.evaluation.run_post_training_eval = False
```

## Benefits Over Original Architecture

| Aspect | Original | Refactored |
|--------|----------|------------|
| **Lines of code** | 1285 lines | ~150 per task |
| **Configuration** | 60+ global constants | Pydantic models |
| **Extensibility** | Hard-coded | Abstract base classes |
| **Type Safety** | None | Full type hints + validation |
| **Reusability** | Monolithic | Modular components |
| **Testability** | Difficult | Easy (unit test each component) |

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture explanation
- **[TASK_EXAMPLE.md](TASK_EXAMPLE.md)** - Step-by-step task creation guide
- **[README.md](README.md)** - Original README (NER task)

## Environment Variables

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers
- PEFT
- TRL
- Pydantic 2.0+
- RapidFuzz (for fuzzy matching)

Install with:
```bash
uv sync
```

## License

Same as original project.

## Contributing

To add a new task:
1. Create `tasks/your_task.py`
2. Implement `DataProcessor` and `Evaluator`
3. Create `train_your_task.py`
4. Submit PR!

See [TASK_EXAMPLE.md](TASK_EXAMPLE.md) for details.

