# Modular Training Framework

A clean, extensible training framework for fine-tuning language models on **any task**.

## Quick Start

```bash
# Install dependencies
uv sync

# Run training
uv run train.py

# Convert to GGUF
uv run scripts/convert_to_gguf.py --quantize Q4_K_M
```

## Architecture

### The 3-Component Design

This framework lets you adapt the same training pipeline to **any task** by implementing just 2 abstract classes:

1. **DataProcessor** - How to format your data
2. **Evaluator** - How to evaluate predictions
3. **TrainerOrchestrator** - Handles everything else (you don't touch this)

```
Your Task Implementation (2 classes)
         ↓
  TrainerOrchestrator
         ↓
   LoRA Fine-Tuning
         ↓
    Best Model Saved
```

## Creating a New Task

### Step 1: Implement DataProcessor

```python
from core.data_processor import DataProcessor

class MyTaskDataProcessor(DataProcessor):
    def parse_example(self, example):
        return {"input": example["text"], "output": example["label"]}
    
    def format_for_training(self, data):
        return [
            {"role": "user", "content": data["input"]},
            {"role": "assistant", "content": data["output"]}
        ]
    
    def format_for_inference(self, data):
        return [{"role": "user", "content": data["input"]}]
```

### Step 2: Implement Evaluator

```python
from core.evaluator import Evaluator

class MyTaskEvaluator(Evaluator):
    def parse_dataset_example(self, example):
        return {
            "context": example["text"],
            "question": None,
            "ground_truth": example["label"]
        }
    
    def generate_prediction(self, model, context, question=None):
        # Format and generate
        formatted = self.data_processor.format_for_inference({"input": context})
        messages = self.tokenizer.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=True
        )
        return self._generate_text(model, messages)
    
    def calculate_metrics(self, predicted, ground_truth):
        # Your metrics (accuracy, F1, BLEU, etc.)
        return {"accuracy": int(predicted == ground_truth)}
    
    def is_valid_prediction(self, pred):
        return pred is not None
    
    def is_schema_valid(self, pred):
        return True
    
    def get_empty_prediction(self):
        return ""
```

### Step 3: Create Entry Point

```python
import os
from dotenv import load_dotenv
from core.config import ExperimentConfig
from core.trainer_orchestrator import TrainerOrchestrator

load_dotenv()

config = ExperimentConfig(
    task_name="my_task",
    output_dir="results",
    hf_token=os.getenv("HF_TOKEN"),
    wandb_api_key=os.getenv("WANDB_API_KEY"),
)

# Configure
config.model.base_model = "Qwen/Qwen3-0.6B"
config.dataset.train_dataset_path = "./data.jsonl"
config.training.num_train_epochs = 3

# Initialize
processor = MyTaskDataProcessor(tokenizer=None)
evaluator = MyTaskEvaluator(config, None, None, processor)

# Run
orchestrator = TrainerOrchestrator(config, processor, evaluator)
orchestrator.run()
```

**That's it!** The orchestrator handles model loading, training, evaluation, and saving automatically.

## Configuration

Key settings (see `train.py` for full config):

```python
config.model.base_model = "Qwen/Qwen3-0.6B"
config.training.num_train_epochs = 3
config.training.learning_rate = 2e-4
config.lora.r = 32
config.monitoring.early_stopping_patience = 3
```

## Model Export & Conversion

### After Training

Models are automatically saved to:
```
results/{task_name}/training_results_{timestamp}/
├── lora_adapter/           # Best LoRA checkpoint
├── final_model/            # Final model
└── evaluation_results.json # Metrics
```

### Convert to GGUF (Automated)

Use the provided script to merge LoRA and convert to GGUF:

```bash
# Basic conversion (F16)
uv run scripts/convert_to_gguf.py

# With quantization (smaller, faster)
uv run scripts/convert_to_gguf.py --quantize Q4_K_M

# Available quantization types:
# Q4_K_M  - 4-bit, good balance (recommended)
# Q5_K_M  - 5-bit, higher quality
# Q8_0    - 8-bit, best quality
```

Output saved to: `best_model/{task_name}/`

### Inference with GGUF

```python
from scripts.inference_gguf import infer

response_text, json_response = infer(
    model_path="best_model/ner/model.gguf",
    system_prompt="Your system prompt",
    report_text="Your input text",
    question="Your question"
)

print(json_response)
```

Or test directly:

```bash
# Test with your data
uv run scripts/test_inference.py

# Custom inference
uv run scripts/inference_gguf.py \
    --model best_model/ner/model.gguf \
    --report "Your text here" \
    --question "What entities are present?"
```

## Project Structure

```
NER-Finetuning/
├── core/
│   ├── config.py              # Configuration models
│   ├── data_processor.py      # Abstract DataProcessor
│   ├── evaluator.py           # Abstract Evaluator
│   ├── metrics.py             # Metric utilities
│   └── trainer_orchestrator.py # Training pipeline
├── tasks/
│   ├── ner_task.py            # Example: NER implementation
│   └── your_task.py           # Your custom task
├── scripts/
│   ├── convert_to_gguf.py     # Automated GGUF conversion
│   ├── inference_gguf.py      # GGUF inference
│   └── test_inference.py      # Test script
├── train.py                   # Training entry point
└── api.py                     # FastAPI inference server
```

## Example Tasks

### NER (Named Entity Recognition)
See `tasks/ner_task.py` for complete implementation

### Question Answering
```python
def format_for_training(self, data):
    prompt = f"Context: {data['context']}\nQuestion: {data['question']}"
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": data["answer"]}
    ]
```

### Text Classification
```python
def format_for_training(self, data):
    return [
        {"role": "user", "content": f"Classify: {data['text']}"},
        {"role": "assistant", "content": data["label"]}
    ]
```

## How It Works

```
1. You define:    DataProcessor + Evaluator
                          ↓
2. Orchestrator:  Loads model, tokenizer, data
                          ↓
3. Training:      LoRA fine-tuning with early stopping
                          ↓
4. Evaluation:    Automatic metrics calculation
                          ↓
5. Saving:        Best model saved automatically
                          ↓
6. Export:        Convert to GGUF for deployment
```

## Environment Setup

Create `.env` file:

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

## Requirements

```bash
uv sync
```

- Python 3.10+
- PyTorch 2.0+
- Transformers, PEFT, TRL
- Pydantic 2.0+

## License

MIT License
