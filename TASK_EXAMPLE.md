# Example: Creating a New Task (Question Answering)

This guide shows step-by-step how to create a new task using the modular architecture.

## Scenario: Question Answering Task

Let's create a simple question-answering task where:
- **Input:** A context paragraph + a question
- **Output:** A text answer
- **Evaluation:** Exact match and F1 score on tokens

## Step 1: Create Task File

Create `tasks/qa_task.py`:

```python
"""Question Answering task implementation."""

import torch
from typing import Dict, Any, List, Tuple
import logging

from data_processor import DataProcessor
from evaluator import Evaluator


logger = logging.getLogger(__name__)


class QADataProcessor(DataProcessor):
    """Data processor for QA task."""
    
    def parse_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse a QA example.
        
        Expected format:
        {
            "context": "The context paragraph...",
            "question": "What is...?",
            "answer": "The answer"
        }
        """
        return [{
            "context": example.get("context", ""),
            "question": example.get("question", ""),
            "answer": example.get("answer", "")
        }]
    
    def format_for_training(self, parsed_data: Dict[str, Any]) -> str:
        """Format QA pair for training."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the given context."
            },
            {
                "role": "user",
                "content": f"Context: {parsed_data['context']}\n\nQuestion: {parsed_data['question']}"
            },
            {
                "role": "assistant",
                "content": parsed_data["answer"]
            }
        ]
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False)
    
    def format_for_inference(self, input_data: Dict[str, Any]) -> str:
        """Format data for inference."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the given context."
            },
            {
                "role": "user",
                "content": f"Context: {input_data['context']}\n\nQuestion: {input_data['question']}"
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


class QATaskEvaluator(Evaluator):
    """Evaluator for QA task."""
    
    def __init__(self, config, tokenizer, device, data_processor: QADataProcessor):
        super().__init__(config, tokenizer, device)
        self.data_processor = data_processor
    
    def parse_dataset_example(self, example: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Parse test example.
        
        Returns:
            (context, [(question, answer), ...])
        """
        context = example.get("context", "")
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        return context, [(question, answer)]
    
    def generate_prediction(self, model, context: str, question: str) -> Tuple[str, str]:
        """Generate answer prediction."""
        prompt = self.data_processor.format_for_inference({
            "context": context,
            "question": question
        })
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_token_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        new_tokens = outputs[0, input_token_length:]
        predicted_answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        return predicted_answer, predicted_answer
    
    def calculate_metrics(self, predicted: str, ground_truth: str) -> Dict[str, Any]:
        """
        Calculate token-level F1 score.
        
        Returns:
            Dict with tp, fp, fn (required by base class)
        """
        pred_tokens = set(predicted.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        tp = len(pred_tokens & gt_tokens)
        fp = len(pred_tokens - gt_tokens)
        fn = len(gt_tokens - pred_tokens)
        
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    def is_valid_prediction(self, prediction: str) -> bool:
        """Check if prediction is a non-empty string."""
        return isinstance(prediction, str) and len(prediction.strip()) > 0
    
    def is_schema_valid(self, prediction: str) -> bool:
        """For text output, same as is_valid_prediction."""
        return self.is_valid_prediction(prediction)
    
    def get_empty_prediction(self) -> str:
        """Return empty string as fallback."""
        return ""
    
    def check_exact_match(self, predicted: str, ground_truth: str) -> bool:
        """Check exact match (case-insensitive, whitespace normalized)."""
        pred_normalized = " ".join(predicted.lower().split())
        gt_normalized = " ".join(ground_truth.lower().split())
        return pred_normalized == gt_normalized
```

## Step 2: Update Task Registry

Edit `tasks/__init__.py`:

```python
"""Task-specific implementations."""

from .ner_task import NERDataProcessor, NERTaskEvaluator
from .qa_task import QADataProcessor, QATaskEvaluator

__all__ = [
    "NERDataProcessor",
    "NERTaskEvaluator",
    "QADataProcessor",
    "QATaskEvaluator",
]
```

## Step 3: Create Training Script

Create `train_qa.py`:

```python
"""Train model on Question Answering task."""

import os
from dotenv import load_dotenv

from config import ExperimentConfig
from tasks.qa_task import QADataProcessor, QATaskEvaluator
from trainer_orchestrator import TrainerOrchestrator


def main():
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = ExperimentConfig(
        random_seed=42,
        output_dir="results_qa",
        hf_token=os.getenv("HF_TOKEN"),
        wandb_api_key=os.getenv("WANDB_API_KEY"),
    )
    
    # Configure for QA task
    config.model.base_model = "Qwen/Qwen3-0.6B"
    config.model.new_model = "Qwen/Qwen3-0.6B-qa-finetuned"
    config.dataset.train_dataset_path = "./qa_train.jsonl"
    config.dataset.test_dataset_path = "./qa_test.jsonl"
    config.monitoring.wandb_project_name = "QA Fine-Tuning"
    
    # QA-specific settings
    config.training.num_train_epochs = 3
    config.generation.max_new_tokens = 128  # Shorter answers
    
    # Initialize components
    data_processor = QADataProcessor(tokenizer=None)
    evaluator = QATaskEvaluator(
        config=config,
        tokenizer=None,
        device=None,
        data_processor=data_processor
    )
    
    # Run training
    orchestrator = TrainerOrchestrator(
        config=config,
        data_processor=data_processor,
        evaluator=evaluator
    )
    
    results_folder = orchestrator.run()
    
    print(f"\n{'='*80}")
    print("QA TRAINING COMPLETED!")
    print(f"Results: {results_folder}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
```

## Step 4: Prepare Data

Create your dataset files in JSONL format:

**qa_train.jsonl:**
```jsonl
{"context": "Paris is the capital of France.", "question": "What is the capital of France?", "answer": "Paris"}
{"context": "The Eiffel Tower is in Paris.", "question": "Where is the Eiffel Tower?", "answer": "Paris"}
```

**qa_test.jsonl:**
```jsonl
{"context": "London is the capital of the UK.", "question": "What is the capital of the UK?", "answer": "London"}
```

## Step 5: Run Training

```bash
uv run train_qa.py
```

## Step 6: Customize Further (Optional)

### Add Custom Metrics

You can add custom metrics by overriding the `log_summary` method:

```python
class QATaskEvaluator(Evaluator):
    # ... other methods ...
    
    def log_summary(self, summary: Dict):
        """Log QA-specific metrics."""
        super().log_summary(summary)  # Call base implementation
        
        # Add custom metrics
        logger.info("\nQA-Specific Metrics:")
        logger.info(f"Average answer length: {summary.get('avg_answer_length', 0):.1f} tokens")
```

### Use Different Models

```python
# Try different models
config.model.base_model = "meta-llama/Llama-3.2-1B"
config.model.base_model = "microsoft/phi-2"
config.model.base_model = "mistralai/Mistral-7B-v0.1"
```

### Adjust LoRA Settings

```python
# For larger models, increase LoRA rank
config.lora.r = 64
config.lora.lora_alpha = 128

# Target more modules
config.lora.target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "lm_head"  # Also adapt output layer
]
```

### Load Config from File

```python
import json
from config import ExperimentConfig

# Load base config
with open("qa_config.json") as f:
    config_dict = json.load(f)

config = ExperimentConfig(**config_dict)
```

**qa_config.json:**
```json
{
  "random_seed": 42,
  "output_dir": "results_qa",
  "model": {
    "base_model": "Qwen/Qwen3-0.6B",
    "new_model": "Qwen/Qwen3-0.6B-qa-finetuned"
  },
  "training": {
    "num_train_epochs": 3,
    "learning_rate": 0.0001
  }
}
```

## Summary

With the modular architecture, creating a new task requires:

1. ✅ Implement `DataProcessor` (parsing and formatting)
2. ✅ Implement `Evaluator` (inference and metrics)
3. ✅ Create entry point script
4. ✅ Run training!

The core infrastructure handles everything else:
- Model loading
- Training loop
- Memory management
- Checkpointing
- W&B logging
- Results saving



## Next Steps

- Add more tasks (summarization, classification, etc.)
- Create task registry for auto-discovery
- Add support for multiple evaluation metrics
- Build ensemble evaluators
- Add hyperparameter tuning support

