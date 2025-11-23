"""
Training script with modular architecture.

This script demonstrates how to use the modular training system.
To create a new task, simply:
1. Implement DataProcessor for your task
2. Implement Evaluator for your task
3. Configure ExperimentConfig
4. Run the orchestrator
"""

import os
# Force single GPU to prevent multi-GPU deadlocks
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

from core.config import ExperimentConfig
from tasks.ner_task import NERDataProcessor, NERTaskEvaluator
from core.trainer_orchestrator import TrainerOrchestrator


def main():
    """Main entry point for training."""
    
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = ExperimentConfig(
        task_name="ner",
        random_seed=42,
        output_dir="results",
        
        # Set API keys from environment
    hf_token=os.getenv("HF_TOKEN"),
    wandb_api_key=os.getenv("WANDB_API_KEY"),
    )

    # =================================================================================================
    # CONFIGURATION
    # =================================================================================================
    
    # 1. Experiment Setup
    config.task_name = "ner"                                # Task name (used for output directories)
    config.random_seed = 42                                 # Random seed for reproducibility
    config.output_dir = "results"                           # Base directory for results
    
    # 2. Model Configuration
    config.model.base_model = "Qwen/Qwen3-0.6B"             # Base model identifier
    config.model.new_model = "Qwen/Qwen3-0.6B-finetuned"    # Name for the fine-tuned model
    config.model.torch_dtype = "bfloat16"                   # Precision: bfloat16 (best for Ampere+), float16 (older GPUs)
    config.model.attn_implementation = "sdpa"               # Attention: sdpa (memory efficient), eager (fallback)
    
    # 3. Dataset Configuration
    config.dataset.train_dataset_path = "./dataset.jsonl"   # Path to training data
    config.dataset.test_dataset_path = "./test_dataset.jsonl" # Path to test/validation data
    config.dataset.max_seq_length = 1024                    # Reduced to 1024 to save VRAM (sufficient for most NER)
    config.dataset.train_test_split = 0.1                   # 10% for validation (~400 samples)
    config.dataset.keep_in_memory = False                   # Keep dataset in memory (faster but higher RAM)
    
    # 4. LoRA (Low-Rank Adaptation) Configuration
    config.lora.r = 32                                      # Increased rank to 32 for better structured output learning
    config.lora.lora_alpha = 64                             # Alpha = 2 * r
    config.lora.lora_dropout = 0.05                         # Lower dropout for structured tasks
    config.lora.bias = "none"                               # Bias handling
    config.lora.task_type = "CAUSAL_LM"                     # Task type
    config.lora.target_modules = [                          # Target all linear layers for best performance
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # 5. Training Hyperparameters
    config.training.num_train_epochs = 3                    # 3 epochs is usually sufficient for 4k samples
    config.training.per_device_train_batch_size = 4         # Increased batch size (0.6B model is tiny)
    config.training.per_device_eval_batch_size = 4          # Match train batch size
    config.training.gradient_accumulation_steps = 8         # Effective batch size = 4 * 8 = 32
    config.training.learning_rate = 2e-4                    # Higher LR for LoRA (standard is 2e-4)
    config.training.lr_scheduler_type = "cosine"            # Cosine schedule works well
    config.training.warmup_ratio = 0.05                     # Slightly longer warmup
    config.training.weight_decay = 0.01                     # Standard weight decay
    config.training.max_grad_norm = 0.3                     # Gradient clipping
    config.training.optimizer = "adamw_8bit"                # 8-bit optimizer to save memory
    config.training.fp16 = False                            # Disable fp16
    config.training.bf16 = True                             # Enable bf16 (more stable training)
    config.training.gradient_checkpointing = True           # Critical for saving VRAM
    config.training.group_by_length = True                  # Faster training
    
    # 6. Monitoring & Logging
    config.monitoring.logging_steps = 10                    # Log frequently
    config.monitoring.eval_steps = 100                      # Evaluate ~once per epoch (4000/32 ≈ 125 steps)
    config.monitoring.save_steps = 100                      # Save checkpoint with eval
    config.monitoring.save_total_limit = 2                  # Keep last 2 checkpoints
    config.monitoring.early_stopping_patience = 3           # Stop if no improvement
    config.monitoring.early_stopping_threshold = 0.001      # Minimum improvement
    config.monitoring.wandb_project_name = "Fine-Tune NER"  # W&B project name
    config.monitoring.report_to = "wandb"                    # Report to: wandb, tensorboard, or none
    
    # 7. Evaluation Configuration
    config.evaluation.run_pre_training_eval = False         # Skip pre-train eval to save time
    config.evaluation.run_post_training_eval = True         # Essential for final metrics
    config.evaluation.fuzzy_match_threshold = 85.0          # Threshold for fuzzy matching
    
    # 8. Generation Configuration (for Inference/Eval)
    config.generation.max_new_tokens = 512                  # Allow longer outputs for JSON structure
    config.generation.temperature = 0.1                     # Low temperature for consistent structure
    config.generation.top_p = 0.95                          # Nucleus sampling
    config.generation.repetition_penalty = 1.05             # Slight penalty to prevent loops
    config.generation.do_sample = True                      # Sampling enabled
    
    # Initialize data processor (tokenizer will be set by orchestrator)
    data_processor = NERDataProcessor(tokenizer=None)
    
    # Initialize evaluator (device will be set by orchestrator)
    evaluator = NERTaskEvaluator(
        config=config,
        tokenizer=None,
        device=None,
        data_processor=data_processor
    )
    
    # Create orchestrator
    orchestrator = TrainerOrchestrator(
        config=config,
        data_processor=data_processor,
        evaluator=evaluator
    )
    
    # Run full pipeline
    results_folder = orchestrator.run()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_folder}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

