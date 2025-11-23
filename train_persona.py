"""
Training script for PersonaChat task.

This script demonstrates how to use the modular training system for persona-based conversational AI.
The model learns to generate responses that are consistent with a given persona across multi-turn conversations.
"""

import os
# Force single GPU to prevent multi-GPU deadlocks
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

from core.config import ExperimentConfig
from tasks.personachat_task import PersonaChatDataProcessor, PersonaChatEvaluator
from core.trainer_orchestrator import TrainerOrchestrator


def main():
    """Main entry point for PersonaChat training."""
    
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = ExperimentConfig(
        task_name="personachat",
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
    config.task_name = "personachat"                            # Task name (used for output directories)
    config.random_seed = 42                                     # Random seed for reproducibility
    config.output_dir = "results"                               # Base directory for results
    
    # 2. Model Configuration
    config.model.base_model = "Qwen/Qwen3-0.6B"      # Base model identifier
    config.model.new_model = "Qwen/Qwen3-0.6B-finetuned"         # Name for the fine-tuned model
    config.model.torch_dtype = "bfloat16"                       # Precision: bfloat16 (best for Ampere+), float16 (older GPUs)
    config.model.attn_implementation = "sdpa"                   # Attention: sdpa (memory efficient), eager (fallback)
    
    # 3. Dataset Configuration
    config.dataset.train_dataset_path = "tasks/personachat_train.json"  # Path to training data
    config.dataset.test_dataset_path = "tasks/personachat_train.json"   # Same path = splits into train/eval/test
    config.dataset.max_seq_length = 2048                        # Longer context for multi-turn conversations
    config.dataset.train_test_split = 0.05                      # 5% for eval+test (split 50/50 = 2.5% each ~1600 samples)
    config.dataset.keep_in_memory = False                       # Dataset is large, use disk
    
    # 4. LoRA (Low-Rank Adaptation) Configuration
    config.lora.r = 16                                          # Rank for LoRA adaptation
    config.lora.lora_alpha = 32                                 # Alpha = 2 * r
    config.lora.lora_dropout = 0.05                             # Low dropout for conversational tasks
    config.lora.bias = "none"                                   # Bias handling
    config.lora.task_type = "CAUSAL_LM"                         # Task type
    config.lora.target_modules = [                              # Target all linear layers for best performance
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # 5. Training Hyperparameters
    config.training.num_train_epochs = 2                        # 2 epochs for 64k samples
    config.training.per_device_train_batch_size = 2             # Small batch size for long sequences
    config.training.per_device_eval_batch_size = 2              # Match train batch size
    config.training.gradient_accumulation_steps = 16            # Effective batch size = 2 * 16 = 32
    config.training.learning_rate = 2e-4                        # Standard LR for LoRA
    config.training.lr_scheduler_type = "cosine"                # Cosine schedule with warmup
    config.training.warmup_ratio = 0.03                         # Small warmup for large dataset
    config.training.weight_decay = 0.01                         # Standard weight decay
    config.training.max_grad_norm = 0.3                         # Gradient clipping
    config.training.optimizer = "adamw_8bit"                    # 8-bit optimizer to save memory
    config.training.fp16 = False                                # Disable fp16
    config.training.bf16 = True                                 # Enable bf16 (more stable training)
    config.training.gradient_checkpointing = True               # Critical for saving VRAM with long sequences
    config.training.group_by_length = True                      # Group similar lengths for efficiency
    
    # 6. Monitoring & Logging
    config.monitoring.logging_steps = 50                        # Log every 50 steps
    config.monitoring.eval_steps = 500                          # Evaluate periodically
    config.monitoring.save_steps = 500                          # Save checkpoint with eval
    config.monitoring.save_total_limit = 3                      # Keep last 3 checkpoints
    config.monitoring.early_stopping_patience = 3               # Stop if no improvement
    config.monitoring.early_stopping_threshold = 0.01           # Minimum improvement threshold
    config.monitoring.wandb_project_name = "PersonaChat-Finetuning"  # W&B project name
    config.monitoring.report_to = "wandb"                       # Report to: wandb, tensorboard, or none
    
    # 7. Evaluation Configuration
    config.evaluation.run_pre_training_eval = False             # Skip pre-train eval to save time
    config.evaluation.run_post_training_eval = True             # Essential for final metrics
    config.evaluation.fuzzy_match_threshold = 85.0              # Threshold for semantic similarity (not used directly)
    
    # 8. Generation Configuration (for Inference/Eval)
    config.generation.max_new_tokens = 128                      # Reasonable response length
    config.generation.min_new_tokens = 10                       # Minimum response length
    config.generation.temperature = 0.7                         # Moderate temperature for natural responses
    config.generation.top_p = 0.9                               # Nucleus sampling
    config.generation.repetition_penalty = 1.1                  # Prevent repetitive responses
    config.generation.do_sample = True                          # Enable sampling for diversity
    config.generation.num_beams = 1                             # Use sampling, not beam search
    
    # Initialize data processor (tokenizer will be set by orchestrator)
    data_processor = PersonaChatDataProcessor(tokenizer=None)
    
    # Initialize evaluator (device will be set by orchestrator)
    evaluator = PersonaChatEvaluator(
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

